from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from uuid import UUID

import cv2
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.types import ASGIApp, Receive, Scope, Send

from . import (
    ALGORITHM_VERSION,
    CONTRACT_VERSION,
    MINIMUM_INTEGRATION_VERSION,
    __version__,
)
from .calibration import from_farmbot_calibration
from .calibration_store import CalibrationStore, FarmbotCalibrationInput
from .curve_edit import propose_curve_point
from .curves import fit_monotonic_curve
from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError, StaleRadiusError
from .jobs import JobManager
from .models import (
    ApplyRadiusRequest,
    ApplyRemovalRequest,
    Calibration,
    InventoryRequest,
    Measurement,
    OperatingMode,
    OriginLocation,
    UpsertCurveRequest,
    VisionImageRequest,
)
from .settings import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
settings = Settings.load()
database = Database(settings.data_dir / "farmbot_vision.db")
calibration_store = CalibrationStore(settings.data_dir / "farmbot_calibration.json")
client = HomeAssistantClient()
jobs = JobManager(settings, database, client)


def _calibration_from_input(entry_id: str, values: FarmbotCalibrationInput) -> Calibration:
    """Build a processed-resolution calibration from stored FarmBot inputs."""
    resolution = settings.resolution
    return from_farmbot_calibration(
        coordinate_scale_mm_per_px=values.coordinate_scale,
        reference_width=values.reference_width,
        reference_height=values.reference_height,
        processed_width=resolution.width,
        processed_height=resolution.height,
        rotation_degrees=values.rotation_degrees,
        offset_x_mm=values.offset_x_mm,
        offset_y_mm=values.offset_y_mm,
        origin_location=values.origin_location,
        uncertainty_mm=settings.calibration_uncertainty_mm,
        analysis_resolution=resolution.value,
    )


def seed_calibration_from_store() -> None:
    """Restore the active DB calibration from the durable /data store on boot.

    The store is the master record of the FarmBot calibration the user entered;
    the SQLite active calibration is the runtime source the analysis pipeline
    reads. If a bot has a stored calibration but no active DB calibration (fresh
    container, wiped DB), re-derive and persist it so a restart never loses
    calibration and never requires re-entry.
    """
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        return
    stored = calibration_store.get(entry_id)
    if stored is None or database.active_calibration(entry_id) is not None:
        return
    try:
        database.save_calibration(entry_id, _calibration_from_input(entry_id, stored))
        LOGGER.info("Restored calibration for %s from the /data store", entry_id)
    except ValueError as exc:
        LOGGER.warning("Could not restore stored calibration for %s: %s", entry_id, exc)


def _normalize_leading_slashes(value: str) -> str:
    """Collapse only duplicate slashes at the beginning of an ASGI path."""

    return f"/{value.lstrip('/')}" if value.startswith("//") else value


class NormalizeIngressPathMiddleware:
    """Normalize duplicate leading slashes before FastAPI route matching."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in {"http", "websocket"}:
            path = scope.get("path", "")
            raw_path = scope.get("raw_path", b"")
            normalized_path = _normalize_leading_slashes(path)
            normalized_raw_path = (
                b"/" + raw_path.lstrip(b"/") if raw_path.startswith(b"//") else raw_path
            )
            if normalized_path != path or normalized_raw_path != raw_path:
                scope = dict(scope)
                scope["path"] = normalized_path
                scope["raw_path"] = normalized_raw_path
        await self.app(scope, receive, send)


# FarmBot-style composite calibration view. One photo-row (images sharing an X
# coordinate) is stitched in garden-coordinate space using the FarmBot camera
# calibration, with plant and weed centres overlaid so alignment across the
# whole row can be verified at once. Vanilla JS on a canvas -- no frontend build
# toolchain (Part 5). The rotation direction here MUST match
# vision.ROTATION_SIGN and vision.garden_to_pixel.
_CALIBRATION_JS = r"""
(function(){
  const ROT_SIGN=1;            // matches vision.ROTATION_SIGN
  const MAX_CANVAS=2400;       // cap composite dimensions to bound memory
  const canvas=document.getElementById('canvas');
  const ctx=canvas.getContext('2d');
  const rowSel=document.getElementById('row');
  const status=document.getElementById('status');
  const ppmEl=document.getElementById('ppm');
  let scene={images:[],plants:[],weeds:[]}, rows=[], current=null, pending=false;

  function entry(){return document.getElementById('entry_id').value.trim();}
  function num(id){return parseFloat(document.getElementById(id).value)||0;}
  function checked(id){return document.getElementById(id).checked;}
  function origin(){return document.getElementById('origin').value;}
  function originSigns(o){
    return [(o==='top_right'||o==='bottom_right')?-1:1,
            (o==='bottom_left'||o==='bottom_right')?-1:1];
  }
  // FarmBot calibration inputs, or null when incomplete.
  function params(){
    const scale=num('fb_scale'), refw=num('fb_refw'), refh=num('fb_refh');
    if(!(scale>0&&refw>0&&refh>0)) return null;
    const s=originSigns(origin());
    return {scale:scale,refw:refw,refh:refh,sx:s[0],sy:s[1],
            rot:num('rotation')*Math.PI/180*ROT_SIGN,ox:num('offx'),oy:num('offy')};
  }
  // Pixels-per-mm of one processed image (its own natural size) under p.
  function imagePpm(p,iw,ih){return [(1/p.scale)*iw/p.refw,(1/p.scale)*ih/p.refh];}
  // Map a source pixel (u,v) of an image taken at (cx,cy) to a garden coord.
  // Inverse of vision.garden_to_pixel.
  function pixelToCoord(p,cx,cy,iw,ih,u,v){
    const ppm=imagePpm(p,iw,ih);
    const rx=u-iw/2, ry=v-ih/2;
    const c=Math.cos(p.rot), s=Math.sin(p.rot);
    const vx=c*rx - s*ry, vy=s*rx + c*ry;
    return [cx + vx/(p.sx*ppm[0]), cy + vy/(p.sy*ppm[1])];
  }
  // Group images into rows by shared X (within tolerance, mm).
  function buildRows(images,tol){
    const imgs=images.filter(im=>isFinite(im.x)&&isFinite(im.y)).slice()
                     .sort((a,b)=>a.x-b.x);
    const out=[]; let cur=null;
    imgs.forEach(im=>{
      if(!cur||Math.abs(im.x-cur.x)>tol){cur={x:im.x,sum:im.x,images:[im]};out.push(cur);}
      else{cur.images.push(im);cur.sum+=im.x;cur.x=cur.sum/cur.images.length;}
    });
    out.forEach(r=>r.images.sort((a,b)=>a.y-b.y));
    return out;
  }
  function populateRows(){
    rows=buildRows(scene.images,num('rowtol')||50);
    rowSel.innerHTML='';
    rows.forEach((r,i)=>{
      const o=document.createElement('option');
      o.value=i;
      o.textContent='X≈'+Math.round(r.x)+' mm ('+r.images.length+' photos)';
      rowSel.appendChild(o);
    });
  }
  function selectRow(){
    const p=params();
    ppmEl.textContent=p?('Pixels per mm (at analysis res): scale '+p.scale+' mm/px'):
                        'Enter the FarmBot pixel coordinate scale, and measured-at width/height';
    const row=rows[+rowSel.value];
    if(!row){current=null;clearCanvas('Load a bot, then pick a photo row');return;}
    current={images:[]};
    status.textContent='Loading '+row.images.length+' photos…';
    row.images.forEach(im=>{
      const image=new Image();
      const rec={info:im,img:image,loaded:false};
      current.images.push(rec);
      image.onload=function(){rec.loaded=true;render();};
      image.onerror=function(){status.textContent='Could not load image #'+im.id;};
      image.src='api/vision/image/'+im.id+'.jpg?entry_id='+encodeURIComponent(entry());
    });
  }
  function clearCanvas(msg){
    canvas.width=640;canvas.height=200;
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#888';ctx.font='14px sans-serif';ctx.fillText(msg||'',12,28);
  }
  function scheduleRender(){
    if(pending) return; pending=true;
    requestAnimationFrame(function(){pending=false;render();});
  }
  function render(){
    const p=params();
    document.getElementById('save').disabled=!(p&&checked('confirm'));
    if(!current){return;}
    const ready=current.images.filter(r=>r.loaded&&r.img.naturalWidth>0);
    if(!p){clearCanvas('Enter FarmBot calibration values to build the composite');return;}
    if(!ready.length){return;}
    // Garden-space bounding box from every image's four corners.
    let gxmin=Infinity,gxmax=-Infinity,gymin=Infinity,gymax=-Infinity,ppmSum=0;
    ready.forEach(r=>{
      const iw=r.img.naturalWidth, ih=r.img.naturalHeight;
      const pp=imagePpm(p,iw,ih); ppmSum+=(pp[0]+pp[1])/2;
      [[0,0],[iw,0],[0,ih],[iw,ih]].forEach(c=>{
        const g=pixelToCoord(p,r.info.x,r.info.y,iw,ih,c[0],c[1]);
        gxmin=Math.min(gxmin,g[0]);gxmax=Math.max(gxmax,g[0]);
        gymin=Math.min(gymin,g[1]);gymax=Math.max(gymax,g[1]);
      });
    });
    let P=ppmSum/ready.length;
    const rangeX=Math.max(1,gxmax-gxmin), rangeY=Math.max(1,gymax-gymin);
    P=Math.min(P,MAX_CANVAS/rangeX,MAX_CANVAS/rangeY);
    canvas.width=Math.max(1,Math.round(rangeX*P));
    canvas.height=Math.max(1,Math.round(rangeY*P));
    const toCanvas=function(gx,gy){return [(gx-gxmin)*P,(gy-gymin)*P];};
    ctx.setTransform(1,0,0,1,0,0);
    ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);
    // Paint each image via the affine that maps its source pixels into the
    // composite (three mapped points fully determine the affine).
    ctx.imageSmoothingEnabled=true;
    ready.forEach(r=>{
      const iw=r.img.naturalWidth, ih=r.img.naturalHeight;
      const p0=toCanvas.apply(null,pixelToCoord(p,r.info.x,r.info.y,iw,ih,0,0));
      const pu=toCanvas.apply(null,pixelToCoord(p,r.info.x,r.info.y,iw,ih,iw,0));
      const pv=toCanvas.apply(null,pixelToCoord(p,r.info.x,r.info.y,iw,ih,0,ih));
      ctx.setTransform((pu[0]-p0[0])/iw,(pu[1]-p0[1])/iw,
                       (pv[0]-p0[0])/ih,(pv[1]-p0[1])/ih,p0[0],p0[1]);
      ctx.drawImage(r.img,0,0);
    });
    ctx.setTransform(1,0,0,1,0,0);
    if(checked('showoverlay')) drawOverlay(p,toCanvas,P);
    status.textContent='Row composite: '+ready.length+' photos, '
      +scene.plants.length+' plants, '+scene.weeds.length+' weeds. '
      +'Confirm centres sit on their plants across the row.';
  }
  function marker(p,toCanvas,P,pt,colour,label){
    // Offset shifts a point's projected position exactly as garden_to_pixel does.
    const c=toCanvas(pt.x+p.ox,pt.y+p.oy);
    if(c[0]<-40||c[1]<-40||c[0]>canvas.width+40||c[1]>canvas.height+40) return;
    ctx.strokeStyle=colour;ctx.fillStyle=colour;ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(c[0],c[1],Math.max(4,(pt.radius||0)*P),0,7);ctx.stroke();
    ctx.beginPath();ctx.arc(c[0],c[1],2.5,0,7);ctx.fill();
    if(label&&checked('showlabels')){
      ctx.font='12px sans-serif';
      ctx.fillText(label,c[0]+5,c[1]-5);
    }
  }
  function drawOverlay(p,toCanvas,P){
    scene.plants.forEach(pl=>marker(p,toCanvas,P,pl,'#2ecc40',
      (pl.name||('#'+pl.id))+(pl.slug?(' ('+pl.slug+')'):'')));
    scene.weeds.forEach(w=>marker(p,toCanvas,P,w,'#ff4136',w.name||'Weed'));
  }

  document.getElementById('load').addEventListener('click',async function(){
    status.textContent='Loading inventory…';
    try{
      const r=await fetch('api/vision/images?entry_id='+encodeURIComponent(entry()));
      if(!r.ok) throw new Error('HTTP '+r.status);
      scene=await r.json();
      scene.images=scene.images||[];scene.plants=scene.plants||[];scene.weeds=scene.weeds||[];
      populateRows();
      status.textContent=scene.images.length+' images, '+rows.length+' rows, '
        +scene.plants.length+' plants, '+scene.weeds.length+' weeds';
      if(rows.length) selectRow(); else clearCanvas('No images with coordinates found');
    }catch(err){status.textContent='Could not load inventory: '+err.message;}
  });
  rowSel.addEventListener('change',selectRow);
  document.getElementById('rowtol').addEventListener('input',function(){
    populateRows();selectRow();
  });
  ['fb_scale','fb_refw','fb_refh','rotation','origin','offx','offy'].forEach(function(id){
    document.getElementById(id).addEventListener('input',scheduleRender);
    document.getElementById(id).addEventListener('change',scheduleRender);
  });
  ['showoverlay','showlabels','confirm'].forEach(function(id){
    document.getElementById(id).addEventListener('change',scheduleRender);
  });
  document.getElementById('save').addEventListener('click',function(){
    const p=params();
    if(!p){status.textContent='Enter the FarmBot calibration values first';return;}
    const f=document.createElement('form');f.method='post';f.action='calibration';
    const fields={entry_id:entry(),coordinate_scale:num('fb_scale'),
      reference_width:num('fb_refw'),reference_height:num('fb_refh'),
      rotation:num('rotation'),origin_location:origin(),
      offset_x:num('offx'),offset_y:num('offy')};
    for(const k in fields){const i=document.createElement('input');i.type='hidden';
      i.name=k;i.value=fields[k];f.appendChild(i);}
    document.body.appendChild(f);f.submit();
  });
  clearCanvas('Load a bot, then pick a photo row');
})();
"""

_DASHBOARD_JS = r"""
(function(){
  const modal=document.getElementById('overlay-modal');
  const modalImg=document.getElementById('overlay-modal-img');
  const modalDetails=document.getElementById('overlay-modal-details');
  const closeButton=document.getElementById('overlay-modal-close');
  const counter=document.getElementById('overlay-modal-counter');
  let artifacts=[], index=0, returnFocus=null;
  function showArtifact(){
    if(!artifacts.length) return;
    modalImg.src=artifacts[index];
    counter.textContent=(index+1)+' / '+artifacts.length;
  }
  function closeModal(){
    modal.hidden=true; modalImg.removeAttribute('src');
    if(returnFocus) returnFocus.focus();
  }
  document.addEventListener('click',async function(event){
    const viewer=event.target.closest('[data-artifacts]');
    if(viewer){
      try{artifacts=JSON.parse(viewer.dataset.artifacts||'[]');}catch(_){artifacts=[];}
      if(!artifacts.length) return;
      index=0; returnFocus=viewer;
      let details={}; try{details=JSON.parse(viewer.dataset.details||'{}');}catch(_){}
      modalDetails.textContent=details.formula||'';
      modal.hidden=false; showArtifact(); closeButton.focus(); return;
    }
    const action=event.target.closest('.review-action');
    if(action){
      event.preventDefault();
      const row=action.closest('.review-item');
      const message=row&&row.querySelector('.action-message');
      action.disabled=true;
      try{
        const response=await fetch(action.dataset.url,{method:'POST',headers:{Accept:'application/json'}});
        const result=await response.json();
        const explicitReject=/\/(reject|keep)$/.test(action.dataset.url);
        if(response.ok&&(result.status==='applied'||(result.status==='rejected'&&explicitReject))) row.remove();
        else if(message) message.textContent=result.message||('HTTP '+response.status);
      }catch(error){if(message) message.textContent='Request failed: '+error.message;}
      finally{action.disabled=false;}
      return;
    }
    const curveAction=event.target.closest('.curve-action');
    if(curveAction){
      event.preventDefault();
      const row=curveAction.closest('.review-item');
      const message=row.querySelector('.action-message');
      const data=new FormData();
      if(curveAction.dataset.action==='apply'){
        const input=row.querySelector('.curve-value'); data.append('value',input.value);
        if(!window.confirm('Apply this curve value? Flagged values override the automatic gate.')) return;
        data.append('confirm_override','true');
      }
      curveAction.disabled=true;
      try{
        const response=await fetch(curveAction.dataset.url,{method:'POST',headers:{Accept:'application/json'},body:data});
        const result=await response.json();
        if(response.ok&&(result.status==='applied'||result.status==='rejected')) row.remove();
        else message.textContent=result.message||('HTTP '+response.status);
      }catch(error){message.textContent='Request failed: '+error.message;}
      finally{curveAction.disabled=false;}
    }
  });
  closeButton.addEventListener('click',closeModal);
  modal.addEventListener('click',function(event){if(event.target===modal) closeModal();});
  document.getElementById('overlay-modal-prev').addEventListener('click',function(){
    index=(index-1+artifacts.length)%artifacts.length;showArtifact();
  });
  document.getElementById('overlay-modal-next').addEventListener('click',function(){
    index=(index+1)%artifacts.length;showArtifact();
  });
  document.addEventListener('keydown',function(event){if(event.key==='Escape'&&!modal.hidden)closeModal();});
})();
"""


async def event_listener() -> None:
    async for event in client.vision_events():
        # Await each automatic request so photos cannot be silently discarded
        # merely because the previous image is still being analysed.
        await jobs.run(
            entry_id=event.config_entry_id,
            mode=OperatingMode(event.mode) if event.mode is not None else settings.mode,
            plant_ids=event.plant_ids,
            image_ids=[event.image_id] if event.image_id is not None else None,
            trigger="new_image" if event.image_id is not None else "event",
            queue_if_busy=True,
        )


async def heartbeat() -> None:
    while True:
        if settings.selected_config_entry_id:
            if jobs.lock.locked():
                try:
                    job_id = UUID(str(jobs.current.get("id")))
                except (TypeError, ValueError):
                    job_id = None
                await jobs._status(
                    settings.selected_config_entry_id,
                    job_id,
                    "running",
                    str(jobs.current.get("progress") or "analysing")[:240],
                )
            else:
                await jobs._status(settings.selected_config_entry_id, None, "idle", "ready")
        # Older installations may retain the former 15-minute option. Cap the
        # effective interval so they also stay inside the integration's
        # ten-minute availability window after upgrading.
        await asyncio.sleep(min(settings.heartbeat_minutes, 5) * 60)


async def resolve_config_entry() -> None:
    """Select the only loaded FarmBot automatically when no ID was configured."""
    if settings.selected_config_entry_id:
        return
    try:
        bots = (await client.list_bots()).bots
    except HomeAssistantError as exc:
        LOGGER.warning("Could not discover FarmBot config entries at startup: %s", exc)
        return
    if len(bots) == 1:
        settings.selected_config_entry_id = bots[0].config_entry_id
        LOGGER.info(
            "Automatically selected the only loaded FarmBot config entry: %s",
            settings.selected_config_entry_id,
        )
    elif len(bots) > 1:
        LOGGER.warning(
            "Multiple FarmBots are loaded; select one in the add-on options to enable heartbeats"
        )


async def scheduler() -> None:
    last_run_date = None
    while True:
        now = datetime.now().astimezone()
        if (
            settings.schedule_enabled
            and settings.selected_config_entry_id
            and now.strftime("%H:%M") == settings.schedule_time
            and now.date() != last_run_date
            and database.active_calibration(settings.selected_config_entry_id)
        ):
            last_run_date = now.date()
            await jobs.run(trigger="schedule")
        await asyncio.sleep(30)


async def retention_cleanup() -> None:
    while True:
        artifacts = settings.data_dir / "artifacts"
        now = datetime.now().astimezone()
        if artifacts.exists():
            for path in artifacts.glob("*"):
                days = (
                    settings.successful_mask_retention_days
                    if path.name.endswith("-mask.png")
                    else settings.diagnostic_retention_days
                )
                cutoff = now - timedelta(days=days)
                if datetime.fromtimestamp(path.stat().st_mtime).astimezone() < cutoff:
                    path.unlink(missing_ok=True)
        await asyncio.sleep(6 * 60 * 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await resolve_config_entry()
    LOGGER.info(
        "FarmBot Vision %s starting: selected_config_entry_id=%s mode=%s analysis_resolution=%s",
        __version__,
        settings.selected_config_entry_id or "(not set)",
        settings.mode.value,
        settings.resolution.label,
    )
    if not settings.selected_config_entry_id:
        LOGGER.warning(
            "No FarmBot config entry ID configured; scheduled/heartbeat status reports and "
            "the calibration page will not work until one is set in the add-on options"
        )
    seed_calibration_from_store()
    tasks = [
        asyncio.create_task(event_listener(), name="event_listener"),
        asyncio.create_task(heartbeat(), name="heartbeat"),
        asyncio.create_task(scheduler(), name="scheduler"),
        asyncio.create_task(retention_cleanup(), name="retention_cleanup"),
    ]

    def _log_task_failure(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            LOGGER.error(
                "Background task %s crashed and will not restart: %s",
                task.get_name(),
                exc,
                exc_info=exc,
            )

    for task in tasks:
        task.add_done_callback(_log_task_failure)
    yield
    for task in tasks:
        task.cancel()
    await client.close()


app = FastAPI(
    title="FarmBot Vision", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
)
app.add_middleware(NormalizeIngressPathMiddleware)


def ingress_base(request: Request) -> str:
    value = request.headers.get("X-Ingress-Path", "./").strip()
    if value in {"", ".", "./"}:
        return "./"
    value = _normalize_leading_slashes(value).rstrip("/")
    return f"{value}/"


def layout(request: Request, body: str, title: str = "FarmBot Vision") -> HTMLResponse:
    base = escape(ingress_base(request), quote=True)
    return HTMLResponse(f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><base href="{base}">
<title>{escape(title)}</title><style>
:root{{--green:#52b788;--dark:#17221b;--muted:#74817a}}*{{box-sizing:border-box}}
body{{font:15px system-ui;margin:0;background:#f3f7f4;color:var(--dark)}}header{{background:#173f2c;color:white;padding:1rem 4vw}}
main{{max-width:1100px;margin:auto;padding:1.2rem}}nav a{{color:white;margin-right:1rem}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem}}
.card{{background:white;border-radius:10px;padding:1rem;box-shadow:0 1px 4px #0002;overflow:auto}}table{{width:100%;border-collapse:collapse}}td,th{{padding:.5rem;text-align:left;border-bottom:1px solid #ddd}}
button{{background:var(--green);border:0;border-radius:6px;padding:.65rem 1rem;cursor:pointer}}.warn{{color:#9b4b00}}.muted{{color:var(--muted)}}input,select{{padding:.5rem;max-width:100%}}img{{max-width:100%}}
.action-message{{display:block;color:#a40000;max-width:24rem}}.overlay-modal[hidden]{{display:none}}
.overlay-modal{{position:fixed;inset:0;z-index:1000;background:#000b;display:flex;align-items:center;justify-content:center;padding:1rem}}
.overlay-modal figure{{position:relative;background:white;border-radius:10px;margin:0;padding:1rem;max-width:min(95vw,1000px);max-height:95vh;overflow:auto}}
.overlay-modal img{{display:block;max-height:70vh;margin:auto}}.modal-close{{position:absolute;right:.5rem;top:.5rem;font-size:1.5rem}}
.modal-controls{{display:flex;gap:.5rem;align-items:center;justify-content:center;margin-top:.6rem}}.legend{{font-size:.9rem;color:var(--muted)}}
</style></head><body><header><h1>🌱 FarmBot Vision</h1><nav><a href="./">Dashboard</a><a href="settings">Calibration</a><a href="api/health">Health JSON</a></nav></header>
<main>{body}</main></body></html>""")


@app.get("/health")
@app.get("/api/health")
async def health() -> JSONResponse:
    artifacts = settings.data_dir / "artifacts"
    artifact_bytes = (
        sum(p.stat().st_size for p in artifacts.glob("*") if p.is_file())
        if artifacts.exists()
        else 0
    )
    resolution = settings.resolution
    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "algorithm_version": ALGORITHM_VERSION,
            "contract_version": CONTRACT_VERSION,
            "minimum_integration_version": MINIMUM_INTEGRATION_VERSION,
            "opencv_threads": cv2.getNumThreads(),
            "analysis_resolution": resolution.value,
            "analysis_width": resolution.width,
            "analysis_height": resolution.height,
            "analysis_pixels": resolution.pixel_count,
            "relative_workload": resolution.relative_workload,
            "job": jobs.current,
            "last_job": jobs.last,
            "database": database.stats(),
            "artifact_bytes": artifact_bytes,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    rows = database.pending_measurements()
    crop_slugs = sorted({row["crop_slug"] for row in rows})
    curves = {
        slug: fit_monotonic_curve(
            database.measurements_for_crop(slug), safety_margin_mm=settings.safety_margin_mm
        )
        for slug in crop_slugs
    }

    def _artifact_button(r: dict) -> str:
        paths = r.get("artifact_paths") or []
        if not paths and r.get("overlay_path"):
            paths = [r["overlay_path"]]
        urls = [f"artifact/{Path(path).name}" for path in paths if path]
        if not urls:
            return "<span class=muted>None</span>"
        uncertainty = float(r.get("calibration_uncertainty_mm") or 0)
        details = {
            "formula": (
                f"Current {r['current_radius_mm']:.1f} mm; typical "
                f"{r['typical_canopy_radius_mm']:.1f} mm; maximum "
                f"{r['maximum_accepted_canopy_radius_mm']:.1f} mm. Recommended = maximum + "
                f"safety {float(r.get('safety_margin_mm') or 0):.1f} + calibration "
                f"uncertainty {uncertainty:.1f} = {r['recommended_protection_radius_mm']:.1f} mm. "
                "Legend: green = vegetation; palette = plant ownership; red = ambiguous; "
                "cyan/plant colour/white = typical/maximum/protected geometry."
            )
        }
        artifacts_json = escape(json.dumps(urls, separators=(",", ":")), quote=True)
        details_json = escape(json.dumps(details, separators=(",", ":")), quote=True)
        return (
            f'<button type=button data-artifacts="{artifacts_json}" '
            f'data-details="{details_json}">View</button>'
        )

    def _review_controls(r: dict) -> str:
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if not r.get("calibrated", 1):
            return "<span class=warn>Calibration required</span>"
        if r["decision"] != "recommended":
            return "<span class=muted>Not reviewable</span>"
        return (
            f'<form method=post action="recommendations/{r["measurement_id"]}/approve">'
            f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/approve">Approve</button></form>'
            f'<form method=post action="recommendations/{r["measurement_id"]}/reject">'
            f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/reject">Reject</button></form>'
            '<small class=action-message></small>'
        )

    measurement_rows = "".join(
        f'<tr class=review-item id="measurement-{r["measurement_id"]}"><td>{r["plant_id"]}</td><td>{escape(r["crop_slug"])}</td>'
        f"<td>{escape(str(r.get('processed_width') or '—'))}x{escape(str(r.get('processed_height') or '—'))}</td>"
        f"<td>{r['current_radius_mm']:.1f}</td>"
        f"<td>{r['maximum_accepted_canopy_radius_mm']:.1f}</td><td>{r['recommended_protection_radius_mm']:.1f}</td>"
        f"<td>{r['confidence']:.2f}</td><td>{escape(str(r.get('calibration_source') or '—'))}</td>"
        f"<td>{escape(r['decision'])}</td><td>{escape(r['reason'])}</td>"
        f"<td>{_artifact_button(r)}</td>"
        f"<td>{_review_controls(r)}</td></tr>"
        for r in rows
        if not r.get("vegetation_absent")
    )
    last = jobs.last
    curve_rows = "".join(
        f"<tr><td>{escape(slug)}</td><td>{escape(str(curve))}</td><td>diameter mm</td></tr>"
        for slug, curve in curves.items()
    )
    removal_rows = "".join(
        f'<tr class=review-item id="measurement-{r["measurement_id"]}"><td>{r["plant_id"]}</td>'
        f"<td>{r['absent_observations']}</td><td>{r['confidence']:.2f}</td>"
        f"<td>{escape(r['reason'])}</td><td>{_artifact_button(r)}</td><td>"
        f'<form method=post action="removals/{r["measurement_id"]}/approve"><button class=review-action '
        f'data-url="removals/{r["measurement_id"]}/approve">Approve removal</button></form>'
        f'<form method=post action="removals/{r["measurement_id"]}/keep"><button class=review-action '
        f'data-url="removals/{r["measurement_id"]}/keep">Keep plant</button></form>'
        '<small class=action-message></small></td></tr>'
        for r in rows
        if r["decision"] == "removal_recommended"
    )
    proposal_rows = []
    for proposal in database.curve_proposals():
        previous = json.loads(proposal["previous_data_json"] or "{}")
        proposed = json.loads(proposal["data_json"] or "{}")
        day = int(proposal["plant_age_days"])
        value = float(proposed.get(str(day), 0))
        diagnostic = _artifact_button(
            {
                "artifact_paths": [proposal["overlay_path"]] if proposal["overlay_path"] else [],
                "current_radius_mm": value / 2,
                "typical_canopy_radius_mm": value / 2,
                "maximum_accepted_canopy_radius_mm": value / 2,
                "recommended_protection_radius_mm": value / 2,
                "safety_margin_mm": 0,
                "calibration_uncertainty_mm": 0,
            }
        )
        proposal_rows.append(
            f'<tr class=review-item id="curve-proposal-{proposal["id"]}"><td>{proposal["plant_id"]}</td>'
            f"<td>{escape(str(previous))}</td><td>day {day}: "
            f'<input class=curve-value form="curve-apply-{proposal["id"]}" name=value '
            f'type=number min=0 step=any value="{value:g}"> mm diameter</td>'
            f"<td>{escape(proposal['reason'] or '')}; old conflict "
            f"day {escape(str(proposal['conflict_day']))} = {escape(str(proposal['conflict_old_diameter']))}</td>"
            f"<td>{diagnostic}</td><td>"
            f'<form id="curve-apply-{proposal["id"]}" method=post action="curve-proposals/{proposal["id"]}/apply">'
            '<input type=hidden name=confirm_override value=true>'
            f'<button class=curve-action data-action=apply data-url="curve-proposals/{proposal["id"]}/apply">Use value</button></form>'
            f'<form method=post action="curve-proposals/{proposal["id"]}/discard-new"><button class=curve-action '
            f'data-action=discard-new data-url="curve-proposals/{proposal["id"]}/discard-new">Discard new</button></form>'
            f'<form method=post action="curve-proposals/{proposal["id"]}/discard-old"><button class=curve-action '
            f'data-action=discard-old data-url="curve-proposals/{proposal["id"]}/discard-old">Discard old</button></form>'
            '<small class=action-message></small></td></tr>'
        )
    flagged_curve_rows = "".join(proposal_rows)
    decision_rows = "".join(
        f"<tr><td>{escape(row['created_at'])}</td><td>{escape(row['measurement_id'])}</td>"
        f"<td>{escape(row['action'])}</td></tr>"
        for row in database.recent_decisions()
    )
    resolution = settings.resolution

    def _dims(value: object) -> str:
        if isinstance(value, list) and len(value) == 2 and value[0] is not None:
            return f"{value[0]}x{value[1]}"
        return "—"

    warnings = last.get("calibration_warnings") or []
    warning_html = (
        "".join(f"<li class=warn>{escape(str(w))}</li>" for w in warnings)
        if warnings
        else "<li class=muted>None</li>"
    )
    skip_reasons = last.get("skip_reasons") or {}
    skip_html = (
        "".join(
            f"<li>Plant {escape(str(pid))}: {escape(str(reason))}</li>"
            for pid, reason in skip_reasons.items()
        )
        if skip_reasons
        else "<li class=muted>None</li>"
    )
    body = f"""<div class=grid><section class=card><h2>Health</h2><b>{escape(jobs.current["status"])}</b>
<p>{escape(jobs.current.get("progress", ""))}</p><p class=muted>App {__version__} · {ALGORITHM_VERSION} · {CONTRACT_VERSION}</p></section>
<section class=card><h2>FarmBot</h2><p>{escape(settings.selected_config_entry_id or "Not selected")}</p>
<p>Mode: {settings.mode.value}</p></section>
<section class=card><h2>Analysis resolution</h2><p><b>{escape(resolution.label)}</b></p>
<p class=muted>{resolution.pixel_count:,} px · restart to change</p></section>
<section class=card><h2>Queue</h2><p>{jobs.current.get("queue_length", 0)} waiting</p>
<form method=post action="analyse"><button>Analyse now</button></form></section></div>
<section class=card><h2>Last job</h2>
<p>{escape(last.get("message", "Never run"))}</p>
<div class=grid>
<div><b>Timing</b><p class=muted>Duration {last.get("duration_seconds", "—")} s · CPU {last.get("cpu_seconds", "—")} s · peak {last.get("peak_memory_mb", "—")} MB</p></div>
<div><b>Images</b><p class=muted>{last.get("images_processed", "—")} processed · {last.get("uncalibrated_images", 0)} uncalibrated</p></div>
<div><b>Plants</b><p class=muted>{last.get("plants_measured", "—")} measured · {last.get("uncertain", "—")} uncertain · {last.get("skipped", "—")} skipped</p></div>
<div><b>Dimensions</b><p class=muted>source {escape(_dims(last.get("source_dimensions")))} · oriented {escape(_dims(last.get("oriented_dimensions")))} · processed {escape(_dims(last.get("processed_dimensions")))}</p></div>
<div><b>Calibration</b><p class=muted>source {escape(str(last.get("calibration_source") or "—"))}</p></div>
<div><b>Contract</b><p class=muted>{escape(str(last.get("contract_version") or CONTRACT_VERSION))} · min integration {MINIMUM_INTEGRATION_VERSION}</p></div>
</div>
<p><b>Calibration warnings</b></p><ul>{warning_html}</ul>
<p><b>Skip reasons</b></p><ul>{skip_html}</ul></section>
<section class=card><h2>Measurements</h2><table><thead><tr><th>Plant</th><th>Crop</th><th>Resolution</th><th>Current</th><th>Max leaf</th><th>Recommended</th><th>Confidence</th><th>Calibration</th><th>Decision</th><th>Reason</th><th>Diagnostic</th><th>Review</th></tr></thead><tbody>{measurement_rows or "<tr><td colspan=12>No measurements yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Removed / missing plants</h2><table><thead><tr><th>Plant</th><th>Absent looks</th><th>Confidence</th><th>Reason</th><th>Diagnostic</th><th>Review</th></tr></thead><tbody>{removal_rows or "<tr><td colspan=6>No confirmed missing plants</td></tr>"}</tbody></table></section>
<section class=card><h2>Growth-curve updates</h2><p class=muted>Flagged per-plant diameter points require review.</p><table><tbody>{flagged_curve_rows or "<tr><td>No flagged curve updates</td></tr>"}</tbody></table></section>
<section class=card><h2>Crop protection spread proposals</h2><p class=muted>Monotonic and limited to 10 points. FarmBot values are diameters; assignment requires approval.</p><table><tbody>{curve_rows or "<tr><td>No curve is ready</td></tr>"}</tbody></table></section>
<section class=card><h2>Approval and rollback history</h2><table><tbody>{decision_rows or "<tr><td>No decisions yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Safety warning</h2><p class=warn>Early experimental vision results must not be the sole basis for destructive automatic weeding.</p></section>
<div id=overlay-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Analysis diagnostic"><figure>
<button id=overlay-modal-close class=modal-close type=button aria-label=Close>&times;</button>
<img id=overlay-modal-img alt="Plant analysis diagnostic"><figcaption id=overlay-modal-details></figcaption>
<p class=legend>Vegetation mask and per-plant ownership are shown as separate gallery images.</p>
<div class=modal-controls><button id=overlay-modal-prev type=button>Previous</button><span id=overlay-modal-counter></span><button id=overlay-modal-next type=button>Next</button></div>
</figure></div><script>{_DASHBOARD_JS}</script>"""
    return layout(request, body)


@app.post("/analyse")
async def analyse(background: BackgroundTasks) -> RedirectResponse:
    background.add_task(jobs.run, trigger="manual")
    return RedirectResponse("./", status_code=303)


def _calibration_warnings(calibration: Calibration | None) -> list[str]:
    """Warnings when an existing calibration may not fit the current setup."""
    warnings: list[str] = []
    if calibration is None:
        return warnings
    resolution = settings.resolution
    if calibration.processed_width and (
        calibration.processed_width != resolution.width
        or calibration.processed_height != resolution.height
    ):
        warnings.append(
            f"Calibration belongs to {calibration.processed_width}x{calibration.processed_height}; "
            f"the app is configured for {resolution.width}x{resolution.height}."
        )
    elif not calibration.processed_width:
        warnings.append(
            "Calibration has no recorded resolution and cannot be verified against the "
            "current preset; recalibration is recommended."
        )
    if calibration.source == "manual_transformed":
        warnings.append(
            "This calibration was mathematically transformed from another resolution; "
            "verify plant-centre alignment before trusting it."
        )
    return warnings


def _origin_options(selected: str) -> str:
    labels = {
        "top_left": "Top left",
        "top_right": "Top right",
        "bottom_left": "Bottom left",
        "bottom_right": "Bottom right",
    }
    return "".join(
        f"<option value={value}{' selected' if value == selected else ''}>{escape(label)}</option>"
        for value, label in labels.items()
    )


@app.get("/settings", response_class=HTMLResponse)
async def calibration_page(request: Request) -> HTMLResponse:
    entry_id = settings.selected_config_entry_id
    calibration = database.active_calibration(entry_id)
    resolution = settings.resolution
    warnings = _calibration_warnings(calibration)
    warning_html = "".join(f"<p class=warn>⚠ {escape(w)}</p>" for w in warnings)
    current = "none"
    if calibration is not None:
        current = (
            f"source={calibration.source}, "
            f"{calibration.pixels_per_mm_x:.4f}×{calibration.pixels_per_mm_y:.4f} px/mm, "
            f"resolution={calibration.processed_width}x{calibration.processed_height}, "
            f"rotation={calibration.rotation_degrees}°, "
            f"origin={calibration.origin_location}, "
            f"offsets=({calibration.offset_x_mm},{calibration.offset_y_mm}) mm"
        )
    # Prefill the form with the durable stored inputs so a restart shows the last
    # saved calibration ready to edit (persistence is /data-backed, not options).
    stored = calibration_store.get(entry_id) if entry_id else None
    v_scale = "" if stored is None else stored.coordinate_scale
    v_refw = 2592 if stored is None else stored.reference_width
    v_refh = 1944 if stored is None else stored.reference_height
    v_rot = 0 if stored is None else stored.rotation_degrees
    v_ox = 0 if stored is None else stored.offset_x_mm
    v_oy = 0 if stored is None else stored.offset_y_mm
    v_origin = "top_left" if stored is None else str(stored.origin_location)
    body = f"""<section class=card><h2>FarmBot calibration</h2>
<p>Copy the values from FarmBot's own camera calibration (Photos → Camera
calibration), then verify alignment against a whole photo row. The app rescales
FarmBot's mm/pixel scale (measured at its native frame) to the configured
analysis resolution ({escape(resolution.label)}). Values are saved to the app's
persistent storage and restored automatically after a restart — no external
tools needed.</p>
{warning_html}
<p class=muted>Current active calibration: {escape(current)}</p>
<div class=grid>
<div>
<label>FarmBot config entry ID<br><input id=entry_id value="{escape(entry_id)}"></label>
<p><button type=button id=load>Load bot inventory</button></p>
<label>Photo row (same X)<br><select id=row></select></label>
<label>Row X tolerance (mm)<br><input id=rowtol type=number min=1 step=any value=50></label>
<hr>
<p class=muted>In FarmBot open Photos → Camera calibration and copy each value below.</p>
<label>Pixel coordinate scale (mm/pixel)<br><input id=fb_scale type=number min=0 step=any value="{v_scale}"></label>
<label>Measured at width (px)<br><input id=fb_refw type=number min=1 step=1 value="{v_refw}"></label>
<label>Measured at height (px)<br><input id=fb_refh type=number min=1 step=1 value="{v_refh}"></label>
<p id=ppm class=muted>Enter the FarmBot pixel coordinate scale, and measured-at width/height</p>
<label>Camera rotation (degrees)<br><input id=rotation type=number step=any value="{v_rot}"></label>
<label>Origin location in image<br><select id=origin>{_origin_options(v_origin)}</select></label>
<label>Offset X (mm)<br><input id=offx type=number step=any value="{v_ox}"></label>
<label>Offset Y (mm)<br><input id=offy type=number step=any value="{v_oy}"></label>
<p class=muted>Leave offsets at 0 unless the overlay is shifted. FarmBot's camera offset is
already folded into the image-centre coordinate, so entering it again would double-count.</p>
<label><input type=checkbox id=showoverlay checked> Overlay plant &amp; weed centres</label><br>
<label><input type=checkbox id=showlabels checked> Show labels (name / weed)</label>
<p><label><input type=checkbox id=confirm> Centres align across the row</label></p>
<p><button type=button id=save disabled>Save calibration</button></p>
<p id=status class=muted></p>
</div>
<div>
<canvas id=canvas width=640 height=200
 style="width:100%;border:1px solid #ccc;background:#111"></canvas>
<p class=muted>Green = known plants (name · crop). Red = FarmBot weeds. Adjust the
values above and the composite updates live.</p>
</div>
</div>
</section>
<script>{_CALIBRATION_JS}</script>"""
    return layout(request, body, "Calibration · FarmBot Vision")


@app.get("/api/vision/images")
async def vision_images(entry_id: str) -> JSONResponse:
    try:
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=entry_id, image_lookback_hours=settings.image_lookback_hours
            )
        )
    except HomeAssistantError as exc:
        LOGGER.warning(
            "GET /api/vision/images failed: entry_id=%s (%s): %s",
            entry_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(502, "could not load images") from exc
    images = [
        {"id": i.id, "created_at": i.created_at.isoformat(), "x": i.meta.x, "y": i.meta.y}
        for i in sorted(inventory.images, key=lambda item: item.created_at, reverse=True)
        if i.processed
    ]
    plants = [
        {
            "id": p.id,
            "name": p.name,
            "slug": p.openfarm_slug,
            "x": p.x,
            "y": p.y,
            "radius": p.radius,
        }
        for p in inventory.plants
    ]
    weeds = [
        {"id": w.id, "name": w.name, "x": w.x, "y": w.y, "radius": w.radius}
        for w in inventory.weeds
    ]
    return JSONResponse({"images": images, "plants": plants, "weeds": weeds})


@app.get("/api/vision/image/{image_id}.jpg")
async def vision_image(entry_id: str, image_id: int) -> Response:
    try:
        response = await client.image(
            VisionImageRequest(
                config_entry_id=entry_id,
                image_id=image_id,
                max_width=settings.analysis_width,
                max_height=settings.analysis_height,
            ),
            settings.max_image_payload_bytes,
        )
    except HomeAssistantError as exc:
        LOGGER.warning(
            "GET /api/vision/image/%s.jpg failed: entry_id=%s (%s): %s",
            image_id,
            entry_id,
            type(exc).__name__,
            exc,
        )
        raise HTTPException(502, "could not load image") from exc
    return Response(base64.b64decode(response.image_base64), media_type="image/jpeg")


@app.post("/calibration")
async def save_calibration(
    entry_id: str = Form(...),
    coordinate_scale: float = Form(...),
    reference_width: int = Form(...),
    reference_height: int = Form(...),
    rotation: float = Form(0),
    offset_x: float = Form(0),
    offset_y: float = Form(0),
    origin_location: str = Form("top_left"),
) -> RedirectResponse:
    """Persist the FarmBot camera calibration for a bot.

    The entered values are written to the durable /data store (the master record
    that survives restarts) and the derived processed-resolution calibration is
    made the active one in the database (the runtime source the analysis
    pipeline reads).
    """
    try:
        origin = OriginLocation(origin_location)
    except ValueError as exc:
        raise HTTPException(400, "invalid origin location") from exc
    try:
        values = FarmbotCalibrationInput(
            coordinate_scale=coordinate_scale,
            reference_width=reference_width,
            reference_height=reference_height,
            rotation_degrees=rotation,
            origin_location=origin,
            offset_x_mm=offset_x,
            offset_y_mm=offset_y,
        )
        calibration = _calibration_from_input(entry_id, values)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    calibration_store.save(entry_id, values)
    database.save_calibration(entry_id, calibration)
    return RedirectResponse("settings", status_code=303)


@app.get("/artifact/{name}")
async def artifact(name: str) -> FileResponse:
    safe_name = Path(name).name
    path = settings.data_dir / "artifacts" / safe_name
    if safe_name != name or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


def _measurement_from_row(row: dict) -> Measurement:
    payload = {name: row[name] for name in Measurement.model_fields if name in row}
    try:
        payload["artifact_paths"] = json.loads(row.get("artifact_paths_json") or "[]")
    except (TypeError, json.JSONDecodeError):
        payload["artifact_paths"] = []
    return Measurement.model_validate(payload)


def _action_response(
    request: Request, status: str, message: str, *, error_status: int | None = None
) -> Response:
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(
            {"status": status, "message": message},
            status_code=error_status or 200,
        )
    if error_status is not None:
        raise HTTPException(error_status, message)
    destination = ingress_base(request)
    if destination == "./":
        destination = "../../../"
    return RedirectResponse(destination, status_code=303)


@app.post("/recommendations/{measurement_id}/{action}")
async def recommendation(request: Request, measurement_id: str, action: str) -> Response:
    if action not in {"approve", "reject"}:
        raise HTTPException(400)
    row = database.measurement(measurement_id)
    if row is None:
        raise HTTPException(404)
    if database.has_terminal_decision(measurement_id):
        return _action_response(request, "conflict", "This recommendation was already reviewed", error_status=409)
    if row["decision"] != "recommended":
        return _action_response(request, "conflict", "Only recommended radius changes can be reviewed", error_status=409)
    if action == "approve":
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if not row.get("calibrated", 1):
            return _action_response(request, "conflict", "Calibration is required", error_status=409)
        if row["recommended_protection_radius_mm"] <= row["current_radius_mm"]:
            return _action_response(request, "conflict", "Shrinking is disabled", error_status=409)
        entry_id = row.get("config_entry_id") or settings.selected_config_entry_id
        try:
            result = await client.apply_radius(
                ApplyRadiusRequest(
                    config_entry_id=entry_id,
                    plant_id=row["plant_id"],
                    measurement_id=measurement_id,
                    expected_current_radius_mm=row["current_radius_mm"],
                    recommended_radius_mm=row["recommended_protection_radius_mm"],
                    confidence=row["confidence"],
                    apply=True,
                    human_approved=True,
                )
            )
        except StaleRadiusError:
            await client.inventory(
                InventoryRequest(
                    config_entry_id=entry_id,
                    image_lookback_hours=settings.image_lookback_hours,
                )
            )
            return _action_response(request, "conflict", "The plant radius changed; inventory refreshed", error_status=409)
        status = str(result.get("status", "error"))
        message = str(result.get("message") or status)
        if status != "applied":
            if status == "conflict":
                await client.inventory(
                    InventoryRequest(
                        config_entry_id=entry_id,
                        image_lookback_hours=settings.image_lookback_hours,
                    )
                )
            return _action_response(
                request,
                status,
                message,
                error_status=409 if status == "conflict" else None,
            )
        database.update_measurement_outcome(measurement_id, decision="applied", applied=True)
        database.record_decision(measurement_id, "applied", result)
        approved_measurement = _measurement_from_row(row)
        if approved_measurement.plant_age_days is None:
            curve_message = "skipped because plant age is unavailable"
        else:
            try:
                inventory = await client.inventory(
                    InventoryRequest(
                        config_entry_id=entry_id,
                        image_lookback_hours=settings.image_lookback_hours,
                    )
                )
                curve_result = await jobs._update_curve_after_radius(
                    entry_id, inventory, approved_measurement, human_approved=True
                )
                curve_message = str(
                    curve_result.get("message") or curve_result.get("status", "")
                )
            except HomeAssistantError as exc:
                LOGGER.warning("Radius applied but curve inventory/update failed: %s", exc)
                curve_message = "deferred because inventory was unavailable"
        return _action_response(
            request,
            "applied",
            f"Radius applied. Curve update: {curve_message}",
        )
    database.record_decision(measurement_id, "reject", {})
    return _action_response(request, "rejected", "Recommendation rejected")


@app.post("/removals/{measurement_id}/{action}")
async def removal(request: Request, measurement_id: str, action: str) -> Response:
    if action not in {"approve", "keep"}:
        raise HTTPException(400)
    row = database.measurement(measurement_id)
    if row is None:
        raise HTTPException(404)
    if database.has_terminal_decision(measurement_id):
        return _action_response(request, "conflict", "This removal was already reviewed", error_status=409)
    if row["decision"] != "removal_recommended":
        return _action_response(request, "conflict", "Removal is not currently recommended", error_status=409)
    entry_id = row.get("config_entry_id") or settings.selected_config_entry_id
    if not database.is_latest_plant_measurement(entry_id, row["plant_id"], measurement_id):
        return _action_response(
            request,
            "conflict",
            "A newer canopy observation exists; removal was not applied",
            error_status=409,
        )
    if action == "keep":
        database.record_decision(measurement_id, "keep", {})
        return _action_response(request, "rejected", "Plant kept")
    try:
        result = await client.apply_removal(
            ApplyRemovalRequest(
                config_entry_id=entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_current_radius_mm=row["current_radius_mm"],
                confidence=row["confidence"],
                apply=True,
                human_approved=True,
            )
        )
    except StaleRadiusError:
        return _action_response(request, "conflict", "The plant changed; removal was not applied", error_status=409)
    status = str(result.get("status", "error"))
    message = str(result.get("message") or status)
    if status != "applied":
        return _action_response(request, status, message, error_status=409)
    database.update_measurement_outcome(measurement_id, decision="removed", applied=True)
    database.record_decision(measurement_id, "removed", result)
    return _action_response(request, "applied", message)


@app.post("/curve-proposals/{proposal_id}/{action}")
async def curve_proposal_action(
    request: Request,
    proposal_id: int,
    action: str,
    value: float | None = Form(None),
    confirm_override: bool = Form(False),
) -> Response:
    if action not in {"apply", "discard-new", "discard-old"}:
        raise HTTPException(400)
    proposal = database.curve_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(404)
    if proposal["status"] != "flagged":
        return _action_response(request, "conflict", "This proposal was already reviewed", error_status=409)
    if action == "discard-new":
        database.update_curve_proposal(proposal_id, "rejected")
        return _action_response(request, "rejected", "New curve value discarded")
    previous = json.loads(proposal["previous_data_json"] or "{}")
    proposed = json.loads(proposal["data_json"] or "{}")
    day = int(proposal["plant_age_days"])
    new_value = float(value if value is not None else proposed[str(day)])
    base = dict(previous)
    if action == "discard-old" and proposal["conflict_day"] is not None:
        base.pop(str(proposal["conflict_day"]), None)
    edit = propose_curve_point(
        base,
        day,
        new_value,
        max_daily_growth_mm=settings.maximum_daily_radius_growth_mm,
        maximum_plant_radius_mm=settings.maximum_plant_radius_mm,
    )
    if edit.verdict == "flagged" and not confirm_override:
        return _action_response(
            request,
            "conflict",
            f"Edited value is still flagged: {edit.reason}; confirm the override to apply",
            error_status=409,
        )
    entry_id = proposal["config_entry_id"] or settings.selected_config_entry_id
    inventory = await client.inventory(
        InventoryRequest(
            config_entry_id=entry_id,
            image_lookback_hours=settings.image_lookback_hours,
        )
    )
    plant = next((item for item in inventory.plants if item.id == proposal["plant_id"]), None)
    assigned = None if plant is None else next(
        (item for item in inventory.curves if item.id == plant.spread_curve_id), None
    )
    if plant is None or assigned is None or assigned.data != previous:
        return _action_response(
            request,
            "conflict",
            "The plant's assigned curve changed after this proposal was created",
            error_status=409,
        )
    curve_data = {control_day: float(round(diameter)) for control_day, diameter in edit.data.items()}
    result = await client.upsert_curve(
        UpsertCurveRequest(
            config_entry_id=entry_id,
            crop_slug=proposal["crop_slug"],
            curve_id=proposal["farmbot_curve_id"],
            name=proposal["curve_name"],
            data=curve_data,
            assign_to_plant_ids=[proposal["plant_id"]],
            apply=True,
            human_approved=True,
        )
    )
    status = str(result.get("status", "error"))
    message = str(result.get("message") or status)
    if status != "applied":
        return _action_response(request, status, message, error_status=409)
    database.update_curve_proposal(proposal_id, "applied", curve_data)
    database.record_decision(proposal["measurement_id"], "curve_applied", result)
    return _action_response(request, "applied", message)
