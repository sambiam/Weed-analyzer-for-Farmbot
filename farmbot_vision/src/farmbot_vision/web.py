from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Annotated
from urllib.parse import quote
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
from .canopy_settings import CanopyFusionSettings, CanopyFusionSettingsStore
from .curve_edit import propose_curve_point
from .curves import fit_monotonic_curve
from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError, StaleRadiusError
from .jobs import JobManager
from .models import (
    ApplyPlantCenterRequest,
    ApplyRadiusRequest,
    ApplyRemovalRequest,
    ApplySoilHeightRequest,
    Calibration,
    CreateWeedRequest,
    InventoryRequest,
    Measurement,
    OperatingMode,
    OriginLocation,
    QueueImagesRequest,
    UpsertCurveRequest,
    VisionImageRequest,
)
from .settings import Settings
from .soil_jobs import SoilJobManager
from .vision import garden_to_pixel
from .weed_settings import WeedSettings, WeedSettingsStore
from .weed_verifier import ALL_LABELS, WeedVisualVerifier
from .zones import (
    Zone,
    ZoneAspect,
    ZoneKind,
    ZoneShape,
    ZoneStore,
    ZoneVerdict,
    evaluate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
settings = Settings.load()
database = Database(settings.data_dir / "farmbot_vision.db")
calibration_store = CalibrationStore(settings.data_dir / "farmbot_calibration.json")
client = HomeAssistantClient()
weed_settings_store = WeedSettingsStore(settings.data_dir / "weed_settings.json")
canopy_fusion_settings_store = CanopyFusionSettingsStore(
    settings.data_dir / "canopy_fusion_settings.json"
)
weed_verifier = WeedVisualVerifier(settings.data_dir / "weed_visual_model.json")
zone_store = ZoneStore(settings.data_dir / "zones.json")
jobs = JobManager(settings, database, client, weed_settings_store, zone_store)
soil_jobs = SoilJobManager(database, client, settings.data_dir, jobs.lock, zone_store)


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
  const plantToggle=document.getElementById('plant-view-toggle');
  const plantWithoutOverlay=document.getElementById('plant-modal-without-overlay');
  const plantWithOverlay=document.getElementById('plant-modal-with-overlay');
  const artifactControls=document.getElementById('artifact-controls');
  const overlayLegend=document.getElementById('overlay-modal-legend');
  let artifacts=[], index=0, returnFocus=null;
  let plantComposite=null;
  const queueModal=document.getElementById('queue-modal');
  const queueRows=document.getElementById('queue-image-rows');
  const queueMessage=document.getElementById('queue-message');
  async function loadQueueImages(){
    const dateFrom=document.getElementById('queue-from').value;
    const dateTo=document.getElementById('queue-to').value;
    if(!dateFrom||!dateTo){queueMessage.textContent='Choose both a from and to date';return;}
    if(new Date(dateFrom)>new Date(dateTo)){queueMessage.textContent='From must be before to';return;}
    queueMessage.textContent='Loading images…';
    try{
      const query=new URLSearchParams({date_from:new Date(dateFrom).toISOString(),
        date_to:new Date(dateTo).toISOString()});
      const response=await fetch('api/analysis/images?'+query.toString());
      const data=await response.json();
      if(!response.ok) throw new Error(data.detail||('HTTP '+response.status));
      queueRows.innerHTML=(data.images||[]).map(function(image){
        const plants=(image.plants||[]).map(p=>p.name+' (#'+p.id+')').join(', ')||'None';
        return '<tr><td><input class=queue-checkbox type=checkbox value="'+image.id+'"></td>'
          +'<td>'+image.x.toFixed(1)+', '+image.y.toFixed(1)+', '+image.z.toFixed(1)+'</td>'
          +'<td>'+plants+'</td><td>'+new Date(image.created_at).toLocaleString()+'</td></tr>';
      }).join('')||'<tr><td colspan=4>No images in this timeframe</td></tr>';
      queueMessage.textContent=data.images.length+' images found';
    }catch(error){queueMessage.textContent='Could not load images: '+error.message;}
  }
  function showArtifact(){
    if(!artifacts.length) return;
    modalImg.src=artifacts[index];
    counter.textContent=(index+1)+' / '+artifacts.length;
  }
  function closeModal(){
    modal.hidden=true; modalImg.removeAttribute('src');
    plantComposite=null;
    if(returnFocus) returnFocus.focus();
  }
  function showPlantComposite(withOverlay){
    if(!plantComposite) return;
    const useOverlay=withOverlay&&plantComposite.overlay;
    modalImg.src=useOverlay?plantComposite.overlay:plantComposite.clean;
    plantWithoutOverlay.setAttribute('aria-pressed',String(!useOverlay));
    plantWithOverlay.setAttribute('aria-pressed',String(Boolean(useOverlay)));
  }
  const weedModal=document.getElementById('weed-modal');
  const weedImg=document.getElementById('weed-modal-img');
  const weedMarker=document.getElementById('weed-modal-marker');
  const weedDetails=document.getElementById('weed-modal-details');
  const weedMessage=document.getElementById('weed-modal-message');
  const weedAccept=document.getElementById('weed-modal-accept');
  const weedReject=document.getElementById('weed-modal-reject');
  const weedAcceptAll=document.getElementById('weed-modal-accept-all');
  const weedWithoutOverlay=document.getElementById('weed-modal-without-overlay');
  const weedWithOverlay=document.getElementById('weed-modal-with-overlay');
  let weedData=null, weedReturnFocus=null;
  function showWeedView(withOverlay){
    if(!weedData) return;
    const noCleanImage=!weedData.reviewArtifact;
    const useOverlay=withOverlay||noCleanImage;
    weedImg.src=useOverlay?weedData.overlayArtifact:weedData.reviewArtifact;
    weedWithoutOverlay.setAttribute('aria-pressed',String(!useOverlay));
    weedWithOverlay.setAttribute('aria-pressed',String(useOverlay));
    weedMessage.textContent=(!withOverlay&&noCleanImage)
      ?'No image without the overlay was saved for this older detection; showing the analysis overlay instead.'
      :'';
  }
  function closeWeedModal(){
    weedModal.hidden=true; weedImg.removeAttribute('src'); weedData=null;
    if(weedReturnFocus) weedReturnFocus.focus();
  }
  function openWeedModal(data,trigger){
    weedData=data; weedReturnFocus=trigger; weedMessage.textContent='';
    showWeedView(false);
    if(data.x!=null&&data.y!=null&&data.width&&data.height){
      weedMarker.style.left=(data.x/data.width*100)+'%';
      weedMarker.style.top=(data.y/data.height*100)+'%';
      weedMarker.hidden=false;
    } else weedMarker.hidden=true;
    const others=Math.max(0,(data.siblings||[]).length-1);
    weedDetails.textContent='Area '+data.areaMm2.toFixed(1)+' mm² · confidence '+data.confidence.toFixed(2)
      +' · '+(data.observations||1)+' independent look(s)'
      +(data.verifierConfidence!=null?(' · verifier '+data.verifierConfidence.toFixed(2)):'')
      +(others?(' · '+others+' other weed(s) in this image'):'');
    weedModal.hidden=false; weedModal.querySelector('.modal-close').focus();
  }
  async function postWeedAction(id,action){
    try{
      const response=await fetch('weeds/'+id+'/'+action,{method:'POST',headers:{Accept:'application/json'}});
      const result=await response.json().catch(function(){return {};});
      const ok=response.ok&&(result.status==='applied'||result.status==='rejected');
      if(ok){const row=document.getElementById('weed-'+id); if(row) row.remove();}
      return ok;
    }catch(_){return false;}
  }
  weedAccept.addEventListener('click',async function(){
    if(!weedData) return;
    weedAccept.disabled=true;
    try{
      const ok=await postWeedAction(weedData.detectionId,'approve');
      if(ok) closeWeedModal(); else weedMessage.textContent='Could not accept weed';
    }finally{weedAccept.disabled=false;}
  });
  weedReject.addEventListener('click',async function(){
    if(!weedData) return;
    weedReject.disabled=true;
    try{
      const ok=await postWeedAction(weedData.detectionId,'reject');
      if(ok) closeWeedModal(); else weedMessage.textContent='Could not reject weed';
    }finally{weedReject.disabled=false;}
  });
  weedAcceptAll.addEventListener('click',async function(){
    if(!weedData) return;
    weedAcceptAll.disabled=true;
    try{
      const ids=(weedData.siblings&&weedData.siblings.length)?weedData.siblings:[weedData.detectionId];
      let failures=0;
      for(const id of ids){ if(!await postWeedAction(id,'approve')) failures++; }
      if(!failures) closeWeedModal();
      else weedMessage.textContent=failures+' weed(s) could not be accepted';
    }finally{weedAcceptAll.disabled=false;}
  });
  weedWithoutOverlay.addEventListener('click',function(){showWeedView(false);});
  weedWithOverlay.addEventListener('click',function(){showWeedView(true);});
  document.getElementById('weed-modal-close').addEventListener('click',closeWeedModal);
  weedModal.addEventListener('click',function(event){if(event.target===weedModal) closeWeedModal();});
  plantWithoutOverlay.addEventListener('click',function(){showPlantComposite(false);});
  plantWithOverlay.addEventListener('click',function(){showPlantComposite(true);});
  document.addEventListener('click',async function(event){
    const weedViewer=event.target.closest('.weed-view');
    if(weedViewer){
      let data=null; try{data=JSON.parse(weedViewer.dataset.weed||'null');}catch(_){data=null;}
      if(data) openWeedModal(data,weedViewer);
      return;
    }
    const plantViewer=event.target.closest('[data-composite-clean]');
    if(plantViewer){
      plantComposite={
        clean:plantViewer.dataset.compositeClean,
        overlay:plantViewer.dataset.compositeOverlay||null
      };
      if(!plantComposite.clean) return;
      returnFocus=plantViewer;
      let details={}; try{details=JSON.parse(plantViewer.dataset.details||'{}');}catch(_){}
      modalDetails.textContent=details.formula||'';
      plantToggle.hidden=false;
      artifactControls.hidden=true;
      plantWithOverlay.disabled=!plantComposite.overlay;
      overlayLegend.textContent='Cyan circle = original radius; red circle = new radius; white dot = plant center.';
      modal.hidden=false; showPlantComposite(false); closeButton.focus(); return;
    }
    const viewer=event.target.closest('[data-artifacts]');
    if(viewer){
      try{artifacts=JSON.parse(viewer.dataset.artifacts||'[]');}catch(_){artifacts=[];}
      if(!artifacts.length) return;
      index=0; returnFocus=viewer;
      let details={}; try{details=JSON.parse(viewer.dataset.details||'{}');}catch(_){}
      modalDetails.textContent=details.formula||'';
      plantToggle.hidden=true;
      artifactControls.hidden=false;
      overlayLegend.textContent='Cyan circle = original radius; red circle = planned radius.';
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
  document.addEventListener('keydown',function(event){
    if(event.key!=='Escape') return;
    if(!modal.hidden) closeModal();
    if(!weedModal.hidden) closeWeedModal();
  });
  document.getElementById('queue-open').addEventListener('click',function(){
    const to=new Date(), from=new Date(to.getTime()-72*60*60*1000);
    function localValue(value){
      const shifted=new Date(value.getTime()-value.getTimezoneOffset()*60000);
      return shifted.toISOString().slice(0,16);
    }
    if(!document.getElementById('queue-to').value){
      document.getElementById('queue-from').value=localValue(from);
      document.getElementById('queue-to').value=localValue(to);
    }
    queueModal.hidden=false;loadQueueImages();
  });
  document.getElementById('queue-close').addEventListener('click',function(){queueModal.hidden=true;});
  document.getElementById('queue-refresh').addEventListener('click',loadQueueImages);
  document.getElementById('queue-select-all').addEventListener('change',function(){
    document.querySelectorAll('.queue-checkbox').forEach(box=>box.checked=this.checked);
  });
  document.getElementById('queue-add').addEventListener('click',async function(){
    const ids=[...document.querySelectorAll('.queue-checkbox:checked')].map(box=>+box.value);
    if(!ids.length){queueMessage.textContent='Select at least one image';return;}
    const response=await fetch('analysis/queue',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image_ids:ids})});
    const data=await response.json();
    if(response.ok){document.getElementById('queue-count').textContent=data.queue_length;
      queueMessage.textContent=ids.length+' images added';}
    else queueMessage.textContent=data.detail||'Could not add images';
  });
})();
"""

# Boundaries and exclusion zones. The add form only shows the geometry fields of
# the selected shape, permissions start at the sensible polarity for the chosen
# kind, and a top-down garden map draws every zone (optionally with the bot's
# plants and weeds) so a zone can be checked before it starts gating writes.
_ZONES_JS = r"""
(function(){
  const shape=document.getElementById('shape');
  const kind=document.getElementById('kind');
  const canvas=document.getElementById('zone-map');
  const ctx=canvas.getContext('2d');
  const status=document.getElementById('zone-map-status');
  let zones=[]; try{zones=JSON.parse(canvas.dataset.zones||'[]');}catch(_){zones=[];}
  let items={plants:[],weeds:[]};

  function showShapeFields(){
    ['rectangle','circle','polygon'].forEach(function(name){
      const box=document.getElementById('fields-'+name);
      const active=(shape.value===name);
      box.hidden=!active;
      box.querySelectorAll('input,textarea').forEach(function(field){field.disabled=!active;});
    });
  }
  // Boundaries usually permit everything inside; exclusion zones usually
  // forbid everything. Both stay editable afterwards.
  function applyKindDefaults(){
    const allow=(kind.value==='boundary');
    ['allow_weeds','allow_plant_centers','allow_plant_radius'].forEach(function(name){
      document.getElementById('new_'+name).checked=allow;
    });
  }
  function zonePoints(zone){
    if(zone.shape==='rectangle')
      return [[zone.min_x,zone.min_y],[zone.max_x,zone.min_y],
              [zone.max_x,zone.max_y],[zone.min_x,zone.max_y]];
    if(zone.shape==='circle')
      return [[zone.center_x-zone.radius_mm,zone.center_y-zone.radius_mm],
              [zone.center_x+zone.radius_mm,zone.center_y+zone.radius_mm]];
    return zone.points||[];
  }
  function bounds(){
    let xs=[],ys=[];
    zones.forEach(function(zone){zonePoints(zone).forEach(function(p){xs.push(p[0]);ys.push(p[1]);});});
    items.plants.concat(items.weeds).forEach(function(p){xs.push(p.x);ys.push(p.y);});
    if(!xs.length) return null;
    let minX=Math.min.apply(null,xs), maxX=Math.max.apply(null,xs);
    let minY=Math.min.apply(null,ys), maxY=Math.max.apply(null,ys);
    const padX=Math.max(50,(maxX-minX)*0.08), padY=Math.max(50,(maxY-minY)*0.08);
    return {minX:minX-padX,maxX:maxX+padX,minY:minY-padY,maxY:maxY+padY};
  }
  function drawZone(zone,project,scale){
    const boundary=(zone.kind==='boundary');
    ctx.save();
    ctx.setLineDash(zone.enabled?[]:[6,4]);
    ctx.lineWidth=2;
    ctx.strokeStyle=boundary?'#2ecc40':'#ff4136';
    ctx.fillStyle=boundary?'rgba(46,204,64,.12)':'rgba(255,65,54,.16)';
    ctx.beginPath();
    if(zone.shape==='circle'){
      const c=project(zone.center_x,zone.center_y);
      ctx.arc(c[0],c[1],Math.max(2,zone.radius_mm*scale),0,Math.PI*2);
    } else {
      const pts=zonePoints(zone);
      pts.forEach(function(p,i){
        const c=project(p[0],p[1]);
        if(i===0) ctx.moveTo(c[0],c[1]); else ctx.lineTo(c[0],c[1]);
      });
      ctx.closePath();
    }
    ctx.fill();ctx.stroke();
    const label=zonePoints(zone)[0]||[zone.center_x,zone.center_y];
    const anchor=project(zone.shape==='circle'?zone.center_x:label[0],
                         zone.shape==='circle'?zone.center_y:label[1]);
    ctx.setLineDash([]);
    ctx.fillStyle='#17221b';ctx.font='12px system-ui';
    ctx.fillText(zone.name+(zone.enabled?'':' (off)'),anchor[0]+6,anchor[1]-6);
    ctx.restore();
  }
  function render(){
    const box=bounds();
    ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#fbfdfb';ctx.fillRect(0,0,canvas.width,canvas.height);
    if(!box){
      ctx.fillStyle='#74817a';ctx.font='14px system-ui';
      ctx.fillText('Add a zone to see the garden map',14,26);
      return;
    }
    const scale=Math.min(canvas.width/(box.maxX-box.minX),canvas.height/(box.maxY-box.minY));
    const project=function(x,y){return [(x-box.minX)*scale,(y-box.minY)*scale];};
    zones.forEach(function(zone){drawZone(zone,project,scale);});
    items.plants.forEach(function(plant){
      const c=project(plant.x,plant.y);
      ctx.strokeStyle='#1a7f4b';ctx.lineWidth=1.5;
      ctx.beginPath();ctx.arc(c[0],c[1],Math.max(3,(plant.radius||0)*scale),0,Math.PI*2);ctx.stroke();
      ctx.fillStyle='#1a7f4b';ctx.beginPath();ctx.arc(c[0],c[1],2.5,0,Math.PI*2);ctx.fill();
    });
    items.weeds.forEach(function(weed){
      const c=project(weed.x,weed.y);
      ctx.fillStyle='#b3002d';ctx.beginPath();ctx.arc(c[0],c[1],3,0,Math.PI*2);ctx.fill();
    });
    ctx.fillStyle='#74817a';ctx.font='12px system-ui';
    ctx.fillText('X '+Math.round(box.minX)+'…'+Math.round(box.maxX)+' mm, Y '
      +Math.round(box.minY)+'…'+Math.round(box.maxY)+' mm (Y increases downwards)',10,
      canvas.height-10);
  }
  document.getElementById('zone-load-items').addEventListener('click',async function(){
    const entry=canvas.dataset.entry||'';
    if(!entry){status.textContent='Select a FarmBot in the app options first';return;}
    status.textContent='Loading plants and weeds…';
    try{
      const response=await fetch('api/vision/images?entry_id='+encodeURIComponent(entry));
      if(!response.ok) throw new Error('HTTP '+response.status);
      const data=await response.json();
      items={plants:data.plants||[],weeds:data.weeds||[]};
      status.textContent=items.plants.length+' plants and '+items.weeds.length+' FarmBot weeds shown';
      render();
    }catch(err){status.textContent='Could not load garden items: '+err.message;}
  });
  shape.addEventListener('change',showShapeFields);
  kind.addEventListener('change',applyKindDefaults);
  showShapeFields();applyKindDefaults();render();
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
    await soil_jobs.close()
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
    return HTMLResponse(
        f"""<!doctype html><html lang=en><head><meta charset=utf-8>
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
.modal-controls[hidden]{{display:none}}
.queue-dialog{{width:min(95vw,900px)}}.button-row{{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center}}
.weed-dialog{{width:min(95vw,900px)}}.weed-image-wrap{{position:relative;display:inline-block;margin:auto}}
.weed-marker{{position:absolute;width:34px;height:34px;margin-left:-17px;margin-top:-17px;pointer-events:none;
border-radius:50%;border:3px solid #168cff;box-shadow:0 0 0 1px #001b3d}}
.weed-view-toggle button[aria-pressed=true]{{background:#1672c4;color:white;box-shadow:inset 0 0 0 2px #0b4779}}
.plant-view-toggle button[aria-pressed=true]{{background:#1672c4;color:white;box-shadow:inset 0 0 0 2px #0b4779}}
td.actions{{min-width:9rem}}.actions-group{{display:flex;flex-direction:column;align-items:stretch;gap:.4rem}}
.actions-group form{{margin:0}}.actions-group button{{width:100%;padding:.45rem .8rem;font-size:.9rem}}
.actions-group button[data-artifacts]{{background:#e4ede7;color:var(--dark)}}
.hint{{display:inline-flex;align-items:center;justify-content:center;width:1.1em;height:1.1em;
border-radius:50%;background:var(--muted);color:white;font-size:.72em;font-weight:bold;
margin-left:.3em;cursor:help;vertical-align:middle;line-height:1}}
</style></head><body><header><h1>🌱 FarmBot Vision</h1><nav><a href="./">Analysis</a><a href="soil-height">Soil height</a><a href="settings">Calibration</a><a href="weed-settings">Weed settings</a><a href="canopy-settings">Canopy fusion</a><a href="zones">Boundaries &amp; zones</a><a href="api/health">Health JSON</a></nav></header>
<main>{body}</main></body></html>"""
    )


def hint(text: str) -> str:
    """A small hover-tooltip badge ("?") explaining a nearby form field."""
    return f'<span class=hint tabindex=0 title="{escape(text, quote=True)}">?</span>'


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
            "canopy_fusion": canopy_fusion_settings_store.load().model_dump(),
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
        center_x = r.get("recorded_center_x")
        center_y = r.get("recorded_center_y")
        center = (
            f"({float(center_x):.1f}, {float(center_y):.1f})"
            if center_x is not None and center_y is not None
            else "(unavailable)"
        )
        details = {
            "formula": (
                f"Current {r['current_radius_mm']:.1f} mm; Recommended = "
                f"{r['recommended_protection_radius_mm']:.1f} mm. "
                f"Plant center = {center}; crop: {r.get('crop_slug', 'unknown')}. "
                + (
                    f"Fused from {r.get('fusion_view_count', 0)} calibrated views; "
                    f"angular coverage {float(r.get('fusion_angular_coverage') or 0):.0%}; "
                    f"corroborated pixels "
                    f"{float(r.get('fusion_corroborated_fraction') or 0):.0%}."
                    if r.get("fused_canopy")
                    else "Measured from one image or consolidated per-image radii."
                )
            )
        }
        details_json = escape(json.dumps(details, separators=(",", ":")), quote=True)
        if r.get("composite_path"):
            clean_url = escape(
                f"artifact/{Path(r['composite_path']).name}",
                quote=True,
            )
            overlay_url = (
                escape(
                    f"artifact/{Path(r['composite_overlay_path']).name}",
                    quote=True,
                )
                if r.get("composite_overlay_path")
                else ""
            )
            composite_button = (
                f'<button type=button data-composite-clean="{clean_url}" '
                f'data-composite-overlay="{overlay_url}" '
                f'data-details="{details_json}">View</button>'
            )
            if r.get("fusion_diagnostic_path"):
                fusion_url = escape(
                    json.dumps(
                        [f"artifact/{Path(r['fusion_diagnostic_path']).name}"],
                        separators=(",", ":"),
                    ),
                    quote=True,
                )
                composite_button += (
                    f'<button type=button data-artifacts="{fusion_url}" '
                    f'data-details="{details_json}">Fusion</button>'
                )
            return composite_button
        paths = r.get("artifact_paths") or []
        if not paths and r.get("overlay_path"):
            paths = [r["overlay_path"]]
        if r.get("fusion_diagnostic_path"):
            paths = [*paths, r["fusion_diagnostic_path"]]
        urls = [f"artifact/{Path(path).name}" for path in paths if path]
        if not urls:
            return "<span class=muted>None</span>"
        artifacts_json = escape(json.dumps(urls, separators=(",", ":")), quote=True)
        return (
            f'<button type=button data-artifacts="{artifacts_json}" '
            f'data-details="{details_json}">View</button>'
        )

    def _format_center(
        x: object,
        y: object,
        *,
        fallback: str = "Unavailable for older result",
    ) -> str:
        if x is None or y is None:
            return f"<span class=muted>{escape(fallback)}</span>"
        try:
            return f"X {float(x):.1f}, Y {float(y):.1f}"
        except (TypeError, ValueError):
            return f"<span class=muted>{escape(fallback)}</span>"

    def _format_coordinates(x: object, y: object) -> str:
        if x is None or y is None:
            return "<span class=muted>Unavailable</span>"
        return f"({float(x):.1f}, {float(y):.1f})"

    def _review_controls(r: dict) -> str:
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if not r.get("calibrated", 1):
            return (
                "<span class=warn>Calibration required to apply a radius</span>"
                f'<form method=post action="recommendations/{r["measurement_id"]}/reject">'
                f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/reject">'
                "Reject</button></form><small class=action-message></small>"
            )
        if r["recommended_protection_radius_mm"] > r["current_radius_mm"]:
            approve_label = "Apply radius"
        elif r["recommended_protection_radius_mm"] < r["current_radius_mm"]:
            approve_label = "Apply smaller radius"
        else:
            approve_label = "Approve observation"
        return (
            f'<form method=post action="recommendations/{r["measurement_id"]}/approve">'
            f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/approve">'
            f"{approve_label}</button></form>"
            f'<form method=post action="recommendations/{r["measurement_id"]}/reject">'
            f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/reject">Reject</button></form>'
            + (
                f'<form method=post action="recommendations/{r["measurement_id"]}/move-center">'
                f'<button class=review-action data-url="recommendations/{r["measurement_id"]}/move-center">'
                "Move center</button></form>"
                if r.get("center_misaligned")
                else ""
            )
            + "<small class=action-message></small>"
        )

    measurement_rows = "".join(
        f'<tr class=review-item id="measurement-{r["measurement_id"]}"><td>{escape(r["crop_slug"])}</td>'
        f"<td>{_format_coordinates(r.get('recorded_center_x'), r.get('recorded_center_y'))}</td>"
        f"<td>{r['current_radius_mm']:.1f}</td>"
        f"<td>{r['maximum_accepted_canopy_radius_mm']:.1f}</td><td>{r['recommended_protection_radius_mm']:.1f}</td>"
        f"<td>{r['confidence']:.2f}</td>"
        f"<td>{escape(r['decision'])}</td><td>{escape(r['reason'])}</td>"
        f"<td class=actions><div class=actions-group>{_artifact_button(r)}{_review_controls(r)}</div></td></tr>"
        for r in rows
        if not r.get("vegetation_absent")
    )
    last = jobs.last
    curve_rows = "".join(
        f"<tr><td>{escape(slug)}</td><td>{escape(str(curve))}</td><td>diameter mm</td></tr>"
        for slug, curve in curves.items()
    )

    removal_rows = "".join(
        f'<tr class=review-item id="measurement-{r["measurement_id"]}">'
        f"<td>{escape(r['crop_slug'])}</td>"
        f"<td>{_format_center(r.get('recorded_center_x'), r.get('recorded_center_y'))}</td>"
        f"<td>{_format_center(r.get('recommended_center_x'), r.get('recommended_center_y'), fallback='No move suggested')}</td>"
        f"<td>{r['absent_observations']}</td><td>{r['confidence']:.2f}</td>"
        f"<td>{escape(r['reason'])}</td><td>{_artifact_button(r)}</td><td>"
        f'<form method=post action="removals/{r["measurement_id"]}/approve"><button class=review-action '
        f'data-url="removals/{r["measurement_id"]}/approve">Approve removal</button></form>'
        f'<form method=post action="removals/{r["measurement_id"]}/keep"><button class=review-action '
        f'data-url="removals/{r["measurement_id"]}/keep">Keep plant</button></form>'
        + (
            f'<form method=post action="removals/{r["measurement_id"]}/move-center"><button class=review-action '
            f'data-url="removals/{r["measurement_id"]}/move-center">Move center</button></form>'
            if r.get("center_misaligned")
            else ""
        )
        + "<small class=action-message></small></td></tr>"
        for r in rows
        if r.get("vegetation_absent")
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
            "<input type=hidden name=confirm_override value=true>"
            f'<button class=curve-action data-action=apply data-url="curve-proposals/{proposal["id"]}/apply">Use value</button></form>'
            f'<form method=post action="curve-proposals/{proposal["id"]}/discard-new"><button class=curve-action '
            f'data-action=discard-new data-url="curve-proposals/{proposal["id"]}/discard-new">Discard new</button></form>'
            f'<form method=post action="curve-proposals/{proposal["id"]}/discard-old"><button class=curve-action '
            f'data-action=discard-old data-url="curve-proposals/{proposal["id"]}/discard-old">Discard old</button></form>'
            "<small class=action-message></small></td></tr>"
        )
    flagged_curve_rows = "".join(proposal_rows)
    decision_rows = "".join(
        f"<tr><td>{escape(row['created_at'])}</td><td>{escape(row['measurement_id'])}</td>"
        f"<td>{escape(row['action'])}</td></tr>"
        for row in database.recent_decisions()
    )
    pending_weeds = database.pending_weed_detections()
    weeds_by_image: dict[int, list[dict]] = {}
    for w in pending_weeds:
        weeds_by_image.setdefault(w["image_id"], []).append(w)

    def _weed_view_button(w: dict) -> str:
        if not w.get("overlay_path"):
            return "<span class=muted>None</span>"
        siblings = [str(other["detection_id"]) for other in weeds_by_image.get(w["image_id"], [])]
        marker = {
            "overlayArtifact": f"artifact/{Path(w['overlay_path']).name}",
            "reviewArtifact": (
                f"artifact/{Path(w['review_path']).name}" if w.get("review_path") else None
            ),
            "x": w.get("center_px_x"),
            "y": w.get("center_px_y"),
            "width": w.get("processed_width"),
            "height": w.get("processed_height"),
            "detectionId": str(w["detection_id"]),
            "siblings": siblings,
            "areaMm2": w["area_mm2"],
            "confidence": w["confidence"],
            "observations": w.get("observation_count", 1),
            "verifierConfidence": w.get("verifier_confidence"),
        }
        marker_json = escape(json.dumps(marker, separators=(",", ":")), quote=True)
        return f'<button type=button class=weed-view data-weed="{marker_json}">View</button>'

    def _weed_row(w: dict) -> str:
        verifier = (
            f"{float(w['verifier_confidence']):.2f}"
            if w.get("verifier_confidence") is not None
            else "—"
        )
        return (
            f'<tr class=review-item id="weed-{w["detection_id"]}"><td>{w["image_id"]}</td>'
            f"<td>{w['x']:.1f}, {w['y']:.1f}, {w['z']:.1f}</td>"
            f"<td>{w['area_mm2']:.1f}</td><td>{w.get('observation_count', 1)}</td>"
            f"<td>{float(w.get('heuristic_confidence') or w['confidence']):.2f}</td>"
            f"<td>{verifier}</td><td>{_weed_view_button(w)}</td><td>"
            f'<form><button class=review-action data-url="weeds/{w["detection_id"]}/approve">'
            "Create weed</button></form>"
            f'<form><button class=review-action data-url="weeds/{w["detection_id"]}/reject">'
            "Reject as mulch/soil</button></form>"
            f'<button class=review-action data-url="weeds/{w["detection_id"]}/label/crop">'
            "Crop</button>"
            f'<button class=review-action data-url="weeds/{w["detection_id"]}/label/fungus_moss">'
            "Fungus/moss</button>"
            f'<button class=review-action data-url="weeds/{w["detection_id"]}/label/hardware_other">'
            "Hardware/other</button>"
            "<small class=action-message></small></td></tr>"
        )

    weed_rows = "".join(_weed_row(w) for w in pending_weeds)
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
<section class=card><h2>Automatic decision threshold</h2>
<p><b>{settings.minimum_auto_confidence:.0%} confidence</b></p>
<p class=muted>Set <code>minimum_auto_confidence</code> in the add-on configuration.
It affects automatic changes only; every result remains manually reviewable.</p></section>
<section class=card><h2>Analysis</h2><p><span id=queue-count>{len(jobs.queued_image_ids)}</span> queued</p>
<div class=button-row><form method=post action="analyse"><button>Analyse queue</button></form>
<button id=queue-open type=button>Add to queue</button></div></section></div>
<section class=card><h2>Last job</h2>
<p>{escape(last.get("message", "Never run"))}</p>
<div class=grid>
<div><b>Timing</b><p class=muted>Duration {last.get("duration_seconds", "—")} s · CPU {last.get("cpu_seconds", "—")} s · peak {last.get("peak_memory_mb", "—")} MB</p></div>
<div><b>Images</b><p class=muted>{last.get("images_processed", "—")} processed · {last.get("uncalibrated_images", 0)} uncalibrated</p></div>
<div><b>Plants</b><p class=muted>{last.get("plants_measured", "—")} measured · {last.get("uncertain", "—")} uncertain · {last.get("skipped", "—")} skipped</p></div>
<div><b>Dimensions</b><p class=muted>source {escape(_dims(last.get("source_dimensions")))} · oriented {escape(_dims(last.get("oriented_dimensions")))} · processed {escape(_dims(last.get("processed_dimensions")))}</p></div>
<div><b>Calibration</b><p class=muted>source {escape(str(last.get("calibration_source") or "—"))}</p></div>
<div><b>Contract</b><p class=muted>{escape(str(last.get("contract_version") or CONTRACT_VERSION))} · min integration {MINIMUM_INTEGRATION_VERSION}</p></div>
<div><b>Zones</b><p class=muted>{last.get("zone_blocked_weeds", 0)} weeds · {last.get("zone_blocked_radius", 0)} radius increases blocked</p></div>
</div>
<p><b>Calibration warnings</b></p><ul>{warning_html}</ul>
<p><b>Skip reasons</b></p><ul>{skip_html}</ul></section>
<section class=card><h2>Measurements</h2><table><thead><tr><th>Crop</th><th>Coordinates (x, y)</th><th>Current</th><th>Max leaf</th><th>Recommended</th><th>Confidence</th><th>Decision</th><th>Reason</th><th>Actions</th></tr></thead><tbody>{measurement_rows or "<tr><td colspan=9>No measurements yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Removed / missing plants</h2><table><thead><tr><th>Crop</th><th>Recorded center (X, Y mm)</th><th>Move center to (X, Y mm)</th><th>Absent looks</th><th>Confidence</th><th>Reason</th><th>Diagnostic</th><th>Review</th></tr></thead><tbody>{removal_rows or "<tr><td colspan=8>No confirmed missing plants</td></tr>"}</tbody></table></section>
<section class=card><h2>Detected weeds</h2><p class=muted>Unowned vegetation outside known plant protection areas.</p>
<table><thead><tr><th>Image</th><th>Coordinates</th><th>Area mm²</th><th>Looks</th>
<th>Heuristic</th><th>Verifier</th><th>View</th><th>Review / training label</th></tr></thead>
<tbody>{weed_rows or "<tr><td colspan=8>No weed recommendations</td></tr>"}</tbody></table></section>
<section class=card><h2>Growth-curve updates</h2><p class=muted>Flagged per-plant diameter points require review.</p><table><tbody>{flagged_curve_rows or "<tr><td>No flagged curve updates</td></tr>"}</tbody></table></section>
<section class=card><h2>Crop protection spread proposals</h2><p class=muted>Monotonic and limited to 10 points. FarmBot values are diameters; assignment requires approval.</p><table><tbody>{curve_rows or "<tr><td>No curve is ready</td></tr>"}</tbody></table></section>
<section class=card><h2>Approval and rollback history</h2><table><tbody>{decision_rows or "<tr><td>No decisions yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Safety warning</h2><p class=warn>Early experimental vision results must not be the sole basis for destructive automatic weeding.</p></section>
<div id=overlay-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Analysis diagnostic"><figure>
<button id=overlay-modal-close class=modal-close type=button aria-label=Close>&times;</button>
<div id=plant-view-toggle class="modal-controls plant-view-toggle" role=group aria-label="Plant image view" hidden>
<button id=plant-modal-without-overlay type=button aria-pressed=true>Original images</button>
<button id=plant-modal-with-overlay type=button aria-pressed=false>Show mask overlay</button>
</div>
<img id=overlay-modal-img alt="Plant analysis diagnostic"><figcaption id=overlay-modal-details></figcaption>
<p id=overlay-modal-legend class=legend>Cyan circle = original radius; red circle = planned radius.</p>
<div id=artifact-controls class=modal-controls><button id=overlay-modal-prev type=button>Previous</button><span id=overlay-modal-counter></span><button id=overlay-modal-next type=button>Next</button></div>
</figure></div>
<div id=weed-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Weed review">
<figure class=weed-dialog><button id=weed-modal-close class=modal-close type=button aria-label=Close>&times;</button>
<div class="modal-controls weed-view-toggle" role=group aria-label="Weed image view">
<button id=weed-modal-without-overlay type=button aria-pressed=true>Without overlay</button>
<button id=weed-modal-with-overlay type=button aria-pressed=false>With overlay</button>
</div>
<div class=weed-image-wrap><img id=weed-modal-img alt="Weed detection"><div id=weed-modal-marker class=weed-marker hidden></div></div>
<figcaption id=weed-modal-details></figcaption>
<p class=legend>Blue circle = the weed being reviewed; red circles = other detected weeds in this image.</p>
<div class=modal-controls>
<button id=weed-modal-accept type=button>Accept weed</button>
<button id=weed-modal-reject type=button>Reject weed</button>
<button id=weed-modal-accept-all type=button>Accept all weeds</button>
</div>
<small id=weed-modal-message class=action-message></small>
</figure></div>
<div id=queue-modal class=overlay-modal hidden role=dialog aria-modal=true aria-label="Add images to analysis queue">
<figure class=queue-dialog><button id=queue-close class=modal-close type=button aria-label=Close>&times;</button>
<h2>Add images to analysis queue</h2>
<div class=button-row><label>From <input id=queue-from type=datetime-local></label>
<label>To <input id=queue-to type=datetime-local></label>
<button id=queue-refresh type=button>Refresh</button><label><input id=queue-select-all type=checkbox> Select all</label></div>
<p id=queue-message class=muted></p><table><thead><tr><th>Select</th><th>Coordinates (x, y, z)</th>
<th>Plants present</th><th>Date taken</th></tr></thead><tbody id=queue-image-rows></tbody></table>
<div class=button-row><button id=queue-add type=button>Add selected images to queue</button></div>
</figure></div><script>{_DASHBOARD_JS}</script>"""
    return layout(request, body)


@app.post("/analyse")
async def analyse(background: BackgroundTasks) -> RedirectResponse:
    background.add_task(jobs.run, trigger="manual")
    return RedirectResponse("./", status_code=303)


@app.get("/api/analysis/images")
async def analysis_images(
    date_from: datetime | None = None, date_to: datetime | None = None
) -> JSONResponse:
    if not settings.selected_config_entry_id:
        raise HTTPException(400, "Select a FarmBot before loading images")
    now = datetime.now(UTC)
    date_to = date_to or now
    date_from = date_from or (date_to - timedelta(hours=72))
    if date_from.tzinfo is None:
        date_from = date_from.replace(tzinfo=UTC)
    if date_to.tzinfo is None:
        date_to = date_to.replace(tzinfo=UTC)
    if date_from > date_to:
        raise HTTPException(422, "From must be before to")
    hours = max(1, min(720, int((now - date_from).total_seconds() / 3600) + 1))
    try:
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=settings.selected_config_entry_id,
                image_lookback_hours=hours,
            )
        )
    except HomeAssistantError as exc:
        raise HTTPException(502, str(exc)) from exc
    calibration = database.active_calibration(settings.selected_config_entry_id)
    width, height = settings.analysis_width, settings.analysis_height
    items = []
    for image in sorted(inventory.images, key=lambda item: item.created_at, reverse=True):
        if not date_from <= image.created_at <= date_to:
            continue
        present = []
        for plant in inventory.plants:
            if calibration is not None:
                px, py = garden_to_pixel(
                    plant.x,
                    plant.y,
                    image.meta.x,
                    image.meta.y,
                    width,
                    height,
                    calibration,
                )
                is_present = (
                    -plant.radius * calibration.pixels_per_mm_x
                    <= px
                    <= width + plant.radius * calibration.pixels_per_mm_x
                    and -plant.radius * calibration.pixels_per_mm_y
                    <= py
                    <= height + plant.radius * calibration.pixels_per_mm_y
                )
            else:
                # Useful conservative fallback before calibration: images still
                # remain selectable and nearby plants are listed approximately.
                is_present = (
                    abs(plant.x - image.meta.x) <= 500 and abs(plant.y - image.meta.y) <= 400
                )
            if is_present:
                present.append({"id": plant.id, "name": plant.name})
        items.append(
            {
                "id": image.id,
                "created_at": image.created_at.isoformat(),
                "x": image.meta.x,
                "y": image.meta.y,
                "z": image.meta.z,
                "plants": present,
            }
        )
    return JSONResponse(
        {"images": items, "date_from": date_from.isoformat(), "date_to": date_to.isoformat()}
    )


@app.post("/analysis/queue")
async def add_analysis_queue(request: QueueImagesRequest) -> JSONResponse:
    return JSONResponse({"queue_length": jobs.add_to_queue(request.image_ids)})


def _soil_artifacts(paths: list[str]) -> str:
    links = []
    for path in paths:
        if not path:
            continue
        url = f"artifact/{Path(path).name}"
        label = Path(path).stem.rsplit("-", 1)[-1]
        links.append(
            f'<a href="{escape(url, quote=True)}" target=_blank rel=noopener>{escape(label)}</a>'
        )
    return " ".join(links) or "<span class=muted>None</span>"


@app.get("/soil-height", response_class=HTMLResponse)
async def soil_height_page(request: Request) -> HTMLResponse:
    entry_id = settings.selected_config_entry_id
    inventory = None
    sites = []
    inventory_error = ""
    calibration = database.active_soil_calibration(entry_id) if entry_id else None
    planning_baseline = calibration.baseline_mm if calibration else 15
    if entry_id:
        try:
            inventory, sites = await soil_jobs.safe_sites(entry_id, planning_baseline)
        except HomeAssistantError as exc:
            inventory_error = str(exc)
    measurements = database.recent_soil_measurements(entry_id, 200)
    persisted_job = database.latest_soil_job(entry_id)
    current_job = soil_jobs.current if soil_jobs.running else (persisted_job or soil_jobs.current)

    latest_by_point: dict[int, dict] = {}
    for measurement in measurements:
        latest_by_point.setdefault(int(measurement["point_id"]), measurement)

    point_rows = ""
    point_options = ""
    retry_ids: list[int] = []
    if inventory:
        for site in sites:
            measurement = latest_by_point.get(site.point_id)
            status = measurement["status"] if measurement else "not measured"
            proposed = (
                f"{measurement['proposed_z_mm']:.0f} mm"
                if measurement and measurement["proposed_z_mm"] is not None
                else "—"
            )
            uncertainty = (
                f"±{measurement['uncertainty_mm']:.1f} mm"
                if measurement and measurement["uncertainty_mm"] is not None
                else "—"
            )
            confidence = f"{100 * measurement['confidence']:.0f}%" if measurement else "—"
            reason = escape(measurement["reason"] if measurement else "")
            diagnostics = _soil_artifacts(measurement["artifact_paths"] if measurement else [])
            if measurement and measurement["status"] == "failed":
                retry_ids.append(site.point_id)
            apply_control = ""
            if measurement and measurement["status"] == "valid":
                measurement_id = escape(measurement["measurement_id"], quote=True)
                apply_control = (
                    f'<form method=post action="soil/measurements/{measurement_id}/apply">'
                    "<button type=submit>Apply</button></form>"
                    f'<form method=post action="soil/measurements/{measurement_id}/reject">'
                    "<button type=submit>Reject</button></form>"
                )
            point_rows += (
                "<tr>"
                f'<td><input form=measure-points type=checkbox name=point_ids value="{site.point_id}"></td>'
                f"<td>{site.point_id}</td><td>{escape(site.point_name)}</td>"
                f"<td>{site.expected_x:.1f}, {site.expected_y:.1f}</td>"
                f"<td>{site.capture_x:.1f}, {site.capture_y:.1f}</td>"
                f"<td>{site.relocation_distance_mm:.1f} mm</td>"
                f"<td>{site.point_updated_at.date().isoformat()}</td>"
                f"<td>{site.expected_z:.1f} mm</td>"
                f"<td>{proposed}</td><td>{uncertainty}</td><td>{confidence}</td>"
                f"<td>{escape(status)}</td><td>{reason}</td><td>{diagnostics}</td>"
                f"<td>{apply_control}</td></tr>"
            )
            point_options += (
                f'<option value="{site.point_id}">{escape(site.point_name)}: clear soil '
                f"({site.capture_x:.0f}, {site.capture_y:.0f})</option>"
            )

    valid_measurements = [
        item
        for item in measurements
        if item["status"] == "valid"
        and item.get("capture_x") is not None
        and item.get("capture_y") is not None
        and item.get("point_updated_at")
    ]
    measurement_rows = "".join(
        "<tr>"
        f"<td><input form=apply-selected type=checkbox name=measurement_ids "
        f'value="{escape(item["measurement_id"], quote=True)}"></td>'
        f"<td>{escape(item['point_name'])}</td>"
        f"<td>{item['expected_x']:.1f}, {item['expected_y']:.1f}</td>"
        f"<td>{item['capture_x']:.1f}, {item['capture_y']:.1f}</td>"
        f"<td>{item['old_z_mm']:.1f} mm</td>"
        f"<td>{item['proposed_z_mm']:.0f} mm</td>"
        f"<td>{100 * item['confidence']:.0f}%</td>"
        f"<td>{escape(item['reason'])}</td></tr>"
        for item in valid_measurements
    )
    point_count = len(inventory.points) if inventory else 0
    site_count = len(sites)
    warning = (
        "<p class=warn>Fewer than three stale soil points currently have a nearby "
        "clear-soil replacement. FarmBot soil-height interpolation needs at least "
        "three measured points.</p>"
        if site_count < 3
        else ""
    )
    motion = inventory.motion if inventory else None
    motion_summary = (
        f"connected={motion.connected}, busy={motion.busy}, emergency stop={motion.locked}, "
        f"position={escape(json.dumps(motion.position))}"
        if motion
        else "unavailable"
    )
    calibration_summary = (
        f"Active calibration #{calibration.calibration_id}: "
        f"{calibration.processed_width}×{calibration.processed_height}, "
        f"{calibration.baseline_mm:.0f} mm baseline, "
        f"{calibration.residual_mm:.1f} mm residual"
        if calibration
        else "No active soil calibration. Complete the guided calibration before measuring."
    )
    default_capture_z = calibration.capture_z if calibration else 0
    default_baseline = calibration.baseline_mm if calibration else 15
    capture_z_hint = hint(
        "The FarmBot Z-axis (height) position the gantry moves to before taking soil "
        "photos. During calibration the bot also steps down 25 mm and 50 mm from this "
        "height to build the depth curve; during measurement it only captures here. "
        "Use the same Capture Z every time — changing it invalidates the calibration."
    )
    baseline_hint = hint(
        "How far, in mm, the camera shifts sideways (along Y) between the shots taken "
        "at each point. This lateral shift is the 'virtual stereo' separation used to "
        "compute soil depth from the difference between images, similar to the "
        "distance between two eyes. It must match the value used for calibration — "
        "changing it requires recalibrating."
    )
    job_message = escape(str(current_job.get("message", "Not run")))
    job_status = escape(str(current_job.get("status", "idle")))
    retry_values = "".join(
        f'<input type=hidden name=point_ids value="{point_id}">' for point_id in retry_ids
    )
    live_refresh = (
        "<script>setTimeout(()=>location.reload(),3000)</script>" if soil_jobs.running else ""
    )
    body = f"""
<h2>Supplemental soil-height measurement</h2>
<p>Finds plant- and weed-free soil within 200 mm of FarmBot soil points that have
not been updated for more than 14 days. Measurements are captured at those clear
locations and, after review, replace the assigned stale point.</p>
{warning}
<section class=grid>
 <div class=card><h3>Bot</h3><p>{escape(entry_id or "No FarmBot selected")}</p>
 <p class=muted>{motion_summary}</p><p>{escape(inventory_error)}</p></div>
 <div class=card><h3>Calibration</h3><p>{escape(calibration_summary)}</p>
 <p class=warn>Recalibrate after moving, rotating, or refocusing the camera.</p></div>
 <div class=card><h3>Current job</h3><p><strong>{job_status}</strong>: {job_message}</p>
 <form method=post action=soil/stop><button type=submit>Stop after current point</button></form></div>
</section>
<section class=card>
 <h3>Guided calibration</h3>
 <p>Choose one of the calculated clear-soil sites. Enter the manually measured
camera-to-soil distance at the capture Z, then confirm that a 50 mm movement
toward the soil is safe.</p>
 <form method=post action=soil/calibrate>
  <label>Clear soil site <select name=point_id required>{point_options}</select></label>
  <label>Camera-to-soil distance (mm) <input type=number min=1 step=0.1
   name=reference_distance_mm required></label>
  <label>Capture Z (mm){capture_z_hint} <input type=number step=0.1 name=capture_z value=0 required></label>
  <label>Baseline (mm){baseline_hint} <input type=number min=5 max=30 step=0.1
   name=baseline_mm value=15 required></label>
  <label><input type=checkbox name=safety_confirm required> I confirm the automated
   50 mm movement toward the soil is safe</label>
  <button type=submit>Calibrate</button>
 </form>
</section>
<section class=card>
 <h3>Clear-soil replacements ({site_count} from {point_count} existing points)</h3>
 <p class=muted>Each candidate has a 75 mm clear-soil margin, expanded for the
stereo movement, around all current FarmBot plants and weeds, the latest
detected plant canopies, and pending or created Vision weeds. Fresh points and
points without a trustworthy update date are not replaced.</p>
 <form id=measure-points method=post action=soil/measure>
  <label>Capture Z (mm){capture_z_hint} <input type=number step=0.1 name=capture_z
   value="{default_capture_z:g}" required></label>
  <label>Baseline (mm){baseline_hint} <input type=number min=5 max=30 step=0.1 name=baseline_mm
   value="{default_baseline:g}" required></label>
  <button type=submit name=mode value=selected>Measure selected</button>
  <button type=submit name=mode value=all>Measure all</button>
 </form>
 <form method=post action=soil/measure>{retry_values}
  <input type=hidden name=capture_z value="{default_capture_z:g}">
  <input type=hidden name=baseline_mm value="{default_baseline:g}">
  <button type=submit name=mode value=retry {"disabled" if not retry_ids else ""}>Retry failed</button>
 </form>
 <table><thead><tr><th>Select</th><th>ID</th><th>Replaces</th><th>Old X, Y</th>
 <th>Clear X, Y</th><th>Move</th><th>Last updated</th><th>Current Z</th>
 <th>Proposed Z</th><th>Uncertainty</th><th>Confidence</th>
 <th>Status</th><th>Message</th><th>Diagnostics</th><th>Review</th></tr></thead>
 <tbody>{point_rows or "<tr><td colspan=15>No stale point has a safe clear-soil site within 200 mm.</td></tr>"}</tbody></table>
</section>
<section class=card>
 <h3>Pending valid results</h3>
 <form id=apply-selected method=post action=soil/apply-selected>
  <button type=submit>Apply selected</button>
 </form>
 <table><thead><tr><th>Select</th><th>Point</th><th>Old X, Y</th><th>New X, Y</th>
 <th>Old Z</th><th>Proposed Z</th>
 <th>Confidence</th><th>Quality result</th></tr></thead>
 <tbody>{measurement_rows or "<tr><td colspan=8>No unapplied valid results.</td></tr>"}</tbody></table>
</section>
 {live_refresh}"""  # noqa: S608 - HTML template; no SQL is constructed here.
    return layout(request, body, "Soil height · FarmBot Vision")


@app.get("/api/soil/points")
async def soil_points_api() -> JSONResponse:
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        raise HTTPException(409, "No FarmBot config entry is selected")
    calibration = database.active_soil_calibration(entry_id)
    inventory, sites = await soil_jobs.safe_sites(
        entry_id, calibration.baseline_mm if calibration else 15
    )
    return JSONResponse(
        {
            "inventory": inventory.model_dump(mode="json"),
            "safe_sites": [site.model_dump(mode="json") for site in sites],
        }
    )


@app.get("/api/soil/job")
async def soil_job_api() -> JSONResponse:
    return JSONResponse(soil_jobs.current)


@app.post("/soil/calibrate")
async def start_soil_calibration(
    point_id: int = Form(...),
    reference_distance_mm: float = Form(...),
    capture_z: float = Form(0),
    baseline_mm: float = Form(15),
    safety_confirm: bool = Form(False),
) -> RedirectResponse:
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        raise HTTPException(409, "No FarmBot config entry is selected")
    if not safety_confirm:
        raise HTTPException(422, "Confirm that the 50 mm calibration movement is safe")
    _inventory, sites = await soil_jobs.safe_sites(entry_id, baseline_mm)
    site = next((item for item in sites if item.point_id == point_id), None)
    if site is None:
        raise HTTPException(404, "Clear-soil calibration site not found")
    try:
        soil_jobs.start_calibration(
            config_entry_id=entry_id,
            point_id=point_id,
            capture_z=capture_z,
            baseline_mm=baseline_mm,
            reference_distance_mm=reference_distance_mm,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return RedirectResponse("../soil-height", status_code=303)


@app.post("/soil/measure")
async def start_soil_measurement(
    point_ids: Annotated[list[int] | None, Form()] = None,
    mode: str = Form("selected"),
    capture_z: float = Form(0),
    baseline_mm: float = Form(15),
) -> RedirectResponse:
    entry_id = settings.selected_config_entry_id
    if not entry_id:
        raise HTTPException(409, "No FarmBot config entry is selected")
    if mode == "all":
        _inventory, sites = await soil_jobs.safe_sites(entry_id, baseline_mm)
        point_ids = [site.point_id for site in sites]
    point_ids = point_ids or []
    try:
        soil_jobs.start_measurements(
            config_entry_id=entry_id,
            point_ids=point_ids,
            capture_z=capture_z,
            baseline_mm=baseline_mm,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return RedirectResponse("../soil-height", status_code=303)


@app.post("/soil/stop")
async def stop_soil_measurement() -> RedirectResponse:
    if soil_jobs.running:
        soil_jobs.request_stop()
    return RedirectResponse("../soil-height", status_code=303)


async def _apply_soil_measurement(measurement_id: str) -> dict:
    measurement = database.soil_measurement(measurement_id)
    if (
        measurement is None
        or measurement["status"] != "valid"
        or measurement["proposed_z_mm"] is None
        or measurement.get("capture_x") is None
        or measurement.get("capture_y") is None
        or not measurement.get("point_updated_at")
    ):
        raise HTTPException(404, "Applicable soil result not found")
    apply_request = ApplySoilHeightRequest(
        config_entry_id=measurement["config_entry_id"],
        point_id=measurement["point_id"],
        measurement_id=measurement["measurement_id"],
        expected_x=measurement["expected_x"],
        expected_y=measurement["expected_y"],
        expected_z=measurement["old_z_mm"],
        expected_updated_at=measurement["point_updated_at"],
        recommended_x=measurement["capture_x"],
        recommended_y=measurement["capture_y"],
        recommended_z_mm=measurement["proposed_z_mm"],
        confidence=measurement["confidence"],
        apply=True,
        human_approved=True,
    )
    try:
        response = await client.apply_soil_height(apply_request)
    except HomeAssistantError as exc:
        response = {"status": "conflict", "message": str(exc)}
    response_status = str(response.get("status") or "rejected")
    if response_status == "applied":
        status, action = "applied", "approve"
    elif response_status == "conflict":
        status, action = "conflict", "stale_conflict"
    else:
        status, action = "rejected", "rejected_write"
    reason = str(response.get("message") or response.get("status") or status)[:240]
    database.update_soil_measurement_status(measurement_id, status, reason)
    database.record_soil_decision(measurement_id, action, response)
    return response


@app.post("/soil/measurements/{measurement_id}/apply")
async def apply_soil_measurement(measurement_id: UUID) -> RedirectResponse:
    await _apply_soil_measurement(str(measurement_id))
    return RedirectResponse("../../../soil-height", status_code=303)


@app.post("/soil/apply-selected")
async def apply_selected_soil_measurements(
    measurement_ids: Annotated[list[str] | None, Form()] = None,
) -> RedirectResponse:
    for measurement_id in measurement_ids or []:
        try:
            UUID(measurement_id)
        except ValueError as exc:
            raise HTTPException(422, "Malformed soil measurement ID") from exc
        await _apply_soil_measurement(measurement_id)
    return RedirectResponse("../soil-height", status_code=303)


@app.post("/soil/measurements/{measurement_id}/reject")
async def reject_soil_measurement(measurement_id: UUID) -> RedirectResponse:
    measurement = database.soil_measurement(str(measurement_id))
    if measurement is None or measurement["status"] != "valid":
        raise HTTPException(404, "Reviewable soil result not found")
    database.update_soil_measurement_status(
        str(measurement_id), "rejected", "Rejected during human review"
    )
    database.record_soil_decision(str(measurement_id), "reject", {"status": "rejected"})
    return RedirectResponse("../../../soil-height", status_code=303)


@app.get("/canopy-settings", response_class=HTMLResponse)
async def canopy_settings_page(request: Request) -> HTMLResponse:
    values = canopy_fusion_settings_store.load()

    def checked(value: bool) -> str:
        return " checked" if value else ""

    body = f"""<section class=card><h2>Multi-image canopy fusion</h2>
<p>Plant segmentation still runs on each original image. When a plant reaches an image edge,
the resulting ownership masks are aligned in calibrated garden coordinates and fused before
its radius is measured. This avoids seams and duplicate leaves from an RGB panorama.</p>
<form method=post action="canopy-settings">
<fieldset><legend>Activation</legend>
<label><input type=checkbox name=enabled value=true{checked(values.enabled)}> Enable calibrated mask fusion</label><br>
<label><input type=checkbox name=always_fuse_when_available value=true{checked(values.always_fuse_when_available)}> Fuse whenever enough views are available</label><br>
<label>Fuse below visible fraction <input type=number name=activation_visible_fraction min=0 max=1 step=.01 value="{values.activation_visible_fraction:g}"></label><br>
<label>Minimum views <input type=number name=minimum_views min=2 max=20 step=1 value="{values.minimum_views}"></label><br>
<label>Maximum time gap (hours) <input type=number name=maximum_time_gap_hours min=.1 max=720 step=.1 value="{values.maximum_time_gap_hours:g}"></label>
</fieldset>
<fieldset><legend>Evidence acceptance</legend>
<label>Minimum per-view confidence <input type=number name=minimum_view_confidence min=0 max=1 step=.01 value="{values.minimum_view_confidence:g}"></label><br>
<label>Supporting views required per pixel <input type=number name=minimum_supporting_views min=1 max=10 step=1 value="{values.minimum_supporting_views}"></label><br>
<label>Single-view pixel confidence <input type=number name=single_view_acceptance_confidence min=0 max=1 step=.01 value="{values.single_view_acceptance_confidence:g}"></label><br>
<label>Source-edge evidence margin (mm) <input type=number name=source_edge_margin_mm min=0 max=250 step=1 value="{values.source_edge_margin_mm:g}"></label>
</fieldset>
<fieldset><legend>Radius measurement</legend>
<label>Outer radial percentile <input type=number name=radial_percentile min=80 max=100 step=.1 value="{values.radial_percentile:g}"></label><br>
<label>Angular sectors <input type=number name=angular_sectors min=12 max=360 step=1 value="{values.angular_sectors}"></label><br>
<label>Maximum fusion canvas (pixels) <input type=number name=maximum_canvas_pixels min=480 max=6000 step=10 value="{values.maximum_canvas_pixels}"></label>
</fieldset>
<fieldset><legend>Automatic-action guardrails</legend>
<label><input type=checkbox name=automatic_requires_reliable_fusion value=true{checked(values.automatic_requires_reliable_fusion)}> Require reliable fusion when partial views are present</label><br>
<label>Minimum angular coverage <input type=number name=minimum_angular_coverage min=0 max=1 step=.01 value="{values.minimum_angular_coverage:g}"></label><br>
<label>Minimum corroborated mask fraction <input type=number name=minimum_corroborated_fraction min=0 max=1 step=.01 value="{values.minimum_corroborated_fraction:g}"></label><br>
<label>Maximum disagreement with per-image estimate (mm) <input type=number name=maximum_automatic_disagreement_mm min=0 max=500 step=1 value="{values.maximum_automatic_disagreement_mm:g}"></label><br>
<label><input type=checkbox name=save_diagnostics value=true{checked(values.save_diagnostics)}> Save fusion diagnostics for review</label>
</fieldset>
<button>Save canopy fusion settings</button></form>
<p class=muted>Disabling a guardrail permits more automation but does not remove the normal
confidence, calibration, zone, or plant-safety checks.</p></section>"""
    return layout(request, body, "Canopy fusion")


@app.post("/canopy-settings")
async def save_canopy_settings(
    enabled: bool = Form(False),
    always_fuse_when_available: bool = Form(False),
    activation_visible_fraction: float = Form(0.92),
    minimum_views: int = Form(2),
    maximum_time_gap_hours: float = Form(6),
    minimum_view_confidence: float = Form(0.35),
    minimum_supporting_views: int = Form(2),
    single_view_acceptance_confidence: float = Form(0.82),
    source_edge_margin_mm: float = Form(20),
    radial_percentile: float = Form(97),
    angular_sectors: int = Form(72),
    minimum_angular_coverage: float = Form(0.70),
    minimum_corroborated_fraction: float = Form(0.05),
    maximum_automatic_disagreement_mm: float = Form(35),
    automatic_requires_reliable_fusion: bool = Form(False),
    maximum_canvas_pixels: int = Form(2400),
    save_diagnostics: bool = Form(False),
) -> RedirectResponse:
    try:
        values = CanopyFusionSettings(
            enabled=enabled,
            always_fuse_when_available=always_fuse_when_available,
            activation_visible_fraction=activation_visible_fraction,
            minimum_views=minimum_views,
            maximum_time_gap_hours=maximum_time_gap_hours,
            minimum_view_confidence=minimum_view_confidence,
            minimum_supporting_views=minimum_supporting_views,
            single_view_acceptance_confidence=single_view_acceptance_confidence,
            source_edge_margin_mm=source_edge_margin_mm,
            radial_percentile=radial_percentile,
            angular_sectors=angular_sectors,
            minimum_angular_coverage=minimum_angular_coverage,
            minimum_corroborated_fraction=minimum_corroborated_fraction,
            maximum_automatic_disagreement_mm=maximum_automatic_disagreement_mm,
            automatic_requires_reliable_fusion=automatic_requires_reliable_fusion,
            maximum_canvas_pixels=maximum_canvas_pixels,
            save_diagnostics=save_diagnostics,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if values.minimum_supporting_views > values.minimum_views:
        raise HTTPException(
            422, "Supporting views per pixel cannot exceed the minimum view count"
        )
    canopy_fusion_settings_store.save(values)
    return RedirectResponse("canopy-settings", status_code=303)


@app.get("/weed-settings", response_class=HTMLResponse)
async def weed_settings_page(request: Request) -> HTMLResponse:
    values = weed_settings_store.load()
    labels = database.weed_training_summary()
    weed_verifier.reload()
    model = weed_verifier.model

    def checked(value: bool) -> str:
        return " checked" if value else ""

    model_status = (
        f"Trained {escape(str(model['created_at']))} from {model['sample_count']} labels; "
        f"validation precision {model['metrics']['precision']:.1%}, "
        f"recall {model['metrics']['recall']:.1%}."
        if model
        else "No trained verifier model yet."
    )
    training_notice = request.query_params.get("training")
    training_notice_html = (
        f"<p class=warn>{escape(training_notice)}</p>" if training_notice else ""
    )
    body = f"""<section class=card><h2>Weed detection and automation</h2>
<p>Every stage is configurable. Start in review/shadow mode, label real examples, train the
local verifier, then enable enforcement or automatic FarmBot creation when its validation
results and field behaviour are satisfactory.</p>
<form method=post action="weed-settings">
<fieldset><legend>Operation</legend>
<label><input type=checkbox name=enabled value=true{checked(values.enabled)}> Enable weed detection</label><br>
<label><input type=checkbox name=automatic_creation value=true{checked(values.automatic_creation)}>
Automatically create detected weeds in FarmBot</label><br>
<label>Review/recommendation confidence <input type=number name=minimum_confidence min=0 max=1 step=.01 value="{values.minimum_confidence:g}"></label><br>
<label>Automatic creation confidence <input type=number name=automatic_creation_confidence min=0 max=1 step=.01 value="{values.automatic_creation_confidence:g}"></label><br>
<label>Created weed radius (mm) <input type=number name=weed_radius_mm min=1 step=1 value="{values.weed_radius_mm:g}"></label>
</fieldset>
<fieldset><legend>Candidate size, colour and shape</legend>
<label>Minimum weed area (mm²) <input type=number name=minimum_area_mm2 min=5 step=1 value="{values.minimum_area_mm2:g}"></label><br>
<label>Maximum weed area (mm²) <input type=number name=maximum_area_mm2 min=10 step=1 value="{values.maximum_area_mm2:g}"></label><br>
<label><input type=checkbox name=shape_filter_enabled value=true{checked(values.shape_filter_enabled)}> Enable colour/shape filter</label><br>
<label>Strong-green hue range <input type=number name=green_hue_min min=0 max=179 step=1 value="{values.green_hue_min}"> to
<input type=number name=green_hue_max min=0 max=179 step=1 value="{values.green_hue_max}"></label><br>
<label>Strong-green minimum saturation <input type=number name=strong_green_minimum_saturation min=0 max=255 step=1 value="{values.strong_green_minimum_saturation}"></label><br>
<label>Strong-green minimum Excess Green <input type=number name=strong_green_minimum_excess_green min=-255 max=510 step=1 value="{values.strong_green_minimum_excess_green}"></label><br>
<label>Minimum strong-green fraction <input type=number name=minimum_green_purity min=0 max=1 step=.01 value="{values.minimum_green_purity:g}"></label><br>
<label>Minimum solidity <input type=number name=minimum_solidity min=0 max=1 step=.01 value="{values.minimum_solidity:g}"></label><br>
<label>Minimum circularity <input type=number name=minimum_circularity min=0 max=1 step=.01 value="{values.minimum_circularity:g}"></label><br>
<label>Maximum aspect ratio <input type=number name=maximum_aspect_ratio min=1 max=50 step=.1 value="{values.maximum_aspect_ratio:g}"></label>
</fieldset>
<fieldset><legend>Known crop protection</legend>
<label><input type=checkbox name=crop_protection_enabled value=true{checked(values.crop_protection_enabled)}> Protect all known and previously observed crops</label><br>
<label>Canopy radius multiplier <input type=number name=crop_support_radius_multiplier min=.5 max=5 step=.05 value="{values.crop_support_radius_multiplier:g}"></label><br>
<label>Minimum extra canopy support (mm) <input type=number name=crop_support_extra_mm min=0 max=500 step=1 value="{values.crop_support_extra_mm:g}"></label><br>
<label>Extra exclusion around plants (mm) <input type=number name=plant_exclusion_margin_mm min=0 step=1 value="{values.plant_exclusion_margin_mm:g}"></label>
</fieldset>
<fieldset><legend>Multi-image confirmation</legend>
<label><input type=checkbox name=temporal_confirmation_enabled value=true{checked(values.temporal_confirmation_enabled)}> Enable temporal confirmation</label><br>
<label>Looks before recommendation <input type=number name=recommendation_min_observations min=1 max=20 step=1 value="{values.recommendation_min_observations}"></label><br>
<label>Looks before automatic creation <input type=number name=automatic_min_observations min=1 max=20 step=1 value="{values.automatic_min_observations}"></label><br>
<label>Position matching distance (mm) <input type=number name=temporal_match_distance_mm min=1 max=250 step=1 value="{values.temporal_match_distance_mm:g}"></label><br>
<label>Maximum gap between looks (hours) <input type=number name=temporal_max_gap_hours min=1 max=8760 step=1 value="{values.temporal_max_gap_hours}"></label>
</fieldset>
<fieldset><legend>Learned visual verifier</legend>
<label><input type=checkbox name=visual_verifier_enabled value=true{checked(values.visual_verifier_enabled)}> Enable learned verifier</label><br>
<label><input type=checkbox name=visual_verifier_shadow_mode value=true{checked(values.visual_verifier_shadow_mode)}> Shadow mode (score but do not reject)</label><br>
<label><input type=checkbox name=visual_verifier_required_for_automatic value=true{checked(values.visual_verifier_required_for_automatic)}> Require verifier approval for automatic creation</label><br>
<label>Verifier confidence threshold <input type=number name=visual_verifier_minimum_confidence min=0 max=1 step=.01 value="{values.visual_verifier_minimum_confidence:g}"></label><br>
<label>Verifier weight in final score <input type=number name=visual_verifier_weight min=0 max=1 step=.05 value="{values.visual_verifier_weight:g}"></label><br>
<label>Minimum weed and non-weed labels for training <input type=number name=training_minimum_per_class min=2 step=1 value="{values.training_minimum_per_class}"></label><br>
<label><input type=checkbox name=automatic_retraining value=true{checked(values.automatic_retraining)}> Retrain automatically after each new label once enough labels exist</label><br>
<label><input type=checkbox name=candidate_crop_storage_enabled value=true{checked(values.candidate_crop_storage_enabled)}> Store candidate crops for review/training</label>
</fieldset>
<fieldset><legend>Existing weed maintenance</legend>
<label><input type=checkbox name=automatic_radius_adjustment value=true{checked(values.automatic_radius_adjustment)}>
Automatically increase the radius of a matching known weed</label><br>
<label>Radius adjustment confidence <input type=number name=radius_adjustment_confidence min=0 max=1 step=.01 value="{values.radius_adjustment_confidence:g}"></label><br>
<label><input type=checkbox name=automatic_removal value=true{checked(values.automatic_removal)}>
Automatically remove known weeds that disappear</label><br>
<label>Removal confidence <input type=number name=removal_confidence min=0 max=1 step=.01 value="{values.removal_confidence:g}"></label><br>
<label>Absent images before removal <input type=number name=removal_min_consecutive_absent min=1 max=10 step=1 value="{values.removal_min_consecutive_absent}"></label>
</fieldset>
<button>Save all weed settings</button></form></section>
<section class=card><h2>Verifier training</h2>{training_notice_html}<p>{model_status}</p>
<p>Labels: {labels['weed']} weeds · {labels['crop']} crops · {labels['mulch_soil']} mulch/soil ·
{labels['fungus_moss']} fungus/moss · {labels['hardware_other']} hardware/other.</p>
<form method=post action="weed-model/train"><button>Train verifier now</button></form>
<p class=muted>Accepting a weed records a positive label. Rejection and the category buttons on
the Analysis page record hard negative examples from this FarmBot.</p></section>"""
    return layout(request, body, "Weed settings")


@app.post("/weed-settings")
async def save_weed_settings(
    enabled: bool = Form(False),
    automatic_creation: bool = Form(False),
    automatic_radius_adjustment: bool = Form(False),
    radius_adjustment_confidence: float = Form(0.55),
    automatic_removal: bool = Form(False),
    removal_confidence: float = Form(0.6),
    removal_min_consecutive_absent: int = Form(1),
    minimum_area_mm2: float = Form(75),
    maximum_area_mm2: float = Form(2500),
    plant_exclusion_margin_mm: float = Form(35),
    crop_protection_enabled: bool = Form(False),
    crop_support_radius_multiplier: float = Form(1.2),
    crop_support_extra_mm: float = Form(25),
    shape_filter_enabled: bool = Form(False),
    green_hue_min: int = Form(25),
    green_hue_max: int = Form(100),
    strong_green_minimum_saturation: int = Form(45),
    strong_green_minimum_excess_green: int = Form(20),
    minimum_green_purity: float = Form(0.45),
    minimum_solidity: float = Form(0.25),
    minimum_circularity: float = Form(0.03),
    maximum_aspect_ratio: float = Form(7),
    minimum_confidence: float = Form(0.70),
    automatic_creation_confidence: float = Form(0.90),
    temporal_confirmation_enabled: bool = Form(False),
    recommendation_min_observations: int = Form(1),
    automatic_min_observations: int = Form(3),
    temporal_match_distance_mm: float = Form(25),
    temporal_max_gap_hours: int = Form(168),
    visual_verifier_enabled: bool = Form(False),
    visual_verifier_shadow_mode: bool = Form(False),
    visual_verifier_required_for_automatic: bool = Form(False),
    visual_verifier_minimum_confidence: float = Form(0.85),
    visual_verifier_weight: float = Form(0.7),
    training_minimum_per_class: int = Form(10),
    automatic_retraining: bool = Form(False),
    candidate_crop_storage_enabled: bool = Form(False),
    weed_radius_mm: float = Form(15),
) -> RedirectResponse:
    try:
        values = WeedSettings(
            enabled=enabled,
            automatic_creation=automatic_creation,
            automatic_radius_adjustment=automatic_radius_adjustment,
            radius_adjustment_confidence=radius_adjustment_confidence,
            automatic_removal=automatic_removal,
            removal_confidence=removal_confidence,
            removal_min_consecutive_absent=removal_min_consecutive_absent,
            minimum_area_mm2=minimum_area_mm2,
            maximum_area_mm2=maximum_area_mm2,
            plant_exclusion_margin_mm=plant_exclusion_margin_mm,
            crop_protection_enabled=crop_protection_enabled,
            crop_support_radius_multiplier=crop_support_radius_multiplier,
            crop_support_extra_mm=crop_support_extra_mm,
            shape_filter_enabled=shape_filter_enabled,
            green_hue_min=green_hue_min,
            green_hue_max=green_hue_max,
            strong_green_minimum_saturation=strong_green_minimum_saturation,
            strong_green_minimum_excess_green=strong_green_minimum_excess_green,
            minimum_green_purity=minimum_green_purity,
            minimum_solidity=minimum_solidity,
            minimum_circularity=minimum_circularity,
            maximum_aspect_ratio=maximum_aspect_ratio,
            minimum_confidence=minimum_confidence,
            automatic_creation_confidence=automatic_creation_confidence,
            temporal_confirmation_enabled=temporal_confirmation_enabled,
            recommendation_min_observations=recommendation_min_observations,
            automatic_min_observations=automatic_min_observations,
            temporal_match_distance_mm=temporal_match_distance_mm,
            temporal_max_gap_hours=temporal_max_gap_hours,
            visual_verifier_enabled=visual_verifier_enabled,
            visual_verifier_shadow_mode=visual_verifier_shadow_mode,
            visual_verifier_required_for_automatic=visual_verifier_required_for_automatic,
            visual_verifier_minimum_confidence=visual_verifier_minimum_confidence,
            visual_verifier_weight=visual_verifier_weight,
            training_minimum_per_class=training_minimum_per_class,
            automatic_retraining=automatic_retraining,
            candidate_crop_storage_enabled=candidate_crop_storage_enabled,
            weed_radius_mm=weed_radius_mm,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if values.minimum_area_mm2 > values.maximum_area_mm2:
        raise HTTPException(422, "Minimum weed area cannot exceed maximum weed area")
    if values.green_hue_min > values.green_hue_max:
        raise HTTPException(422, "Minimum green hue cannot exceed maximum green hue")
    if values.recommendation_min_observations > values.automatic_min_observations:
        raise HTTPException(
            422, "Looks before recommendation cannot exceed looks before automatic creation"
        )
    weed_settings_store.save(values)
    return RedirectResponse("weed-settings", status_code=303)


async def _train_weed_verifier() -> dict:
    values = weed_settings_store.load()
    model = await asyncio.to_thread(
        weed_verifier.train,
        database.weed_training_samples(),
        values.training_minimum_per_class,
    )
    database.record_weed_model_run(model)
    return model


async def _record_weed_label(detection_id: UUID, label: str) -> None:
    if label not in ALL_LABELS:
        raise HTTPException(422, "Unsupported training label")
    if not database.label_weed_detection(str(detection_id), label):
        raise HTTPException(404, "Weed detection not found")
    if weed_settings_store.load().automatic_retraining:
        try:
            await _train_weed_verifier()
        except ValueError:
            # Label collection intentionally starts before the minimum dataset
            # exists. The settings page shows the live counts.
            pass


@app.post("/weed-model/train")
async def train_weed_model() -> RedirectResponse:
    try:
        model = await _train_weed_verifier()
    except ValueError as exc:
        return RedirectResponse(
            f"../weed-settings?training={quote(str(exc))}", status_code=303
        )
    message = f"Trained from {model['sample_count']} labels"
    return RedirectResponse(
        f"../weed-settings?training={quote(message)}",
        status_code=303,
    )


@app.post("/weeds/{detection_id}/approve")
async def approve_weed(detection_id: UUID) -> JSONResponse:
    detection = database.weed_detection(str(detection_id))
    if detection is None or detection["status"] not in ("recommended", "observing"):
        raise HTTPException(404, "Weed recommendation not found")
    verdict = zone_verdict(ZoneAspect.WEEDS, detection["x"], detection["y"])
    if not verdict.allowed:
        # The recommendation stays pending so the zones can be corrected instead
        # of losing the detection.
        return JSONResponse(
            {
                "status": "conflict",
                "message": f"Weeds are not allowed at this position: {verdict.reason}",
            },
            status_code=409,
        )
    result = await client.create_weed(
        CreateWeedRequest(
            config_entry_id=detection["config_entry_id"],
            detection_id=detection_id,
            x=detection["x"],
            y=detection["y"],
            z=detection["z"],
            radius=detection["radius_mm"],
            confidence=detection["confidence"],
            apply=True,
            human_approved=True,
        )
    )
    if result.get("status") == "applied":
        database.update_weed_detection(str(detection_id), "created")
        await _record_weed_label(detection_id, "weed")
    return JSONResponse(result)


@app.post("/weeds/{detection_id}/reject")
async def reject_weed(detection_id: UUID) -> JSONResponse:
    detection = database.weed_detection(str(detection_id))
    if detection is None:
        raise HTTPException(404, "Weed recommendation not found")
    database.reject_weed_detection(
        str(detection_id), max(20.0, float(detection["radius_mm"]) * 1.5)
    )
    await _record_weed_label(detection_id, "mulch_soil")
    return JSONResponse({"status": "rejected", "message": "Weed recommendation rejected"})


@app.post("/weeds/{detection_id}/label/{label}")
async def label_weed(detection_id: UUID, label: str) -> JSONResponse:
    detection = database.weed_detection(str(detection_id))
    if detection is None:
        raise HTTPException(404, "Weed detection not found")
    await _record_weed_label(detection_id, label)
    if label == "weed":
        database.update_weed_detection(str(detection_id), "labelled")
    else:
        database.reject_weed_detection(
            str(detection_id), max(20.0, float(detection["radius_mm"]) * 1.5)
        )
    return JSONResponse(
        {"status": "applied", "message": f"Saved {label.replace('_', '/')} training label"}
    )


def zone_verdict(aspect: ZoneAspect, x: float, y: float, radius_mm: float = 0.0) -> ZoneVerdict:
    """Evaluate a placement against the persisted zones (empty config allows)."""
    return evaluate(zone_store.zones(), aspect, x, y, radius_mm)


def _parse_polygon_points(raw: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line in raw.replace(";", "\n").splitlines():
        entry = line.strip()
        if not entry:
            continue
        parts = [part for part in entry.replace("\t", ",").split(",") if part.strip()]
        if len(parts) != 2:
            raise HTTPException(422, f"Could not read the polygon point '{entry}'; use 'x, y'")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise HTTPException(422, f"Polygon point '{entry}' is not numeric") from exc
    return points


def _zone_json(zones: list[Zone]) -> str:
    payload = [
        {
            "zone_id": zone.zone_id,
            "name": zone.name,
            "kind": zone.kind.value,
            "shape": zone.shape.value,
            "enabled": zone.enabled,
            "min_x": zone.min_x,
            "min_y": zone.min_y,
            "max_x": zone.max_x,
            "max_y": zone.max_y,
            "center_x": zone.center_x,
            "center_y": zone.center_y,
            "radius_mm": zone.radius_mm,
            "points": [list(point) for point in zone.points],
        }
        for zone in zones
    ]
    return escape(json.dumps(payload, separators=(",", ":")), quote=True)


@app.get("/zones", response_class=HTMLResponse)
async def zones_page(request: Request) -> HTMLResponse:
    zones = zone_store.zones()

    def flag(zone: Zone, field: str, label: str) -> str:
        state = " checked" if getattr(zone, field) else ""
        return (
            f'<label><input type=checkbox form="zone-form-{zone.zone_id}" name={field} '
            f"value=true{state}> {escape(label)}</label>"
        )

    zone_forms = "".join(
        f'<form id="zone-form-{zone.zone_id}" method=post action="zones/{zone.zone_id}/update"></form>'
        f'<form id="zone-delete-{zone.zone_id}" method=post action="zones/{zone.zone_id}/delete"></form>'
        for zone in zones
    )
    zone_rows = "".join(
        f"<tr><td>{escape(zone.name)}</td>"
        f"<td>{'Boundary' if zone.kind is ZoneKind.BOUNDARY else 'Exclusion zone'}</td>"
        f"<td>{escape(zone.describe_geometry())}</td>"
        f"<td>{flag(zone, 'allow_weeds', 'Weeds allowed')}</td>"
        f"<td>{flag(zone, 'allow_plant_centers', 'Centres allowed')}</td>"
        f"<td>{flag(zone, 'allow_plant_radius', 'Radius allowed')}</td>"
        f"<td>{flag(zone, 'enabled', 'Active')}</td>"
        f'<td><div class=button-row><button form="zone-form-{zone.zone_id}">Save</button>'
        f'<button form="zone-delete-{zone.zone_id}">Delete</button></div></td></tr>'
        for zone in zones
    )
    body = f"""<section class=card><h2>Boundaries and exclusion zones</h2>
<p>Zones are areas of the garden in FarmBot coordinates (millimetres). A
<b>boundary</b> encloses where things are allowed; an <b>exclusion zone</b> marks
an area to keep clear. For each zone you choose independently whether weeds may
be placed there, whether a plant centre may be moved there, and whether a plant's
protection radius may extend into it.</p>
<p class=muted>Overlaps resolve in a fixed order: an exclusion zone that allows an
aspect is an explicit exception and wins; otherwise any zone that forbids the
aspect and is touched by the position denies it; otherwise, if at least one
boundary allows that aspect, the position must fall inside one of them. With no
zones configured nothing is restricted. Weeds, plant centres, and a boundary's
test of a protection radius are all point tests: a radius may extend past a
boundary's edge, since only the plant itself has to stay inside the growing
area. Exclusion zones are different -- they mark real hazards, so the full
protection disc must not overlap a forbidding exclusion zone.</p>
<p class=warn>Zones gate both automatic writes and manual approvals: a blocked
weed is never created, a blocked centre move and a blocked radius increase are
refused with the zone's name.</p></section>
<section class=card><h2>Add a zone</h2>
<form method=post action="zones">
<div class=grid>
<div>
<label>Name<br><input name=name maxlength=60 required placeholder="Bed 1"></label><br>
<label>Type<br><select id=kind name=kind>
<option value=boundary selected>Boundary — things may go inside</option>
<option value=exclusion>Exclusion zone — keep this area clear</option></select></label><br>
<label>Shape<br><select id=shape name=shape>
<option value=rectangle selected>Rectangle</option>
<option value=circle>Circle</option>
<option value=polygon>Polygon</option></select></label>
</div>
<div>
<div id=fields-rectangle>
<label>Corner 1 X (mm)<br><input type=number step=any name=min_x value=0></label><br>
<label>Corner 1 Y (mm)<br><input type=number step=any name=min_y value=0></label><br>
<label>Corner 2 X (mm)<br><input type=number step=any name=max_x value=1000></label><br>
<label>Corner 2 Y (mm)<br><input type=number step=any name=max_y value=1000></label>
</div>
<div id=fields-circle hidden>
<label>Centre X (mm)<br><input type=number step=any name=center_x value=0></label><br>
<label>Centre Y (mm)<br><input type=number step=any name=center_y value=0></label><br>
<label>Radius (mm)<br><input type=number step=any min=1 name=radius_mm value=500></label>
</div>
<div id=fields-polygon hidden>
<label>Points, one "X, Y" pair per line (at least three)<br>
<textarea name=points rows=6 cols=24 placeholder="0, 0&#10;1200, 0&#10;1200, 800"></textarea></label>
</div>
</div>
<div>
<p><b>Inside this zone</b></p>
<label><input type=checkbox id=new_allow_weeds name=allow_weeds value=true checked>
Weeds may be placed</label><br>
<label><input type=checkbox id=new_allow_plant_centers name=allow_plant_centers value=true checked>
Plant centres may be moved here</label><br>
<label><input type=checkbox id=new_allow_plant_radius name=allow_plant_radius value=true checked>
A plant radius may extend into it</label>
<p class=muted>Clearing a box on a boundary carves a hole in it; ticking one on an
exclusion zone makes that aspect an allowed exception inside it.</p>
<p><button>Add zone</button></p>
</div>
</div>
</form></section>
<section class=card><h2>Configured zones</h2>{zone_forms}
<table><thead><tr><th>Name</th><th>Type</th><th>Area</th><th>Weeds</th><th>Plant centres</th>
<th>Plant radius</th><th>Active</th><th>Actions</th></tr></thead>
<tbody>{zone_rows or "<tr><td colspan=8>No zones yet; nothing is restricted</td></tr>"}</tbody></table>
<p class=muted>Changing a tick box takes effect after Save.</p></section>
<section class=card><h2>Garden map</h2>
<canvas id=zone-map width=900 height=600 data-zones="{_zone_json(zones)}"
 data-entry="{escape(settings.selected_config_entry_id or "", quote=True)}"
 style="width:100%;border:1px solid #ccc"></canvas>
<div class=button-row><button id=zone-load-items type=button>Show plants &amp; FarmBot weeds</button></div>
<p id=zone-map-status class=muted></p>
<p class=legend>Green outline = boundary, red outline = exclusion zone, dashed = inactive.
Green circles = plants with their protection radius, red dots = FarmBot weeds.</p>
</section>
<script>{_ZONES_JS}</script>"""
    return layout(request, body, "Boundaries and zones")


@app.get("/api/zones")
async def zones_api() -> JSONResponse:
    return JSONResponse(zone_store.load().model_dump(mode="json"))


@app.post("/zones")
async def create_zone(
    name: str = Form(...),
    kind: str = Form("boundary"),
    shape: str = Form("rectangle"),
    allow_weeds: bool = Form(False),
    allow_plant_centers: bool = Form(False),
    allow_plant_radius: bool = Form(False),
    min_x: float = Form(0),
    min_y: float = Form(0),
    max_x: float = Form(0),
    max_y: float = Form(0),
    center_x: float = Form(0),
    center_y: float = Form(0),
    radius_mm: float = Form(0),
    points: str = Form(""),
) -> RedirectResponse:
    try:
        zone_kind, zone_shape = ZoneKind(kind), ZoneShape(shape)
    except ValueError as exc:
        raise HTTPException(400, "Unknown zone type or shape") from exc
    try:
        zone = Zone(
            name=name.strip(),
            kind=zone_kind,
            shape=zone_shape,
            allow_weeds=allow_weeds,
            allow_plant_centers=allow_plant_centers,
            allow_plant_radius=allow_plant_radius,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            center_x=center_x,
            center_y=center_y,
            radius_mm=radius_mm,
            points=_parse_polygon_points(points) if zone_shape is ZoneShape.POLYGON else [],
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    zone_store.add(zone)
    LOGGER.info("Added %s zone '%s' (%s)", zone.kind.value, zone.name, zone.describe_geometry())
    return RedirectResponse("zones", status_code=303)


@app.post("/zones/{zone_id}/update")
async def update_zone(
    zone_id: str,
    allow_weeds: bool = Form(False),
    allow_plant_centers: bool = Form(False),
    allow_plant_radius: bool = Form(False),
    enabled: bool = Form(False),
) -> RedirectResponse:
    updated = zone_store.update(
        zone_id,
        allow_weeds=allow_weeds,
        allow_plant_centers=allow_plant_centers,
        allow_plant_radius=allow_plant_radius,
        enabled=enabled,
    )
    if updated is None:
        raise HTTPException(404, "Zone not found")
    return RedirectResponse("../../zones", status_code=303)


@app.post("/zones/{zone_id}/delete")
async def delete_zone(zone_id: str) -> RedirectResponse:
    if not zone_store.delete(zone_id):
        raise HTTPException(404, "Zone not found")
    return RedirectResponse("../../zones", status_code=303)


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


def _center_move_blocked(row: dict) -> str | None:
    """Refusal message when zones forbid a plant centre at the suggested point."""
    verdict = zone_verdict(
        ZoneAspect.CENTERS, row["recommended_center_x"], row["recommended_center_y"]
    )
    if verdict.allowed:
        return None
    return f"A plant centre cannot be moved there: {verdict.reason}"


def _radius_growth_blocked(row: dict) -> str | None:
    """Refusal message when zones forbid the recommended protection radius.

    Measurements recorded before the plant position was stored cannot be
    checked; those keep their previous behaviour rather than being blocked.
    """
    center_x, center_y = row.get("recorded_center_x"), row.get("recorded_center_y")
    if center_x is None or center_y is None:
        return None
    verdict = zone_verdict(
        ZoneAspect.RADIUS,
        float(center_x),
        float(center_y),
        float(row["recommended_protection_radius_mm"]),
    )
    if verdict.allowed:
        return None
    return f"The recommended radius is not allowed to extend there: {verdict.reason}"


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
    if action not in {"approve", "reject", "move-center"}:
        raise HTTPException(400)
    row = database.measurement(measurement_id)
    if row is None:
        raise HTTPException(404)
    if database.has_terminal_decision(measurement_id):
        return _action_response(
            request, "conflict", "This recommendation was already reviewed", error_status=409
        )
    if action == "move-center":
        if not row.get("center_misaligned") or row.get("recommended_center_x") is None:
            return _action_response(
                request, "conflict", "No centre correction is available", error_status=409
            )
        blocked = _center_move_blocked(row)
        if blocked is not None:
            return _action_response(request, "conflict", blocked, error_status=409)
        entry_id = row.get("config_entry_id") or settings.selected_config_entry_id
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=entry_id,
                image_lookback_hours=settings.image_lookback_hours,
            )
        )
        plant = next((p for p in inventory.plants if p.id == row["plant_id"]), None)
        if plant is None:
            return _action_response(
                request, "conflict", "Plant is no longer active", error_status=409
            )
        result = await client.apply_plant_center(
            ApplyPlantCenterRequest(
                config_entry_id=entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_x=plant.x,
                expected_y=plant.y,
                recommended_x=row["recommended_center_x"],
                recommended_y=row["recommended_center_y"],
                apply=True,
                human_approved=True,
            )
        )
        status = str(result.get("status", "error"))
        if status == "applied":
            database.record_decision(measurement_id, "center_moved", result)
            return _action_response(
                request,
                "updated",
                "Plant centre moved; you can still apply or reject the radius",
            )
        return _action_response(
            request,
            status,
            str(result.get("message") or status),
            error_status=409 if status == "conflict" else None,
        )
    if action == "approve":
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if not row.get("calibrated", 1):
            return _action_response(
                request, "conflict", "Calibration is required", error_status=409
            )
        if row["recommended_protection_radius_mm"] == row["current_radius_mm"]:
            database.record_group_decision(
                measurement_id,
                "approved_no_change",
                {
                    "current_radius_mm": row["current_radius_mm"],
                    "observed_radius_mm": row["recommended_protection_radius_mm"],
                },
            )
            return _action_response(
                request, "applied", "Observation approved; no radius change was needed"
            )
        blocked = _radius_growth_blocked(row)
        if blocked is not None:
            return _action_response(request, "conflict", blocked, error_status=409)
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
            return _action_response(
                request,
                "conflict",
                "The plant radius changed; inventory refreshed",
                error_status=409,
            )
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
        database.record_group_decision(measurement_id, "applied", result)
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
                curve_message = str(curve_result.get("message") or curve_result.get("status", ""))
            except HomeAssistantError as exc:
                LOGGER.warning("Radius applied but curve inventory/update failed: %s", exc)
                curve_message = "deferred because inventory was unavailable"
        return _action_response(
            request,
            "applied",
            f"Radius applied. Curve update: {curve_message}",
        )
    database.record_group_decision(measurement_id, "reject", {})
    return _action_response(request, "rejected", "Recommendation rejected")


@app.post("/removals/{measurement_id}/{action}")
async def removal(request: Request, measurement_id: str, action: str) -> Response:
    if action not in {"approve", "keep", "move-center"}:
        raise HTTPException(400)
    row = database.measurement(measurement_id)
    if row is None:
        raise HTTPException(404)
    if database.has_terminal_decision(measurement_id):
        return _action_response(
            request, "conflict", "This removal was already reviewed", error_status=409
        )
    entry_id = row.get("config_entry_id") or settings.selected_config_entry_id
    if not database.is_latest_plant_measurement(entry_id, row["plant_id"], measurement_id):
        return _action_response(
            request,
            "conflict",
            "A newer canopy observation exists; removal was not applied",
            error_status=409,
        )
    if action == "keep":
        database.record_group_decision(measurement_id, "keep", {})
        return _action_response(request, "rejected", "Plant kept")
    if action == "move-center":
        if not row.get("center_misaligned") or row.get("recommended_center_x") is None:
            return _action_response(
                request, "conflict", "No centre correction is available", error_status=409
            )
        blocked = _center_move_blocked(row)
        if blocked is not None:
            return _action_response(request, "conflict", blocked, error_status=409)
        inventory = await client.inventory(
            InventoryRequest(
                config_entry_id=entry_id,
                image_lookback_hours=settings.image_lookback_hours,
            )
        )
        plant = next((p for p in inventory.plants if p.id == row["plant_id"]), None)
        if plant is None:
            return _action_response(
                request, "conflict", "Plant is no longer active", error_status=409
            )
        result = await client.apply_plant_center(
            ApplyPlantCenterRequest(
                config_entry_id=entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_x=plant.x,
                expected_y=plant.y,
                recommended_x=row["recommended_center_x"],
                recommended_y=row["recommended_center_y"],
                apply=True,
                human_approved=True,
            )
        )
        status = str(result.get("status", "error"))
        if status == "applied":
            database.record_group_decision(measurement_id, "keep", result)
        return _action_response(
            request,
            status,
            str(result.get("message") or status),
            error_status=409 if status != "applied" else None,
        )
    # Radius is not part of the removal itself, but the companion service uses
    # it as an optimistic-concurrency token. Refresh it immediately before the
    # request so an older vision measurement does not make every otherwise
    # valid, explicitly approved removal fail as stale.
    inventory = await client.inventory(
        InventoryRequest(
            config_entry_id=entry_id,
            image_lookback_hours=settings.image_lookback_hours,
        )
    )
    plant = next((p for p in inventory.plants if p.id == row["plant_id"]), None)
    if plant is None:
        return _action_response(request, "conflict", "Plant is no longer active", error_status=409)
    try:
        result = await client.apply_removal(
            ApplyRemovalRequest(
                config_entry_id=entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_current_radius_mm=plant.radius,
                confidence=row["confidence"],
                apply=True,
                human_approved=True,
            )
        )
    except StaleRadiusError:
        return _action_response(
            request, "conflict", "The plant changed; removal was not applied", error_status=409
        )
    status = str(result.get("status", "error"))
    message = str(result.get("message") or status)
    if status != "applied":
        return _action_response(request, status, message, error_status=409)
    database.update_measurement_outcome(measurement_id, decision="removed", applied=True)
    database.record_group_decision(measurement_id, "removed", result)
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
        return _action_response(
            request, "conflict", "This proposal was already reviewed", error_status=409
        )
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
    assigned = (
        None
        if plant is None
        else next((item for item in inventory.curves if item.id == plant.spread_curve_id), None)
    )
    if plant is None or assigned is None or assigned.data != previous:
        return _action_response(
            request,
            "conflict",
            "The plant's assigned curve changed after this proposal was created",
            error_status=409,
        )
    curve_data = {
        control_day: float(round(diameter)) for control_day, diameter in edit.data.items()
    }
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
