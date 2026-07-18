from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

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
from .curves import fit_monotonic_curve
from .database import Database
from .home_assistant import HomeAssistantClient, HomeAssistantError
from .jobs import JobManager
from .models import (
    ApplyRadiusRequest,
    Calibration,
    InventoryRequest,
    OperatingMode,
    VisionImageRequest,
)
from .settings import Settings
from .vision import manual_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
settings = Settings.load()
database = Database(settings.data_dir / "farmbot_vision.db")
client = HomeAssistantClient()
jobs = JobManager(settings, database, client)


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


# Lightweight vanilla-JS point selection and plant-centre overlay for the
# calibration page. No frontend build toolchain is used (Part 5).
_CALIBRATION_JS = r"""
(function(){
  const canvas=document.getElementById('canvas');
  const ctx=canvas.getContext('2d');
  const sel=document.getElementById('image');
  const status=document.getElementById('status');
  const ppmEl=document.getElementById('ppm');
  let img=null, plants=[], meta={x:0,y:0}, A=null, B=null;
  const W=canvas.width, H=canvas.height;
  function entry(){return document.getElementById('entry_id').value.trim();}
  function num(id){return parseFloat(document.getElementById(id).value)||0;}
  function redraw(){
    ctx.clearRect(0,0,W,H);
    if(img) ctx.drawImage(img,0,0,W,H);
    if(A){ctx.fillStyle='#2ecc40';ctx.beginPath();ctx.arc(A[0],A[1],5,0,7);ctx.fill();}
    if(B){ctx.fillStyle='#0074d9';ctx.beginPath();ctx.arc(B[0],B[1],5,0,7);ctx.fill();}
    if(A&&B){ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.stroke();}
  }
  function updatePpm(){
    const d=num('distance');
    if(A&&B&&d>0){
      const px=Math.hypot(B[0]-A[0],B[1]-A[1]);
      const ppm=px/d;
      ppmEl.textContent='Pixels per millimetre: '+ppm.toFixed(4);
      document.getElementById('save').disabled=!document.getElementById('confirm').checked;
      return ppm;
    }
    ppmEl.textContent='Pixels per millimetre: —';
    return 0;
  }
  function gardenToPixel(px,py,ppm,rot,offx,offy){
    let dx=px-meta.x+offx, dy=py-meta.y+offy;
    let t=rot*Math.PI/180;
    let rx=dx*Math.cos(t)-dy*Math.sin(t), ry=dx*Math.sin(t)+dy*Math.cos(t);
    return [W/2+rx*ppm, H/2+ry*ppm];
  }
  canvas.addEventListener('click',function(e){
    const rect=canvas.getBoundingClientRect();
    const x=(e.clientX-rect.left)*(W/rect.width);
    const y=(e.clientY-rect.top)*(H/rect.height);
    if(!A||B){A=[x,y];B=null;}else{B=[x,y];}
    redraw();updatePpm();
  });
  document.getElementById('distance').addEventListener('input',updatePpm);
  document.getElementById('confirm').addEventListener('change',updatePpm);
  document.getElementById('load').addEventListener('click',async function(){
    status.textContent='Loading…';
    try{
      const r=await fetch('api/vision/images?entry_id='+encodeURIComponent(entry()));
      if(!r.ok) throw new Error('HTTP '+r.status);
      const data=await r.json();
      plants=data.plants||[];
      sel.innerHTML='';
      (data.images||[]).forEach(function(im){
        const o=document.createElement('option');
        o.value=im.id;o.dataset.x=im.x;o.dataset.y=im.y;
        o.textContent='#'+im.id+' '+im.created_at;
        sel.appendChild(o);
      });
      status.textContent=(data.images||[]).length+' images, '+plants.length+' plants';
      if(sel.options.length) sel.dispatchEvent(new Event('change'));
    }catch(err){status.textContent='Could not load images: '+err.message;}
  });
  sel.addEventListener('change',function(){
    const o=sel.options[sel.selectedIndex];
    if(!o) return;
    meta={x:parseFloat(o.dataset.x)||0,y:parseFloat(o.dataset.y)||0};
    A=null;B=null;
    img=new Image();
    img.onload=redraw;
    img.onerror=function(){status.textContent='Could not load image bytes';};
    img.src='api/vision/image/'+o.value+'.jpg?entry_id='+encodeURIComponent(entry());
  });
  document.getElementById('overlay').addEventListener('click',function(){
    const ppm=updatePpm();
    if(!ppm){status.textContent='Set points and separation first';return;}
    redraw();
    ctx.strokeStyle='#ff4136';ctx.fillStyle='#ff4136';ctx.font='12px sans-serif';
    plants.forEach(function(p){
      const c=gardenToPixel(p.x,p.y,ppm,num('rotation'),num('offx'),num('offy'));
      ctx.beginPath();ctx.arc(c[0],c[1],Math.max(4,(p.radius||0)*ppm),0,7);ctx.stroke();
      ctx.fillText(p.id,c[0]+4,c[1]-4);
    });
    status.textContent='Overlaid '+plants.length+' plant centres — confirm alignment.';
  });
  document.getElementById('save').addEventListener('click',function(){
    const ppm=updatePpm();
    if(!ppm||!A||!B){status.textContent='Set both points and a separation';return;}
    const o=sel.options[sel.selectedIndex];
    const f=document.createElement('form');f.method='post';f.action='calibration';
    const fields={entry_id:entry(),ax:A[0],ay:A[1],bx:B[0],by:B[1],
      distance_mm:num('distance'),rotation:num('rotation'),offset_x:num('offx'),
      offset_y:num('offy'),image_id:o?o.value:''};
    for(const k in fields){const i=document.createElement('input');i.type='hidden';
      i.name=k;i.value=fields[k];f.appendChild(i);}
    document.body.appendChild(f);f.submit();
  });
})();
"""


async def event_listener() -> None:
    async for event in client.vision_events():
        asyncio.create_task(
            jobs.run(event.config_entry_id, OperatingMode(event.mode), event.plant_ids, "event")
        )


async def heartbeat() -> None:
    while True:
        await asyncio.sleep(settings.heartbeat_minutes * 60)
        if settings.selected_config_entry_id and not jobs.lock.locked():
            await jobs._status(settings.selected_config_entry_id, None, "idle", "ready")


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
    rows = database.recent_measurements()
    crop_slugs = sorted({row["crop_slug"] for row in rows})
    curves = {
        slug: fit_monotonic_curve(
            database.measurements_for_crop(slug), safety_margin_mm=settings.safety_margin_mm
        )
        for slug in crop_slugs
    }

    def _review_controls(r: dict) -> str:
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if not r.get("calibrated", 1):
            return "<span class=warn>Calibration required</span>"
        return (
            f'<form method=post action="recommendations/{r["measurement_id"]}/approve">'
            "<button>Approve</button></form>"
            f'<form method=post action="recommendations/{r["measurement_id"]}/reject">'
            "<button>Reject</button></form>"
        )

    measurement_rows = "".join(
        f"<tr><td>{r['plant_id']}</td><td>{escape(r['crop_slug'])}</td>"
        f"<td>{escape(str(r.get('processed_width') or '—'))}x{escape(str(r.get('processed_height') or '—'))}</td>"
        f"<td>{r['current_radius_mm']:.1f}</td>"
        f"<td>{r['maximum_accepted_canopy_radius_mm']:.1f}</td><td>{r['recommended_protection_radius_mm']:.1f}</td>"
        f"<td>{r['confidence']:.2f}</td><td>{escape(str(r.get('calibration_source') or '—'))}</td>"
        f"<td>{escape(r['decision'])}</td><td>{escape(r['reason'])}</td>"
        f'<td><a href="artifact/{escape(Path(r["overlay_path"]).name) if r["overlay_path"] else ""}">overlay</a></td>'
        f"<td>{_review_controls(r)}</td></tr>"
        for r in rows
    )
    last = jobs.last
    curve_rows = "".join(
        f"<tr><td>{escape(slug)}</td><td>{escape(str(curve))}</td><td>diameter mm</td></tr>"
        for slug, curve in curves.items()
    )
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
<section class=card><h2>Crop protection spread proposals</h2><p class=muted>Monotonic and limited to 10 points. FarmBot values are diameters; assignment requires approval.</p><table><tbody>{curve_rows or "<tr><td>No curve is ready</td></tr>"}</tbody></table></section>
<section class=card><h2>Approval and rollback history</h2><table><tbody>{decision_rows or "<tr><td>No decisions yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Safety warning</h2><p class=warn>Early experimental vision results must not be the sole basis for destructive automatic weeding.</p></section>"""
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


@app.get("/settings", response_class=HTMLResponse)
async def calibration_page(request: Request) -> HTMLResponse:
    calibration = database.active_calibration(settings.selected_config_entry_id)
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
            f"offsets=({calibration.offset_x_mm},{calibration.offset_y_mm}) mm"
        )
    body = f"""<section class=card><h2>Manual calibration</h2>
<p>Interactively calibrate against a recent FarmBot image at the configured analysis
resolution ({escape(resolution.label)}). Select an image, click point A then point B on
two features a known distance apart, enter that distance, then overlay the known plant
centres and confirm several align before saving. No external tools are needed.</p>
{warning_html}
<p class=muted>Current: {escape(current)}</p>
<div class=grid>
<div>
<label>FarmBot config entry ID<br><input id=entry_id value="{escape(settings.selected_config_entry_id)}"></label>
<p><button type=button id=load>Load recent images</button></p>
<label>Image<br><select id=image></select></label>
<p class=muted>Click point A (green), then point B (blue) on the image.</p>
<label>Known separation A→B (mm)<br><input id=distance type=number min=0.1 step=any></label>
<p id=ppm class=muted>Pixels per millimetre: —</p>
<label>Rotation (degrees)<br><input id=rotation type=number step=any value=0></label>
<label>Offset X (mm)<br><input id=offx type=number step=any value=0></label>
<label>Offset Y (mm)<br><input id=offy type=number step=any value=0></label>
<p><button type=button id=overlay>Overlay plant centres</button></p>
<label><input type=checkbox id=confirm> Plant centres align in this image</label>
<p><button type=button id=save disabled>Save calibration</button></p>
<p id=status class=muted></p>
</div>
<div>
<canvas id=canvas width={resolution.width} height={resolution.height}
 style="width:100%;border:1px solid #ccc;cursor:crosshair;background:#111"></canvas>
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
        {"id": p.id, "name": p.name, "x": p.x, "y": p.y, "radius": p.radius}
        for p in inventory.plants
    ]
    return JSONResponse({"images": images, "plants": plants})


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
    ax: float = Form(...),
    ay: float = Form(...),
    bx: float = Form(...),
    by: float = Form(...),
    distance_mm: float = Form(...),
    rotation: float = Form(0),
    offset_x: float = Form(0),
    offset_y: float = Form(0),
    image_id: int | None = Form(None),
) -> RedirectResponse:
    scale = manual_scale((ax, ay), (bx, by), distance_mm)
    resolution = settings.resolution
    database.save_calibration(
        entry_id,
        Calibration(
            source="manual",
            pixels_per_mm_x=scale,
            pixels_per_mm_y=scale,
            rotation_degrees=rotation,
            offset_x_mm=offset_x,
            offset_y_mm=offset_y,
            uncertainty_mm=settings.calibration_uncertainty_mm,
            analysis_resolution=resolution.value,
            image_id=image_id,
            processed_width=resolution.width,
            processed_height=resolution.height,
            point_a_x=ax,
            point_a_y=ay,
            point_b_x=bx,
            point_b_y=by,
            separation_mm=distance_mm,
        ),
    )
    return RedirectResponse("settings", status_code=303)


@app.get("/artifact/{name}")
async def artifact(name: str) -> FileResponse:
    safe_name = Path(name).name
    path = settings.data_dir / "artifacts" / safe_name
    if safe_name != name or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.post("/recommendations/{measurement_id}/{action}")
async def recommendation(request: Request, measurement_id: str, action: str) -> RedirectResponse:
    if action not in {"approve", "reject"}:
        raise HTTPException(400)
    row = database.connection.execute(
        "SELECT * FROM measurements WHERE measurement_id=?", (measurement_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404)
    if action == "approve":
        # Approval is impossible without a valid calibration (Part 6, Part 10).
        if "calibrated" in row.keys() and not row["calibrated"]:
            raise HTTPException(409, "calibration required for millimetre measurements")
        if row["recommended_protection_radius_mm"] <= row["current_radius_mm"]:
            raise HTTPException(409, "shrinking is disabled")
        await client.apply_radius(
            ApplyRadiusRequest(
                config_entry_id=settings.selected_config_entry_id,
                plant_id=row["plant_id"],
                measurement_id=measurement_id,
                expected_current_radius_mm=row["current_radius_mm"],
                recommended_radius_mm=row["recommended_protection_radius_mm"],
                confidence=row["confidence"],
                apply=True,
            )
        )
    database.record_decision(measurement_id, action, {})
    # The ingress prefix is dynamic; returning it explicitly prevents a
    # relative redirect from climbing above the Home Assistant session path.
    destination = ingress_base(request)
    if destination == "./":
        destination = "../../../"
    return RedirectResponse(destination, status_code=303)
