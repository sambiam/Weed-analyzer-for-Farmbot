from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import cv2
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from . import ALGORITHM_VERSION, __version__
from .curves import fit_monotonic_curve
from .database import Database
from .home_assistant import HomeAssistantClient
from .jobs import JobManager
from .models import ApplyRadiusRequest, Calibration, OperatingMode
from .settings import Settings
from .vision import manual_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
settings = Settings.load()
database = Database(settings.data_dir / "farmbot_vision.db")
client = HomeAssistantClient()
jobs = JobManager(settings, database, client)


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
    tasks = [
        asyncio.create_task(event_listener()),
        asyncio.create_task(heartbeat()),
        asyncio.create_task(scheduler()),
        asyncio.create_task(retention_cleanup()),
    ]
    yield
    for task in tasks:
        task.cancel()
    await client.close()


app = FastAPI(
    title="FarmBot Vision", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None
)


def ingress_base(request: Request) -> str:
    value = request.headers.get("X-Ingress-Path", "./").strip()
    return f"{value.rstrip('/')}/" if value not in {"", ".", "./"} else "./"


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
    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "algorithm_version": ALGORITHM_VERSION,
            "opencv_threads": cv2.getNumThreads(),
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
    measurement_rows = "".join(
        f"<tr><td>{r['plant_id']}</td><td>{escape(r['crop_slug'])}</td><td>{r['current_radius_mm']:.1f}</td>"
        f"<td>{r['maximum_accepted_canopy_radius_mm']:.1f}</td><td>{r['recommended_protection_radius_mm']:.1f}</td>"
        f"<td>{r['confidence']:.2f}</td><td>{escape(r['decision'])}</td><td>{escape(r['reason'])}</td>"
        f'<td><a href="artifact/{escape(Path(r["overlay_path"]).name) if r["overlay_path"] else ""}">overlay</a></td>'
        f'<td><form method=post action="recommendations/{r["measurement_id"]}/approve"><button>Approve</button></form>'
        f'<form method=post action="recommendations/{r["measurement_id"]}/reject"><button>Reject</button></form></td></tr>'
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
    body = f"""<div class=grid><section class=card><h2>Health</h2><b>{escape(jobs.current["status"])}</b>
<p>{escape(jobs.current.get("progress", ""))}</p><p class=muted>Version {__version__} · {ALGORITHM_VERSION}</p></section>
<section class=card><h2>FarmBot</h2><p>{escape(settings.selected_config_entry_id or "Not selected")}</p>
<p>Mode: {settings.mode.value}</p></section><section class=card><h2>Last job</h2>
<p>{escape(last.get("message", "Never run"))}</p><p>CPU {last.get("cpu_seconds", "—")} s · peak {last.get("peak_memory_mb", "—")} MB</p></section>
<section class=card><h2>Queue</h2><p>{jobs.current.get("queue_length", 0)} waiting</p>
<form method=post action="analyse"><button>Analyse now</button></form></section></div>
<section class=card><h2>Measurements</h2><table><thead><tr><th>Plant</th><th>Crop</th><th>Current</th><th>Max leaf</th><th>Recommended</th><th>Confidence</th><th>Decision</th><th>Reason</th><th>Diagnostic</th><th>Review</th></tr></thead><tbody>{measurement_rows or "<tr><td colspan=10>No measurements yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Crop protection spread proposals</h2><p class=muted>Monotonic and limited to 10 points. FarmBot values are diameters; assignment requires approval.</p><table><tbody>{curve_rows or "<tr><td>No curve is ready</td></tr>"}</tbody></table></section>
<section class=card><h2>Approval and rollback history</h2><table><tbody>{decision_rows or "<tr><td>No decisions yet</td></tr>"}</tbody></table></section>
<section class=card><h2>Safety warning</h2><p class=warn>Early experimental vision results must not be the sole basis for destructive automatic weeding.</p></section>"""
    return layout(request, body)


@app.post("/analyse")
async def analyse(background: BackgroundTasks) -> RedirectResponse:
    background.add_task(jobs.run, trigger="manual")
    return RedirectResponse("./", status_code=303)


@app.get("/settings", response_class=HTMLResponse)
async def calibration_page(request: Request) -> HTMLResponse:
    calibration = database.active_calibration(settings.selected_config_entry_id)
    body = f"""<section class=card><h2>Manual calibration</h2>
<p>Current: {escape(str(calibration.model_dump() if calibration else "none"))}</p>
<p>Choose two pixel points in a representative image, enter their measured ground separation, then validate known plant centres against the image.</p>
<form method=post action="calibration"><label>Entry ID <input name=entry_id required value="{escape(settings.selected_config_entry_id)}"></label><br>
<label>Point A x,y <input name=ax type=number step=any required> <input name=ay type=number step=any required></label><br>
<label>Point B x,y <input name=bx type=number step=any required> <input name=by type=number step=any required></label><br>
<label>Separation mm <input name=distance_mm type=number min=.1 step=any required></label><br>
<label>Rotation degrees <input name=rotation type=number step=any value=0></label><br>
<label>X/Y offsets mm <input name=offset_x type=number step=any value=0> <input name=offset_y type=number step=any value=0></label><br>
<button>Save calibration</button></form></section>"""
    return layout(request, body, "Calibration · FarmBot Vision")


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
) -> RedirectResponse:
    scale = manual_scale((ax, ay), (bx, by), distance_mm)
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
async def recommendation(measurement_id: str, action: str) -> RedirectResponse:
    if action not in {"approve", "reject"}:
        raise HTTPException(400)
    row = database.connection.execute(
        "SELECT * FROM measurements WHERE measurement_id=?", (measurement_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404)
    if action == "approve":
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
    return RedirectResponse("../../../", status_code=303)
