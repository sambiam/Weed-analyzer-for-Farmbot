from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .models import Calibration, Measurement

MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS calibrations(
      id INTEGER PRIMARY KEY AUTOINCREMENT, config_entry_id TEXT NOT NULL,
      source TEXT NOT NULL, pixels_per_mm_x REAL NOT NULL, pixels_per_mm_y REAL NOT NULL,
      rotation_degrees REAL NOT NULL, offset_x_mm REAL NOT NULL, offset_y_mm REAL NOT NULL,
      uncertainty_mm REAL NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      active INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS measurements(
      measurement_id TEXT PRIMARY KEY, plant_id INTEGER NOT NULL, crop_slug TEXT NOT NULL,
      planted_at TEXT, plant_age_days INTEGER, image_id INTEGER NOT NULL, image_timestamp TEXT NOT NULL,
      current_radius_mm REAL NOT NULL, typical_canopy_radius_mm REAL NOT NULL,
      maximum_accepted_canopy_radius_mm REAL NOT NULL, recommended_protection_radius_mm REAL NOT NULL,
      confidence REAL NOT NULL, calibration_version_id INTEGER, transform_json TEXT NOT NULL,
      algorithm_version TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL,
      ambiguous INTEGER NOT NULL, applied INTEGER NOT NULL, mask_path TEXT, overlay_path TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(calibration_version_id) REFERENCES calibrations(id)
    );
    CREATE INDEX IF NOT EXISTS idx_measurements_plant_time ON measurements(plant_id,image_timestamp);
    CREATE INDEX IF NOT EXISTS idx_measurements_crop_age ON measurements(crop_slug,plant_age_days);
    CREATE TABLE IF NOT EXISTS jobs(
      id TEXT PRIMARY KEY, config_entry_id TEXT NOT NULL, trigger TEXT NOT NULL, mode TEXT NOT NULL,
      status TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT, plants_analysed INTEGER DEFAULT 0,
      plants_skipped INTEGER DEFAULT 0, recommendations INTEGER DEFAULT 0, applied INTEGER DEFAULT 0,
      uncertain INTEGER DEFAULT 0, cpu_seconds REAL, peak_memory_mb REAL, message TEXT
    );
    CREATE TABLE IF NOT EXISTS curve_proposals(
      id INTEGER PRIMARY KEY AUTOINCREMENT, crop_slug TEXT NOT NULL, curve_type TEXT NOT NULL,
      data_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'proposed', farmbot_curve_id INTEGER,
      previous_data_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS vision_owned_curves(
      config_entry_id TEXT NOT NULL, crop_slug TEXT NOT NULL, farmbot_curve_id INTEGER NOT NULL,
      adopted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(config_entry_id,farmbot_curve_id)
    );
    CREATE TABLE IF NOT EXISTS decisions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, measurement_id TEXT NOT NULL, action TEXT NOT NULL,
      details_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
]


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = self._connect()
        try:
            self.migrate()
            self.connection.execute("PRAGMA quick_check").fetchone()
        except sqlite3.DatabaseError:
            self.connection.close()
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            for source in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
                if source.exists():
                    source.replace(path.with_name(f"{source.name}.corrupt-{timestamp}"))
            self.connection = self._connect()
            self.migrate()
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET status='interrupted',completed_at=?,message='container restarted' "
                "WHERE status='running'",
                (datetime.now(UTC).isoformat(),),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def migrate(self) -> None:
        current = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        version = (
            0
            if current is None
            else self.connection.execute(
                "SELECT COALESCE(MAX(version),0) FROM schema_version"
            ).fetchone()[0]
        )
        for number, sql in enumerate(MIGRATIONS, start=1):
            if number > version:
                with self.connection:
                    self.connection.executescript(sql)
                    self.connection.execute(
                        "INSERT INTO schema_version(version) VALUES (?)", (number,)
                    )

    def save_calibration(self, entry_id: str, calibration: Calibration) -> Calibration:
        with self.connection:
            self.connection.execute(
                "UPDATE calibrations SET active=0 WHERE config_entry_id=?", (entry_id,)
            )
            cursor = self.connection.execute(
                """INSERT INTO calibrations(config_entry_id,source,pixels_per_mm_x,pixels_per_mm_y,
                   rotation_degrees,offset_x_mm,offset_y_mm,uncertainty_mm)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    entry_id,
                    calibration.source,
                    calibration.pixels_per_mm_x,
                    calibration.pixels_per_mm_y,
                    calibration.rotation_degrees,
                    calibration.offset_x_mm,
                    calibration.offset_y_mm,
                    calibration.uncertainty_mm,
                ),
            )
        return calibration.model_copy(update={"version_id": cursor.lastrowid})

    def active_calibration(self, entry_id: str) -> Calibration | None:
        row = self.connection.execute(
            "SELECT * FROM calibrations WHERE config_entry_id=? AND active=1 ORDER BY id DESC LIMIT 1",
            (entry_id,),
        ).fetchone()
        if not row:
            return None
        return Calibration(
            version_id=row["id"],
            source=row["source"],
            pixels_per_mm_x=row["pixels_per_mm_x"],
            pixels_per_mm_y=row["pixels_per_mm_y"],
            rotation_degrees=row["rotation_degrees"],
            offset_x_mm=row["offset_x_mm"],
            offset_y_mm=row["offset_y_mm"],
            uncertainty_mm=row["uncertainty_mm"],
        )

    def save_measurements(self, measurements: Iterable[Measurement]) -> None:
        values = []
        for m in measurements:
            values.append(
                (
                    str(m.measurement_id),
                    m.plant_id,
                    m.crop_slug,
                    m.plant_age_days,
                    m.image_id,
                    m.image_timestamp.isoformat(),
                    m.current_radius_mm,
                    m.typical_canopy_radius_mm,
                    m.maximum_accepted_canopy_radius_mm,
                    m.recommended_protection_radius_mm,
                    m.confidence,
                    m.calibration_version_id,
                    m.transform_json,
                    m.algorithm_version,
                    m.decision.value,
                    m.reason,
                    int(m.ambiguous),
                    int(m.applied),
                    m.mask_path,
                    m.overlay_path,
                )
            )
        with self.connection:
            self.connection.executemany(
                """INSERT OR REPLACE INTO measurements(measurement_id,plant_id,crop_slug,plant_age_days,
                image_id,image_timestamp,current_radius_mm,typical_canopy_radius_mm,
                maximum_accepted_canopy_radius_mm,recommended_protection_radius_mm,confidence,
                calibration_version_id,transform_json,algorithm_version,decision,reason,ambiguous,applied,
                mask_path,overlay_path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )

    def latest_mask_path(self, plant_id: int) -> str | None:
        row = self.connection.execute(
            "SELECT mask_path FROM measurements WHERE plant_id=? AND mask_path IS NOT NULL "
            "ORDER BY image_timestamp DESC LIMIT 1",
            (plant_id,),
        ).fetchone()
        return None if row is None else row[0]

    def measurements_for_crop(self, crop_slug: str) -> list[tuple[int, float]]:
        return [
            tuple(row)
            for row in self.connection.execute(
                "SELECT plant_age_days,maximum_accepted_canopy_radius_mm FROM measurements "
                "WHERE crop_slug=? AND plant_age_days IS NOT NULL AND confidence>=0.6 ORDER BY plant_age_days",
                (crop_slug,),
            )
        ]

    def recent_measurements(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM measurements ORDER BY image_timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def record_decision(self, measurement_id: str, action: str, details: dict) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO decisions(measurement_id,action,details_json) VALUES(?,?,?)",
                (measurement_id, action, json.dumps(details, separators=(",", ":"))),
            )

    def start_job(
        self, job_id: str, entry_id: str, trigger: str, mode: str, started_at: str
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO jobs(id,config_entry_id,trigger,mode,status,started_at) "
                "VALUES(?,?,?,?,?,?)",
                (job_id, entry_id, trigger, mode, "running", started_at),
            )

    def finish_job(self, job_id: str, result: dict) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE jobs SET status=?,completed_at=?,plants_analysed=?,recommendations=?,
                applied=?,uncertain=?,cpu_seconds=?,peak_memory_mb=?,message=? WHERE id=?""",
                (
                    result["status"],
                    result["completed_at"],
                    result["plants_analysed"],
                    result["recommendations"],
                    result["automatically_applied"],
                    result["uncertain"],
                    result["cpu_seconds"],
                    result["peak_memory_mb"],
                    result["message"],
                    job_id,
                ),
            )

    def recent_decisions(self, limit: int = 20) -> list[dict]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        ]

    def stats(self) -> dict[str, int]:
        return {
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "measurements": self.connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[
                0
            ],
        }
