from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .models import Calibration, Measurement, OriginLocation

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
    # Migration 2 (contract v2): add resolution/scaling provenance to
    # calibrations and measurements. ADD COLUMN is non-destructive and keeps
    # every existing row; older rows simply have NULL in the new columns.
    """
    ALTER TABLE calibrations ADD COLUMN analysis_resolution TEXT;
    ALTER TABLE calibrations ADD COLUMN image_id INTEGER;
    ALTER TABLE calibrations ADD COLUMN processed_width INTEGER;
    ALTER TABLE calibrations ADD COLUMN processed_height INTEGER;
    ALTER TABLE calibrations ADD COLUMN source_width INTEGER;
    ALTER TABLE calibrations ADD COLUMN source_height INTEGER;
    ALTER TABLE calibrations ADD COLUMN oriented_width INTEGER;
    ALTER TABLE calibrations ADD COLUMN oriented_height INTEGER;
    ALTER TABLE calibrations ADD COLUMN resize_scale_x REAL;
    ALTER TABLE calibrations ADD COLUMN resize_scale_y REAL;
    ALTER TABLE calibrations ADD COLUMN basis TEXT;
    ALTER TABLE calibrations ADD COLUMN calibration_version TEXT;
    ALTER TABLE calibrations ADD COLUMN point_a_x REAL;
    ALTER TABLE calibrations ADD COLUMN point_a_y REAL;
    ALTER TABLE calibrations ADD COLUMN point_b_x REAL;
    ALTER TABLE calibrations ADD COLUMN point_b_y REAL;
    ALTER TABLE calibrations ADD COLUMN separation_mm REAL;
    ALTER TABLE calibrations ADD COLUMN transformed_from_id INTEGER;
    ALTER TABLE measurements ADD COLUMN analysis_resolution TEXT;
    ALTER TABLE measurements ADD COLUMN processed_width INTEGER;
    ALTER TABLE measurements ADD COLUMN processed_height INTEGER;
    ALTER TABLE measurements ADD COLUMN calibration_source TEXT;
    ALTER TABLE measurements ADD COLUMN calibrated INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN contract_version TEXT;
    """,
    # Migration 3: FarmBot-style origin location (garden<->pixel reflection).
    # Existing rows have NULL and are read back as TOP_LEFT, preserving the
    # exact transform every prior calibration produced.
    """
    ALTER TABLE calibrations ADD COLUMN origin_location TEXT;
    """,
    # Migration 4: plant-removal evidence, diagnostic artifact manifests, and
    # enough proposal context to safely resume a flagged per-plant curve edit.
    """
    ALTER TABLE measurements ADD COLUMN artifact_paths_json TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE measurements ADD COLUMN config_entry_id TEXT;
    ALTER TABLE measurements ADD COLUMN vegetation_absent INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN absent_observations INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN safety_margin_mm REAL NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN calibration_uncertainty_mm REAL NOT NULL DEFAULT 0;
    CREATE INDEX IF NOT EXISTS idx_measurements_absent_streak
      ON measurements(plant_id,image_timestamp DESC,vegetation_absent);
    ALTER TABLE curve_proposals ADD COLUMN config_entry_id TEXT;
    ALTER TABLE curve_proposals ADD COLUMN plant_id INTEGER;
    ALTER TABLE curve_proposals ADD COLUMN measurement_id TEXT;
    ALTER TABLE curve_proposals ADD COLUMN plant_age_days INTEGER;
    ALTER TABLE curve_proposals ADD COLUMN curve_name TEXT;
    ALTER TABLE curve_proposals ADD COLUMN reason TEXT;
    ALTER TABLE curve_proposals ADD COLUMN conflict_day INTEGER;
    ALTER TABLE curve_proposals ADD COLUMN conflict_old_diameter REAL;
    ALTER TABLE curve_proposals ADD COLUMN overlay_path TEXT;
    ALTER TABLE curve_proposals ADD COLUMN warning TEXT;
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

    # Sources that represent a user-owned manual calibration (vs a per-image
    # derived calibration recorded only for measurement provenance).
    _MANUAL_SOURCES = ("manual", "manual_transformed")

    def save_calibration(self, entry_id: str, calibration: Calibration) -> Calibration:
        """Persist a user manual calibration as the active one for the bot."""
        with self.connection:
            self.connection.execute(
                "UPDATE calibrations SET active=0 WHERE config_entry_id=? "
                "AND source IN ('manual','manual_transformed')",
                (entry_id,),
            )
            return self._insert_calibration(entry_id, calibration, active=1)

    def record_calibration(self, entry_id: str, calibration: Calibration) -> Calibration:
        """Record a derived (processed/reference) calibration for provenance.

        Does not touch the active manual calibration; it only mints a version
        row so a measurement can reference the exact calibration it used.
        """
        with self.connection:
            return self._insert_calibration(entry_id, calibration, active=0)

    def _insert_calibration(
        self, entry_id: str, calibration: Calibration, *, active: int
    ) -> Calibration:
        cursor = self.connection.execute(
            """INSERT INTO calibrations(active,config_entry_id,source,pixels_per_mm_x,
                   pixels_per_mm_y,rotation_degrees,offset_x_mm,offset_y_mm,uncertainty_mm,
                   analysis_resolution,image_id,processed_width,processed_height,source_width,
                   source_height,oriented_width,oriented_height,resize_scale_x,resize_scale_y,basis,
                   calibration_version,point_a_x,point_a_y,point_b_x,point_b_y,separation_mm,
                   transformed_from_id,origin_location)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                active,
                entry_id,
                calibration.source,
                calibration.pixels_per_mm_x,
                calibration.pixels_per_mm_y,
                calibration.rotation_degrees,
                calibration.offset_x_mm,
                calibration.offset_y_mm,
                calibration.uncertainty_mm,
                calibration.analysis_resolution,
                calibration.image_id,
                calibration.processed_width,
                calibration.processed_height,
                calibration.source_width,
                calibration.source_height,
                calibration.oriented_width,
                calibration.oriented_height,
                calibration.resize_scale_x,
                calibration.resize_scale_y,
                calibration.basis,
                calibration.calibration_version,
                calibration.point_a_x,
                calibration.point_a_y,
                calibration.point_b_x,
                calibration.point_b_y,
                calibration.separation_mm,
                calibration.transformed_from_id,
                str(calibration.origin_location),
            ),
        )
        return calibration.model_copy(update={"version_id": cursor.lastrowid})

    def active_calibration(self, entry_id: str) -> Calibration | None:
        """Return the active user manual calibration for a bot, if any."""
        row = self.connection.execute(
            "SELECT * FROM calibrations WHERE config_entry_id=? AND active=1 "
            "AND source IN ('manual','manual_transformed') ORDER BY id DESC LIMIT 1",
            (entry_id,),
        ).fetchone()
        if not row:
            return None
        keys = row.keys()

        def _opt(name: str) -> object:
            return row[name] if name in keys else None

        return Calibration(
            version_id=row["id"],
            source=row["source"],
            pixels_per_mm_x=row["pixels_per_mm_x"],
            pixels_per_mm_y=row["pixels_per_mm_y"],
            rotation_degrees=row["rotation_degrees"],
            offset_x_mm=row["offset_x_mm"],
            offset_y_mm=row["offset_y_mm"],
            uncertainty_mm=row["uncertainty_mm"],
            analysis_resolution=_opt("analysis_resolution"),
            image_id=_opt("image_id"),
            processed_width=_opt("processed_width"),
            processed_height=_opt("processed_height"),
            source_width=_opt("source_width"),
            source_height=_opt("source_height"),
            oriented_width=_opt("oriented_width"),
            oriented_height=_opt("oriented_height"),
            resize_scale_x=_opt("resize_scale_x"),
            resize_scale_y=_opt("resize_scale_y"),
            basis=_opt("basis"),
            calibration_version=_opt("calibration_version"),
            point_a_x=_opt("point_a_x"),
            point_a_y=_opt("point_a_y"),
            point_b_x=_opt("point_b_x"),
            point_b_y=_opt("point_b_y"),
            separation_mm=_opt("separation_mm"),
            transformed_from_id=_opt("transformed_from_id"),
            origin_location=_opt("origin_location") or OriginLocation.TOP_LEFT,
        )

    def save_measurements(self, measurements: Iterable[Measurement]) -> None:
        values = []
        for m in measurements:
            values.append(
                (
                    str(m.measurement_id),
                    m.config_entry_id,
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
                    m.analysis_resolution,
                    m.processed_width,
                    m.processed_height,
                    m.calibration_source,
                    int(m.calibrated),
                    m.contract_version,
                    json.dumps(m.artifact_paths, separators=(",", ":")),
                    int(m.vegetation_absent),
                    m.absent_observations,
                    m.safety_margin_mm,
                    m.calibration_uncertainty_mm,
                )
            )
        with self.connection:
            self.connection.executemany(
                """INSERT OR REPLACE INTO measurements(measurement_id,config_entry_id,plant_id,crop_slug,plant_age_days,
                image_id,image_timestamp,current_radius_mm,typical_canopy_radius_mm,
                maximum_accepted_canopy_radius_mm,recommended_protection_radius_mm,confidence,
                calibration_version_id,transform_json,algorithm_version,decision,reason,ambiguous,applied,
                mask_path,overlay_path,analysis_resolution,processed_width,processed_height,
                calibration_source,calibrated,contract_version,artifact_paths_json,
                vegetation_absent,absent_observations,safety_margin_mm,calibration_uncertainty_mm)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )

    def latest_mask_path(self, plant_id: int) -> str | None:
        row = self.connection.execute(
            "SELECT mask_path FROM measurements WHERE plant_id=? AND mask_path IS NOT NULL "
            "AND vegetation_absent=0 "
            "ORDER BY image_timestamp DESC LIMIT 1",
            (plant_id,),
        ).fetchone()
        return None if row is None else row[0]

    def absent_streak(
        self,
        config_entry_id: str,
        plant_id: int,
        *,
        current_image_id: int | None = None,
        current_image_timestamp: datetime | None = None,
    ) -> int:
        """Count absent distinct images, optionally replacing the current image with absent."""
        rows = self.connection.execute(
            "SELECT image_id,vegetation_absent,image_timestamp FROM measurements "
            "WHERE config_entry_id=? AND plant_id=? "
            "ORDER BY image_timestamp DESC,created_at DESC",
            (config_entry_id, plant_id),
        )
        by_image: dict[int, tuple[str, bool]] = {}
        for row in rows:
            by_image.setdefault(int(row[0]), (str(row[2]), bool(row[1])))
        if current_image_id is not None and current_image_timestamp is not None:
            by_image[current_image_id] = (current_image_timestamp.isoformat(), True)
        count = 0
        for _, is_absent in sorted(by_image.values(), key=lambda item: item[0], reverse=True):
            if not is_absent:
                break
            count += 1
        return count

    def has_present_measurement(self, config_entry_id: str, plant_id: int) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM measurements WHERE config_entry_id=? AND plant_id=? "
                "AND vegetation_absent=0 LIMIT 1",
                (config_entry_id, plant_id),
            ).fetchone()
            is not None
        )

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
        result = [dict(row) for row in rows]
        for row in result:
            try:
                row["artifact_paths"] = json.loads(row.get("artifact_paths_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                row["artifact_paths"] = []
        return result

    def pending_measurements(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            """SELECT m.* FROM measurements m
            WHERE NOT EXISTS (
              SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id
              AND d.action IN ('applied','reject','removed','keep')
            )
            ORDER BY m.image_timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            try:
                row["artifact_paths"] = json.loads(row.get("artifact_paths_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                row["artifact_paths"] = []
        return result

    def has_terminal_decision(self, measurement_id: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM decisions WHERE measurement_id=? "
                "AND action IN ('applied','reject','removed','keep') LIMIT 1",
                (measurement_id,),
            ).fetchone()
            is not None
        )

    def is_latest_plant_measurement(
        self, config_entry_id: str, plant_id: int, measurement_id: str
    ) -> bool:
        row = self.connection.execute(
            "SELECT measurement_id FROM measurements WHERE config_entry_id=? AND plant_id=? "
            "ORDER BY image_timestamp DESC,created_at DESC LIMIT 1",
            (config_entry_id, plant_id),
        ).fetchone()
        return row is not None and row[0] == measurement_id

    def measurement(self, measurement_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM measurements WHERE measurement_id=?", (measurement_id,)
        ).fetchone()
        return None if row is None else dict(row)

    def create_curve_proposal(
        self,
        *,
        config_entry_id: str,
        plant_id: int,
        measurement_id: str,
        crop_slug: str,
        plant_age_days: int,
        curve_id: int | None,
        curve_name: str,
        previous_data: dict[str, float],
        data: dict[str, float],
        reason: str,
        conflict_day: int | None,
        conflict_old_diameter: float | None,
        overlay_path: str | None,
        warning: str | None = None,
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO curve_proposals(
                config_entry_id,plant_id,measurement_id,crop_slug,curve_type,plant_age_days,
                farmbot_curve_id,curve_name,previous_data_json,data_json,status,reason,
                conflict_day,conflict_old_diameter,overlay_path,warning)
                VALUES(?,?,?,?,?,?,?,?,?,?,'flagged',?,?,?,?,?)""",
                (
                    config_entry_id,
                    plant_id,
                    measurement_id,
                    crop_slug,
                    "spread",
                    plant_age_days,
                    curve_id,
                    curve_name,
                    json.dumps(previous_data, separators=(",", ":")),
                    json.dumps(data, separators=(",", ":")),
                    reason,
                    conflict_day,
                    conflict_old_diameter,
                    overlay_path,
                    warning,
                ),
            )
        return int(cursor.lastrowid)

    def curve_proposals(self, status: str = "flagged") -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM curve_proposals WHERE status=? ORDER BY created_at DESC", (status,)
        ).fetchall()
        return [dict(row) for row in rows]

    def curve_proposal(self, proposal_id: int) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM curve_proposals WHERE id=?", (proposal_id,)
        ).fetchone()
        return None if row is None else dict(row)

    def update_curve_proposal(self, proposal_id: int, status: str, data: dict | None = None) -> None:
        with self.connection:
            if data is None:
                self.connection.execute(
                    "UPDATE curve_proposals SET status=? WHERE id=?", (status, proposal_id)
                )
            else:
                self.connection.execute(
                    "UPDATE curve_proposals SET status=?,data_json=? WHERE id=?",
                    (status, json.dumps(data, separators=(",", ":")), proposal_id),
                )

    def record_decision(self, measurement_id: str, action: str, details: dict) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO decisions(measurement_id,action,details_json) VALUES(?,?,?)",
                (measurement_id, action, json.dumps(details, separators=(",", ":"))),
            )

    def update_measurement_outcome(
        self, measurement_id: str, *, decision: str, applied: bool
    ) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE measurements SET decision=?,applied=? WHERE measurement_id=?",
                (decision, int(applied), measurement_id),
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
