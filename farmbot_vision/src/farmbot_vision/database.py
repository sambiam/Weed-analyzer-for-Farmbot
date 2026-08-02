from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .models import (
    Calibration,
    Measurement,
    OriginLocation,
    SoilMeasurement,
    SoilStereoCalibration,
)
from .plant_measurement import select_measurement_evidence, selection_diagnostics

LOGGER = logging.getLogger(__name__)
CREATED_WEED_SYNC_GUARD_HOURS = 24

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
    # Migration 5: centre-alignment alternatives and reviewable weed detections.
    """
    ALTER TABLE measurements ADD COLUMN center_misaligned INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN recommended_center_x REAL;
    ALTER TABLE measurements ADD COLUMN recommended_center_y REAL;
    CREATE TABLE IF NOT EXISTS weed_detections(
      detection_id TEXT PRIMARY KEY, config_entry_id TEXT NOT NULL, image_id INTEGER NOT NULL,
      image_timestamp TEXT NOT NULL, x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL DEFAULT 0,
      area_mm2 REAL NOT NULL, radius_mm REAL NOT NULL, confidence REAL NOT NULL,
      overlay_path TEXT, status TEXT NOT NULL DEFAULT 'recommended',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_weed_detections_pending
      ON weed_detections(status,image_timestamp DESC);
    """,
    # Migration 6: weed detection pixel geometry, so the review UI can
    # highlight a single weed on its source overlay image.
    """
    ALTER TABLE weed_detections ADD COLUMN center_px_x REAL;
    ALTER TABLE weed_detections ADD COLUMN center_px_y REAL;
    ALTER TABLE weed_detections ADD COLUMN processed_width INTEGER;
    ALTER TABLE weed_detections ADD COLUMN processed_height INTEGER;
    """,
    # Migration 7: retain the plant's recorded garden position alongside any
    # suggested centre so removal review remains understandable even if the
    # FarmBot point is later changed or archived.
    """
    ALTER TABLE measurements ADD COLUMN recorded_center_x REAL;
    ALTER TABLE measurements ADD COLUMN recorded_center_y REAL;
    """,
    # Migration 8: a photo-only weed review artifact keeps weed markers visible
    # while allowing the segmentation/ownership overlay to be hidden.
    """
    ALTER TABLE weed_detections ADD COLUMN review_path TEXT;
    """,
    # Migration 9: multi-image measurement evidence/composites and persistent
    # known-weed tracking for radius updates and disappearance observations.
    """
    ALTER TABLE measurements ADD COLUMN plant_center_px_x REAL;
    ALTER TABLE measurements ADD COLUMN plant_center_px_y REAL;
    ALTER TABLE measurements ADD COLUMN visible_fraction REAL NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN source_image_path TEXT;
    ALTER TABLE measurements ADD COLUMN composite_path TEXT;
    CREATE TABLE IF NOT EXISTS weed_tracks(
      config_entry_id TEXT NOT NULL, weed_id INTEGER NOT NULL,
      x REAL NOT NULL, y REAL NOT NULL, radius_mm REAL NOT NULL,
      last_seen_at TEXT, absent_observations INTEGER NOT NULL DEFAULT 0,
      confidence REAL NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active',
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY(config_entry_id,weed_id)
    );
    """,
    # Migration 10: supplemental virtual-stereo soil-height calibration,
    # measurements, review decisions and restart-safe job state.
    """
    CREATE TABLE IF NOT EXISTS soil_calibrations(
      id INTEGER PRIMARY KEY AUTOINCREMENT, config_entry_id TEXT NOT NULL,
      point_id INTEGER NOT NULL, capture_z REAL NOT NULL, baseline_mm REAL NOT NULL,
      reference_distance_mm REAL NOT NULL, z_direction INTEGER NOT NULL,
      inverse_depth_slope REAL NOT NULL, inverse_depth_intercept REAL NOT NULL,
      residual_mm REAL NOT NULL, processed_width INTEGER NOT NULL,
      processed_height INTEGER NOT NULL, source_width INTEGER NOT NULL,
      source_height INTEGER NOT NULL, source_image_ids_json TEXT NOT NULL,
      camera_signature TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_soil_calibrations_active
      ON soil_calibrations(config_entry_id,active,created_at DESC);
    CREATE TABLE IF NOT EXISTS soil_jobs(
      id TEXT PRIMARY KEY, config_entry_id TEXT NOT NULL, kind TEXT NOT NULL,
      status TEXT NOT NULL, point_ids_json TEXT NOT NULL, current_point_id INTEGER,
      completed_count INTEGER NOT NULL DEFAULT 0, failed_count INTEGER NOT NULL DEFAULT 0,
      stop_requested INTEGER NOT NULL DEFAULT 0, message TEXT,
      started_at TEXT NOT NULL, completed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS soil_measurements(
      measurement_id TEXT PRIMARY KEY, config_entry_id TEXT NOT NULL,
      point_id INTEGER NOT NULL, point_name TEXT NOT NULL,
      expected_x REAL NOT NULL, expected_y REAL NOT NULL, old_z_mm REAL NOT NULL,
      proposed_z_mm REAL, confidence REAL NOT NULL, uncertainty_mm REAL,
      status TEXT NOT NULL, reason TEXT NOT NULL, capture_id TEXT,
      calibration_id INTEGER, frame_ids_json TEXT NOT NULL,
      metrics_json TEXT NOT NULL, artifact_paths_json TEXT NOT NULL,
      algorithm_version TEXT NOT NULL, created_at TEXT NOT NULL,
      FOREIGN KEY(calibration_id) REFERENCES soil_calibrations(id)
    );
    CREATE INDEX IF NOT EXISTS idx_soil_measurements_point_time
      ON soil_measurements(config_entry_id,point_id,created_at DESC);
    CREATE TABLE IF NOT EXISTS soil_decisions(
      id INTEGER PRIMARY KEY AUTOINCREMENT, measurement_id TEXT NOT NULL,
      action TEXT NOT NULL, details_json TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    ALTER TABLE soil_calibrations ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_soil_calibrations_version
      ON soil_calibrations(config_entry_id,version);
    """,
    """
    ALTER TABLE soil_measurements ADD COLUMN point_updated_at TEXT;
    ALTER TABLE soil_measurements ADD COLUMN capture_x REAL;
    ALTER TABLE soil_measurements ADD COLUMN capture_y REAL;
    ALTER TABLE soil_measurements ADD COLUMN relocation_distance_mm REAL;
    """,
    # Migration 14: explainable weed features, review-derived training labels,
    # locally trained verifier history, and multi-image candidate tracks.
    """
    ALTER TABLE weed_detections ADD COLUMN heuristic_confidence REAL;
    ALTER TABLE weed_detections ADD COLUMN verifier_confidence REAL;
    ALTER TABLE weed_detections ADD COLUMN features_json TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE weed_detections ADD COLUMN crop_path TEXT;
    ALTER TABLE weed_detections ADD COLUMN observation_count INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE weed_detections ADD COLUMN candidate_track_id INTEGER;
    CREATE TABLE IF NOT EXISTS weed_training_samples(
      detection_id TEXT PRIMARY KEY, label TEXT NOT NULL, features_json TEXT NOT NULL,
      crop_path TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_weed_training_samples_label
      ON weed_training_samples(label,created_at);
    CREATE TABLE IF NOT EXISTS weed_candidate_tracks(
      id INTEGER PRIMARY KEY AUTOINCREMENT, config_entry_id TEXT NOT NULL,
      x REAL NOT NULL, y REAL NOT NULL, observations INTEGER NOT NULL DEFAULT 1,
      first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, last_image_id INTEGER NOT NULL,
      mean_confidence REAL NOT NULL, status TEXT NOT NULL DEFAULT 'active'
    );
    CREATE INDEX IF NOT EXISTS idx_weed_candidate_tracks_entry
      ON weed_candidate_tracks(config_entry_id,status,last_seen_at);
    CREATE TABLE IF NOT EXISTS weed_model_runs(
      id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
      sample_count INTEGER NOT NULL, positive_count INTEGER NOT NULL,
      negative_count INTEGER NOT NULL, metrics_json TEXT NOT NULL
    );
    """,
    # Migration 15: a clean plant composite and its optional ownership-mask
    # variant support a two-state review viewer without exposing raw masks.
    """
    ALTER TABLE measurements ADD COLUMN composite_overlay_path TEXT;
    """,
    # Migration 16: retain a calibrated multi-image canopy-mask measurement
    # alongside the original per-image observations.
    """
    ALTER TABLE measurements ADD COLUMN fused_canopy INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN fused_typical_radius_mm REAL;
    ALTER TABLE measurements ADD COLUMN fused_maximum_radius_mm REAL;
    ALTER TABLE measurements ADD COLUMN fused_recommended_radius_mm REAL;
    ALTER TABLE measurements ADD COLUMN fused_confidence REAL;
    ALTER TABLE measurements ADD COLUMN fusion_view_count INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN fusion_angular_coverage REAL NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN fusion_corroborated_fraction REAL NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN fusion_disagreement_mm REAL;
    ALTER TABLE measurements ADD COLUMN fusion_reliable INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN fusion_diagnostic_path TEXT;
    """,
    # Migration 17: per-plant image unlinks. A rejected recommendation or kept
    # removal marks the image(s) it was based on as no longer valid evidence
    # for that specific plant, without touching the image file itself -- the
    # same capture can still legitimately back other plants and the
    # whole-garden photo-grid mosaic.
    """
    CREATE TABLE IF NOT EXISTS plant_image_unlinks(
      config_entry_id TEXT NOT NULL, plant_id INTEGER NOT NULL, image_id INTEGER NOT NULL,
      measurement_id TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY(config_entry_id, plant_id, image_id)
    );
    """,
    # Migration 18: keep candidate/useful/selected evidence separate and retain
    # measurable boundary coverage plus confidence factors for diagnostics.
    # Legacy rows default to useful/full-view semantics so existing pending
    # recommendations remain reviewable.
    """
    ALTER TABLE measurements ADD COLUMN center_visible INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN boundary_coverage REAL NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN boundary_sectors_json TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE measurements ADD COLUMN canopy_truncated INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE measurements ADD COLUMN has_plant_evidence INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN plant_fits_single_frame INTEGER NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN image_quality REAL NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN segmentation_quality REAL NOT NULL DEFAULT 1;
    ALTER TABLE measurements ADD COLUMN evidence_status TEXT NOT NULL DEFAULT 'candidate';
    ALTER TABLE measurements ADD COLUMN exclusion_reason TEXT;
    ALTER TABLE measurements ADD COLUMN diagnostics_json TEXT NOT NULL DEFAULT '{}';
    """,
    # Migration 19: dashboard and grid aggregation read these columns for every
    # navigation and completed image batch. Cover them so an accumulated
    # measurement history cannot turn tab loading into a full-table scan.
    """
    CREATE INDEX IF NOT EXISTS idx_measurements_pending_time
      ON measurements(image_timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_measurements_entry_plant_image
      ON measurements(config_entry_id,plant_id,image_id,image_timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_decisions_measurement_action
      ON decisions(measurement_id,action);
    """,
    # Migration 20: one normalized audit stream for changes that were actually
    # applied to FarmBot.  Review decisions remain useful for queue state, but
    # this table is intentionally limited to the user-facing change log.
    """
    CREATE TABLE IF NOT EXISTS change_log(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      entity_type TEXT NOT NULL,
      entity_id TEXT NOT NULL,
      crop_type TEXT NOT NULL,
      x REAL,
      y REAL,
      z REAL,
      change_type TEXT NOT NULL,
      original_radius_mm REAL NOT NULL,
      current_radius_mm REAL NOT NULL,
      decision_method TEXT NOT NULL,
      confidence REAL NOT NULL,
      details_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_change_log_created
      ON change_log(created_at DESC,id DESC);
    """,
    # Migration 21: consecutive verifier-confirmed sightings are required
    # before a known weed may be widened automatically.
    """
    ALTER TABLE weed_tracks ADD COLUMN present_observations INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE weed_tracks ADD COLUMN last_observation_image_id INTEGER;
    CREATE TABLE IF NOT EXISTS known_weed_observations(
      config_entry_id TEXT NOT NULL, weed_id INTEGER NOT NULL, image_id INTEGER NOT NULL,
      status TEXT NOT NULL, confidence REAL NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY(config_entry_id,weed_id,image_id)
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
            self.connection.execute(
                "UPDATE soil_jobs SET status='interrupted',completed_at=?,"
                "message='container restarted' WHERE status='running'",
                (datetime.now(UTC).isoformat(),),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        connection.row_factory = sqlite3.Row
        # Keep SQLite auxiliary temporary storage away from inaccessible host
        # temp directories in the restricted add-on container.
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _reconnect(self) -> None:
        """Reopen the data file after a transient SQLite storage failure."""

        try:
            self.connection.close()
        finally:
            self.connection = self._connect()

    def _storage_summary(self) -> str:
        """Return compact storage diagnostics without masking the original error."""

        try:
            usage = shutil.disk_usage(self.path.parent)
            free = usage.free
        except OSError:
            free = None

        def size(path: Path) -> int:
            try:
                return path.stat().st_size
            except OSError:
                return 0

        free_text = "unknown" if free is None else str(free)
        return (
            f"path={self.path} free_bytes={free_text} "
            f"sqlite_tmpdir={os.environ.get('SQLITE_TMPDIR') or '(unset)'} "
            f"db_bytes={size(self.path)} wal_bytes={size(Path(f'{self.path}-wal'))} "
            f"shm_bytes={size(Path(f'{self.path}-shm'))}"
        )

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
        """Persist measurements, reopening SQLite once after a storage error."""

        measurements = list(measurements)
        try:
            self._save_measurements_once(measurements)
        except sqlite3.OperationalError as exc:
            if "unable to open database file" not in str(exc).casefold():
                raise
            LOGGER.warning(
                "SQLite measurement write could not open the database; reconnecting once "
                "before retrying (%s; sqlite_error=%s/%s): %s",
                self._storage_summary(),
                getattr(exc, "sqlite_errorcode", "unknown"),
                getattr(exc, "sqlite_errorname", "unknown"),
                exc,
            )
            self._reconnect()
            self._save_measurements_once(measurements)

    def _save_measurements_once(self, measurements: list[Measurement]) -> None:
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
                    int(m.center_misaligned),
                    m.recorded_center_x,
                    m.recorded_center_y,
                    m.recommended_center_px[0] if m.recommended_center_px else None,
                    m.recommended_center_px[1] if m.recommended_center_px else None,
                    m.plant_center_px[0] if m.plant_center_px else None,
                    m.plant_center_px[1] if m.plant_center_px else None,
                    m.visible_fraction,
                    m.source_image_path,
                    m.composite_path,
                    m.composite_overlay_path,
                    int(m.fused_canopy),
                    m.fused_typical_radius_mm,
                    m.fused_maximum_radius_mm,
                    m.fused_recommended_radius_mm,
                    m.fused_confidence,
                    m.fusion_view_count,
                    m.fusion_angular_coverage,
                    m.fusion_corroborated_fraction,
                    m.fusion_disagreement_mm,
                    int(m.fusion_reliable),
                    m.fusion_diagnostic_path,
                    int(m.center_visible),
                    m.boundary_coverage,
                    json.dumps(m.boundary_sectors, separators=(",", ":")),
                    int(m.canopy_truncated),
                    int(m.has_plant_evidence),
                    int(m.plant_fits_single_frame),
                    m.image_quality,
                    m.segmentation_quality,
                    m.evidence_status,
                    m.exclusion_reason,
                    m.diagnostics_json,
                )
            )
        with self.connection:
            # Re-analysis replaces the reviewable result for this plant/image
            # while retaining the prior measurement and audit trail.
            for measurement in measurements:
                prior_rows = self.connection.execute(
                    "SELECT m.measurement_id FROM measurements m WHERE m.config_entry_id=? "
                    "AND m.plant_id=? AND m.image_id=? AND m.measurement_id<>? AND NOT EXISTS "
                    "(SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id "
                    "AND d.action='superseded')",
                    (
                        measurement.config_entry_id,
                        measurement.plant_id,
                        measurement.image_id,
                        str(measurement.measurement_id),
                    ),
                ).fetchall()
                self.connection.executemany(
                    "INSERT INTO decisions(measurement_id,action,details_json) VALUES(?,"
                    "'superseded','{}')",
                    [(row[0],) for row in prior_rows],
                )
            self.connection.executemany(
                """INSERT OR REPLACE INTO measurements(measurement_id,config_entry_id,plant_id,crop_slug,plant_age_days,
                image_id,image_timestamp,current_radius_mm,typical_canopy_radius_mm,
                maximum_accepted_canopy_radius_mm,recommended_protection_radius_mm,confidence,
                calibration_version_id,transform_json,algorithm_version,decision,reason,ambiguous,applied,
                mask_path,overlay_path,analysis_resolution,processed_width,processed_height,
                calibration_source,calibrated,contract_version,artifact_paths_json,
                vegetation_absent,absent_observations,safety_margin_mm,calibration_uncertainty_mm,
                center_misaligned,recorded_center_x,recorded_center_y,
                recommended_center_x,recommended_center_y,plant_center_px_x,plant_center_px_y,
                visible_fraction,source_image_path,composite_path,composite_overlay_path,
                fused_canopy,fused_typical_radius_mm,fused_maximum_radius_mm,
                fused_recommended_radius_mm,fused_confidence,fusion_view_count,
                fusion_angular_coverage,fusion_corroborated_fraction,fusion_disagreement_mm,
                fusion_reliable,fusion_diagnostic_path,center_visible,boundary_coverage,
                boundary_sectors_json,canopy_truncated,has_plant_evidence,
                plant_fits_single_frame,image_quality,segmentation_quality,evidence_status,
                exclusion_reason,diagnostics_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )

    def save_weed_detection(
        self,
        *,
        detection_id: str,
        config_entry_id: str,
        image_id: int,
        image_timestamp: datetime,
        x: float,
        y: float,
        z: float,
        area_mm2: float,
        radius_mm: float,
        confidence: float,
        overlay_path: str | None,
        review_path: str | None = None,
        status: str = "recommended",
        center_px_x: float | None = None,
        center_px_y: float | None = None,
        processed_width: int | None = None,
        processed_height: int | None = None,
        heuristic_confidence: float | None = None,
        verifier_confidence: float | None = None,
        features: dict[str, float] | None = None,
        crop_path: str | None = None,
        observation_count: int = 1,
        candidate_track_id: int | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO weed_detections(
                detection_id,config_entry_id,image_id,image_timestamp,x,y,z,area_mm2,
                radius_mm,confidence,overlay_path,review_path,status,center_px_x,center_px_y,
                processed_width,processed_height,heuristic_confidence,verifier_confidence,
                features_json,crop_path,observation_count,candidate_track_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    detection_id,
                    config_entry_id,
                    image_id,
                    image_timestamp.isoformat(),
                    x,
                    y,
                    z,
                    area_mm2,
                    radius_mm,
                    confidence,
                    overlay_path,
                    review_path,
                    status,
                    center_px_x,
                    center_px_y,
                    processed_width,
                    processed_height,
                    heuristic_confidence,
                    verifier_confidence,
                    json.dumps(features or {}, separators=(",", ":")),
                    crop_path,
                    observation_count,
                    candidate_track_id,
                ),
            )

    def observe_weed_candidate(
        self,
        *,
        config_entry_id: str,
        image_id: int,
        seen_at: datetime,
        x: float,
        y: float,
        confidence: float,
        match_distance_mm: float,
        max_gap_hours: int,
    ) -> dict:
        cutoff = seen_at.timestamp() - max_gap_hours * 3600
        matches = []
        for row in self.connection.execute(
            """SELECT * FROM weed_candidate_tracks
            WHERE config_entry_id=? AND status='active'""",
            (config_entry_id,),
        ):
            try:
                last_seen = datetime.fromisoformat(row["last_seen_at"]).timestamp()
            except (TypeError, ValueError):
                continue
            distance = math.hypot(float(row["x"]) - x, float(row["y"]) - y)
            if last_seen >= cutoff and distance <= match_distance_mm:
                matches.append((distance, row))
        if not matches:
            with self.connection:
                cursor = self.connection.execute(
                    """INSERT INTO weed_candidate_tracks(
                    config_entry_id,x,y,observations,first_seen_at,last_seen_at,last_image_id,
                    mean_confidence,status) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        config_entry_id,
                        x,
                        y,
                        1,
                        seen_at.isoformat(),
                        seen_at.isoformat(),
                        image_id,
                        confidence,
                        "active",
                    ),
                )
            return {
                "id": int(cursor.lastrowid),
                "observations": 1,
                "x": x,
                "y": y,
                "mean_confidence": confidence,
            }
        _, row = min(matches, key=lambda item: item[0])
        observations = int(row["observations"])
        if int(row["last_image_id"]) == image_id:
            return dict(row)
        next_count = observations + 1
        next_x = (float(row["x"]) * observations + x) / next_count
        next_y = (float(row["y"]) * observations + y) / next_count
        next_confidence = (float(row["mean_confidence"]) * observations + confidence) / next_count
        with self.connection:
            self.connection.execute(
                """UPDATE weed_candidate_tracks SET x=?,y=?,observations=?,last_seen_at=?,
                last_image_id=?,mean_confidence=? WHERE id=?""",
                (
                    next_x,
                    next_y,
                    next_count,
                    seen_at.isoformat(),
                    image_id,
                    next_confidence,
                    row["id"],
                ),
            )
        return {
            "id": int(row["id"]),
            "observations": next_count,
            "x": next_x,
            "y": next_y,
            "mean_confidence": next_confidence,
        }

    def supersede_pending_weed_detections(
        self, config_entry_id: str, x: float, y: float, tolerance_mm: float
    ) -> None:
        ids = [
            row["detection_id"]
            for row in self.connection.execute(
                """SELECT detection_id,x,y FROM weed_detections
                WHERE config_entry_id=? AND status IN ('recommended','observing')""",
                (config_entry_id,),
            )
            if math.hypot(float(row["x"]) - x, float(row["y"]) - y) <= tolerance_mm
        ]
        with self.connection:
            self.connection.executemany(
                "UPDATE weed_detections SET status='superseded' WHERE detection_id=?",
                ((detection_id,) for detection_id in ids),
            )

    def has_terminal_weed_detection_near(
        self,
        config_entry_id: str,
        x: float,
        y: float,
        tolerance_mm: float,
        *,
        source_image_id: int | None = None,
        source_image_timestamp: datetime | None = None,
    ) -> bool:
        for row in self.connection.execute(
            """SELECT x,y,radius_mm,status,image_id,image_timestamp FROM weed_detections
            WHERE config_entry_id=? AND status IN ('created','rejected','dismissed')""",
            (config_entry_id,),
        ):
            # A rejection judges one photograph, not this soil coordinate
            # forever. Suppress reruns of that same source image so the reviewed
            # item does not immediately return, but let later photos discover a
            # newly emerged weed or correct an earlier mistake.
            if row[3] in ("rejected", "dismissed") and (
                source_image_id is None or int(row[4]) != source_image_id
            ):
                continue
            if row[3] == "created" and source_image_timestamp is not None:
                created_at = datetime.fromisoformat(str(row[5]))
                age_hours = (source_image_timestamp - created_at).total_seconds() / 3600
                if age_hours > CREATED_WEED_SYNC_GUARD_HOURS:
                    continue
            rejected_radius = float(row[2] or 0) * 1.5 if row[3] in ("rejected", "dismissed") else 0
            if math.hypot(float(row[0]) - x, float(row[1]) - y) <= max(
                tolerance_mm, 20.0, rejected_radius
            ):
                return True
        return False

    def label_weed_detection(self, detection_id: str, label: str) -> bool:
        target = self.connection.execute(
            """SELECT detection_id,features_json,crop_path FROM weed_detections
            WHERE detection_id=?""",
            (detection_id,),
        ).fetchone()
        if target is None:
            return False
        with self.connection:
            self.connection.execute(
                """INSERT INTO weed_training_samples(detection_id,label,features_json,crop_path)
                VALUES(?,?,?,?) ON CONFLICT(detection_id) DO UPDATE SET
                label=excluded.label,features_json=excluded.features_json,
                crop_path=excluded.crop_path,created_at=CURRENT_TIMESTAMP""",
                (detection_id, label, target["features_json"], target["crop_path"]),
            )
        return True

    def update_weed_training_sample_label(self, detection_id: str, label: str) -> bool:
        """Change the label of an existing verifier sample without changing review state."""
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE weed_training_samples SET label=?,created_at=CURRENT_TIMESTAMP "
                "WHERE detection_id=?",
                (label, detection_id),
            )
        return cursor.rowcount > 0

    def import_weed_training_samples(self, samples: list[tuple[str, str, dict]]) -> int:
        """Insert labelled samples from an exported bundle.

        No crop path is stored: the imported detection's crop belongs to the
        install that produced it, and a dangling path would break the review
        thumbnails. The features are what training needs.
        """
        with self.connection:
            self.connection.executemany(
                """INSERT INTO weed_training_samples(detection_id,label,features_json,crop_path)
                VALUES(?,?,?,NULL) ON CONFLICT(detection_id) DO UPDATE SET
                label=excluded.label,features_json=excluded.features_json,
                created_at=CURRENT_TIMESTAMP""",
                [
                    (detection_id, label, json.dumps(features, separators=(",", ":")))
                    for detection_id, label, features in samples
                ],
            )
        return len(samples)

    def clear_weed_training_samples(self) -> list[str]:
        """Remove all verifier samples and return their stored crop paths."""
        paths = [
            str(row["crop_path"])
            for row in self.connection.execute(
                "SELECT crop_path FROM weed_training_samples WHERE crop_path IS NOT NULL"
            )
        ]
        with self.connection:
            self.connection.execute("DELETE FROM weed_training_samples")
        return paths

    def weed_training_samples(self) -> list[dict]:
        # The detection carries the image the sample was cut from. Training
        # holds whole images out of validation, so a sample without a surviving
        # detection row simply gets a null image and forms its own group.
        samples = []
        for row in self.connection.execute(
            """SELECT s.*,d.image_id AS image_id FROM weed_training_samples s
            LEFT JOIN weed_detections d ON d.detection_id=s.detection_id
            ORDER BY s.created_at,s.detection_id"""
        ):
            item = dict(row)
            try:
                item["features"] = json.loads(item.pop("features_json"))
            except (TypeError, json.JSONDecodeError):
                item["features"] = {}
            samples.append(item)
        return samples

    def unlabelled_weed_detections(self, limit: int = 200) -> list[dict]:
        """Detections that have been scored but never labelled by the reviewer.

        Ordered arbitrarily here; the caller ranks them by how close they sit
        to the decision boundary, because a label near the boundary teaches the
        verifier far more than another obvious weed.
        """
        return [
            dict(row)
            for row in self.connection.execute(
                """SELECT d.* FROM weed_detections d
                WHERE NOT EXISTS (
                  SELECT 1 FROM weed_training_samples s WHERE s.detection_id=d.detection_id
                )
                AND d.crop_path IS NOT NULL
                ORDER BY d.image_timestamp DESC LIMIT ?""",
                (limit,),
            )
        ]

    def weed_training_summary(self) -> dict[str, int]:
        summary = {
            "weed": 0,
            "crop": 0,
            "fallen_leaf": 0,
            "mushroom": 0,
            "moss": 0,
            "soil": 0,
            "hardware": 0,
            "mulch_soil": 0,
            "fungus_moss": 0,
            "hardware_other": 0,
        }
        for row in self.connection.execute(
            "SELECT label,COUNT(*) AS count FROM weed_training_samples GROUP BY label"
        ):
            summary[str(row["label"])] = int(row["count"])
        return summary

    def weed_labels_since_last_model_run(self) -> int:
        """New/edited labels since the last trained run, for the label-count retrain trigger.

        Compares against total sample count rather than timestamps: the run's
        created_at is an ISO-8601 string while the samples' created_at is
        SQLite's CURRENT_TIMESTAMP format, so the two are not safely
        comparable as strings.
        """
        total = self.connection.execute(
            "SELECT COUNT(*) AS count FROM weed_training_samples"
        ).fetchone()["count"]
        last_run = self.connection.execute(
            "SELECT sample_count FROM weed_model_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        baseline = int(last_run["sample_count"]) if last_run is not None else 0
        return max(0, int(total) - baseline)

    def record_weed_model_run(self, model: dict) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO weed_model_runs(
                created_at,sample_count,positive_count,negative_count,metrics_json)
                VALUES(?,?,?,?,?)""",
                (
                    model["created_at"],
                    model["sample_count"],
                    model["positive_count"],
                    model["negative_count"],
                    json.dumps(model["metrics"], separators=(",", ":")),
                ),
            )

    def pending_weed_detections(self, limit: int = 100) -> list[dict]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM weed_detections WHERE status IN ('recommended','observing') "
                "ORDER BY image_timestamp DESC LIMIT ?",
                (limit,),
            )
        ]

    def clear_pending_measurements(self) -> int:
        """Mark every pending plant measurement as superseded.

        Keep the measurement and its audit history so clearing the review queue
        does not remove diagnostic evidence or make a later re-analysis appear
        to have overwritten an older result.
        """
        rows = self.connection.execute(
            """SELECT m.measurement_id FROM measurements m
            WHERE NOT EXISTS (
              SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id
              AND d.action IN
                ('applied','approved_no_change','reject','removed','keep','superseded')
            )"""
        ).fetchall()
        if not rows:
            return 0
        details = json.dumps(
            {"reason": "cleared from Analysis review queue"}, separators=(",", ":")
        )
        with self.connection:
            self.connection.executemany(
                "INSERT INTO decisions(measurement_id,action,details_json) VALUES(?,?,?)",
                [(row[0], "superseded", details) for row in rows],
            )
        return len(rows)

    def clear_pending_weed_detections(self) -> int:
        """Mark pending weed recommendations as superseded without deleting them."""
        detection_ids = [
            row["detection_id"]
            for row in self.connection.execute(
                "SELECT detection_id FROM weed_detections "
                "WHERE status IN ('recommended','observing')"
            )
        ]
        if not detection_ids:
            return 0
        with self.connection:
            self.connection.executemany(
                "UPDATE weed_detections SET status='superseded' WHERE detection_id=?",
                ((detection_id,) for detection_id in detection_ids),
            )
        return len(detection_ids)

    def clear_flagged_curve_proposals(self) -> int:
        """Discard curve recommendations that are still awaiting review."""
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE curve_proposals SET status='rejected' WHERE status='flagged'"
            )
        return cursor.rowcount

    def weed_detection(self, detection_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM weed_detections WHERE detection_id=?", (detection_id,)
        ).fetchone()
        return None if row is None else dict(row)

    def update_weed_detection(self, detection_id: str, status: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE weed_detections SET status=? WHERE detection_id=?",
                (status, detection_id),
            )

    def reject_weed_detection(self, detection_id: str, tolerance_mm: float) -> bool:
        """Reject matching views while allowing later photos to reassess the position.

        Detection UUIDs change on every analysis. Keep the rejected row as a
        coordinate-based marker for this review batch and also clear any
        already-pending duplicates at the same position.
        """
        target = self.connection.execute(
            "SELECT config_entry_id,x,y,candidate_track_id FROM weed_detections WHERE detection_id=?",
            (detection_id,),
        ).fetchone()
        if target is None:
            return False
        with self.connection:
            nearby_ids = [
                row[0]
                for row in self.connection.execute(
                    """SELECT detection_id,x,y FROM weed_detections
                    WHERE config_entry_id=? AND status IN ('recommended','observing','rejected')""",
                    (target["config_entry_id"],),
                )
                if math.hypot(
                    float(row[1]) - float(target["x"]),
                    float(row[2]) - float(target["y"]),
                )
                <= tolerance_mm
            ]
            self.connection.executemany(
                "UPDATE weed_detections SET status='rejected' WHERE detection_id=?",
                ((nearby_id,) for nearby_id in nearby_ids),
            )
            if target["candidate_track_id"] is not None:
                self.connection.execute(
                    "UPDATE weed_candidate_tracks SET status='rejected' WHERE id=?",
                    (target["candidate_track_id"],),
                )
        return True

    def dismiss_weed_detection(self, detection_id: str, tolerance_mm: float) -> bool:
        """Discard this ambiguous view without judging later photos of the position."""
        target = self.connection.execute(
            "SELECT config_entry_id,x,y,candidate_track_id FROM weed_detections WHERE detection_id=?",
            (detection_id,),
        ).fetchone()
        if target is None:
            return False
        with self.connection:
            nearby_ids = [
                row[0]
                for row in self.connection.execute(
                    """SELECT detection_id,x,y FROM weed_detections
                    WHERE config_entry_id=? AND status IN ('recommended','observing','dismissed')""",
                    (target["config_entry_id"],),
                )
                if math.hypot(
                    float(row[1]) - float(target["x"]),
                    float(row[2]) - float(target["y"]),
                )
                <= tolerance_mm
            ]
            self.connection.executemany(
                "UPDATE weed_detections SET status='dismissed' WHERE detection_id=?",
                ((nearby_id,) for nearby_id in nearby_ids),
            )
            if target["candidate_track_id"] is not None:
                self.connection.execute(
                    "UPDATE weed_candidate_tracks SET status='dismissed' WHERE id=?",
                    (target["candidate_track_id"],),
                )
        return True

    def upsert_weed_track(
        self,
        *,
        config_entry_id: str,
        weed_id: int,
        x: float,
        y: float,
        radius_mm: float,
        confidence: float,
        seen_at: datetime | None,
        status: str = "active",
        absent_observations: int = 0,
        present_observations: int = 0,
        observation_image_id: int | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO weed_tracks(
                config_entry_id,weed_id,x,y,radius_mm,last_seen_at,absent_observations,
                confidence,status,present_observations,last_observation_image_id,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(config_entry_id,weed_id) DO UPDATE SET
                x=excluded.x,y=excluded.y,radius_mm=excluded.radius_mm,
                last_seen_at=COALESCE(excluded.last_seen_at,weed_tracks.last_seen_at),
                absent_observations=excluded.absent_observations,
                present_observations=excluded.present_observations,
                last_observation_image_id=excluded.last_observation_image_id,
                confidence=excluded.confidence,status=excluded.status,
                updated_at=CURRENT_TIMESTAMP""",
                (
                    config_entry_id,
                    weed_id,
                    x,
                    y,
                    radius_mm,
                    seen_at.isoformat() if seen_at else None,
                    absent_observations,
                    confidence,
                    status,
                    present_observations,
                    observation_image_id,
                ),
            )

    def weed_track(self, config_entry_id: str, weed_id: int) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM weed_tracks WHERE config_entry_id=? AND weed_id=?",
            (config_entry_id, weed_id),
        ).fetchone()
        return None if row is None else dict(row)

    def has_known_weed_observation(self, config_entry_id: str, weed_id: int, image_id: int) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM known_weed_observations "
                "WHERE config_entry_id=? AND weed_id=? AND image_id=?",
                (config_entry_id, weed_id, image_id),
            ).fetchone()
            is not None
        )

    def known_weed_observation(
        self, config_entry_id: str, weed_id: int, image_id: int
    ) -> dict | None:
        row = self.connection.execute(
            "SELECT status,confidence FROM known_weed_observations "
            "WHERE config_entry_id=? AND weed_id=? AND image_id=?",
            (config_entry_id, weed_id, image_id),
        ).fetchone()
        return None if row is None else dict(row)

    def record_known_weed_observation(
        self,
        config_entry_id: str,
        weed_id: int,
        image_id: int,
        status: str,
        confidence: float,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO known_weed_observations("
                "config_entry_id,weed_id,image_id,status,confidence) VALUES(?,?,?,?,?)",
                (config_entry_id, weed_id, image_id, status, confidence),
            )

    def has_weed_detection_near(
        self,
        config_entry_id: str,
        x: float,
        y: float,
        tolerance_mm: float,
        *,
        source_image_id: int | None = None,
        source_image_timestamp: datetime | None = None,
    ) -> bool:
        for row in self.connection.execute(
            """SELECT x,y,radius_mm,status,image_id,image_timestamp FROM weed_detections
            WHERE config_entry_id=? AND status IN ('recommended','created','rejected')""",
            (config_entry_id,),
        ):
            if row[3] == "rejected" and (source_image_id is None or int(row[4]) != source_image_id):
                continue
            if row[3] == "created" and source_image_timestamp is not None:
                created_at = datetime.fromisoformat(str(row[5]))
                age_hours = (source_image_timestamp - created_at).total_seconds() / 3600
                if age_hours > CREATED_WEED_SYNC_GUARD_HOURS:
                    continue
            rejected_radius = float(row[2] or 0) * 1.5 if row[3] == "rejected" else 0
            if math.hypot(float(row[0]) - x, float(row[1]) - y) <= max(
                tolerance_mm, 20.0, rejected_radius
            ):
                return True
        return False

    def current_vision_weeds(self, config_entry_id: str) -> list[dict]:
        return [
            dict(row)
            for row in self.connection.execute(
                """SELECT x,y,radius_mm FROM weed_detections
                WHERE config_entry_id=? AND status IN ('recommended','created')""",
                (config_entry_id,),
            )
        ]

    def current_vision_plants(self, config_entry_id: str) -> list[dict]:
        """Return the newest positive canopy detection for each plant."""
        return [
            dict(row)
            for row in self.connection.execute(
                """SELECT m.recorded_center_x AS x,m.recorded_center_y AS y,
                MAX(m.current_radius_mm,m.maximum_accepted_canopy_radius_mm,
                    m.recommended_protection_radius_mm) AS radius_mm
                FROM measurements m
                WHERE m.config_entry_id=? AND m.vegetation_absent=0
                  AND m.recorded_center_x IS NOT NULL
                  AND m.recorded_center_y IS NOT NULL
                  AND m.image_timestamp=(
                    SELECT MAX(newest.image_timestamp) FROM measurements newest
                    WHERE newest.config_entry_id=m.config_entry_id
                      AND newest.plant_id=m.plant_id
                  )""",
                (config_entry_id,),
            )
        ]

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
            try:
                row["boundary_sectors"] = json.loads(row.get("boundary_sectors_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                row["boundary_sectors"] = []
        return result

    @staticmethod
    def _weighted_median(values: list[tuple[float, float]]) -> float:
        ordered = sorted(values, key=lambda item: item[0])
        total = sum(weight for _, weight in ordered)
        if total <= 0:
            return ordered[-1][0]
        threshold = total / 2
        cumulative = 0.0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= threshold:
                return value
        return ordered[-1][0]

    @classmethod
    def _consolidate_measurement_rows(cls, rows: list[dict]) -> dict:
        """Return one robust judgement for repeated views of the same plant.

        Confidence, the fraction of the canopy visible in each frame, and
        recency all contribute to the weight. A weighted median prevents one
        distant false-positive leaf from dominating the recommended radius.
        """
        evidence = select_measurement_evidence(rows)
        candidate_rows = sorted(rows, key=lambda row: str(row["image_timestamp"]), reverse=True)
        diagnostics = selection_diagnostics(evidence)
        if evidence.used:
            ordered = [dict(row) for row in evidence.used]
        else:
            absence_rows = [
                dict(row)
                for row in candidate_rows
                if row.get("vegetation_absent") and row.get("center_visible", 1)
            ]
            if absence_rows:
                ordered = absence_rows
            else:
                representative = dict(candidate_rows[0])
                representative.update(diagnostics)
                representative["confidence"] = min(
                    0.2, float(representative.get("confidence") or 0.05)
                )
                representative["decision"] = "uncertain"
                representative["vegetation_absent"] = 0
                representative["reason"] = (
                    f"No usable target-plant evidence in {len(candidate_rows)} candidate image"
                    f"{'s' if len(candidate_rows) != 1 else ''}; "
                    "excluded images do not contribute to confidence"
                )
                representative["source_measurement_ids"] = [
                    str(row["measurement_id"]) for row in candidate_rows
                ]
                representative["measurement_count"] = len(candidate_rows)
                representative["useful_measurement_count"] = len(evidence.useful)
                representative["used_measurement_count"] = 0
                return representative
        if len(ordered) == 1:
            row = dict(ordered[0])
            legacy_fused = next(
                (
                    candidate
                    for candidate in candidate_rows
                    if candidate.get("fused_canopy")
                    and candidate.get("fusion_reliable")
                    and candidate.get("fused_recommended_radius_mm") is not None
                ),
                None,
            )
            if legacy_fused is not None:
                row["typical_canopy_radius_mm"] = float(legacy_fused["fused_typical_radius_mm"])
                row["maximum_accepted_canopy_radius_mm"] = float(
                    legacy_fused["fused_maximum_radius_mm"]
                )
                row["recommended_protection_radius_mm"] = float(
                    legacy_fused["fused_recommended_radius_mm"]
                )
                row["confidence"] = float(legacy_fused["fused_confidence"])
                for field in (
                    "fused_canopy",
                    "fused_typical_radius_mm",
                    "fused_maximum_radius_mm",
                    "fused_recommended_radius_mm",
                    "fused_confidence",
                    "fusion_view_count",
                    "fusion_angular_coverage",
                    "fusion_corroborated_fraction",
                    "fusion_disagreement_mm",
                    "fusion_reliable",
                    "fusion_diagnostic_path",
                ):
                    row[field] = legacy_fused.get(field)
            row.update(diagnostics)
            row["source_measurement_ids"] = [
                str(candidate["measurement_id"]) for candidate in candidate_rows
            ]
            row["measurement_count"] = len(candidate_rows)
            row["useful_measurement_count"] = len(evidence.useful)
            row["used_measurement_count"] = 1
            if evidence.mode == "single_complete" and len(candidate_rows) > 1:
                row["reason"] = (
                    f"Selected complete image #{row['image_id']} alone from "
                    f"{len(candidate_rows)} candidates; excluded images do not reduce confidence"
                )
            return row
        newest = datetime.fromisoformat(str(ordered[0]["image_timestamp"]))
        weights: list[float] = []
        for row in ordered:
            timestamp = datetime.fromisoformat(str(row["image_timestamp"]))
            age_days = max(0.0, (newest - timestamp).total_seconds() / 86_400)
            recency = math.exp(-math.log(2) * age_days / 14.0)
            visible = max(0.05, min(1.0, float(row.get("visible_fraction") or 0)))
            confidence = max(0.05, min(1.0, float(row.get("confidence") or 0)))
            weights.append(confidence * visible * recency)

        present_weight = sum(
            weight
            for row, weight in zip(ordered, weights, strict=True)
            if not row.get("vegetation_absent")
        )
        absent_weight = sum(
            weight
            for row, weight in zip(ordered, weights, strict=True)
            if row.get("vegetation_absent")
        )
        use_absent = absent_weight > present_weight
        selected = [
            (row, weight)
            for row, weight in zip(ordered, weights, strict=True)
            if bool(row.get("vegetation_absent")) == use_absent
        ] or list(zip(ordered, weights, strict=True))
        # Keep the newest measurement ID as the actionable representative,
        # while the values below come from the selected evidence class.
        representative = dict(ordered[0])
        selected_total = sum(weight for _, weight in selected)
        representative["confidence"] = min(
            0.99,
            sum(float(row["confidence"]) * weight for row, weight in selected)
            / max(selected_total, 1e-9)
            + min(0.08, 0.02 * (len(selected) - 1)),
        )
        representative["visible_fraction"] = min(
            1.0,
            sum(float(row.get("visible_fraction") or 0) * weight for row, weight in selected)
            / max(selected_total, 1e-9),
        )
        representative["vegetation_absent"] = int(use_absent)
        if use_absent:
            representative["typical_canopy_radius_mm"] = 0.0
            representative["maximum_accepted_canopy_radius_mm"] = 0.0
            representative["recommended_protection_radius_mm"] = 0.0
            representative["absent_observations"] = max(
                int(row.get("absent_observations") or 0) for row, _ in selected
            )
        else:
            for field in (
                "typical_canopy_radius_mm",
                "maximum_accepted_canopy_radius_mm",
                "recommended_protection_radius_mm",
            ):
                representative[field] = cls._weighted_median(
                    [(float(row[field]), weight) for row, weight in selected]
                )
            fused = next(
                (
                    row
                    for row in ordered
                    if row.get("fused_canopy")
                    and row.get("fused_maximum_radius_mm") is not None
                    and row.get("fused_recommended_radius_mm") is not None
                ),
                None,
            )
            if fused is not None:
                representative["typical_canopy_radius_mm"] = float(fused["fused_typical_radius_mm"])
                representative["maximum_accepted_canopy_radius_mm"] = float(
                    fused["fused_maximum_radius_mm"]
                )
                representative["recommended_protection_radius_mm"] = float(
                    fused["fused_recommended_radius_mm"]
                )
                representative["confidence"] = float(fused["fused_confidence"])
                for field in (
                    "fused_canopy",
                    "fused_typical_radius_mm",
                    "fused_maximum_radius_mm",
                    "fused_recommended_radius_mm",
                    "fused_confidence",
                    "fusion_view_count",
                    "fusion_angular_coverage",
                    "fusion_corroborated_fraction",
                    "fusion_disagreement_mm",
                    "fusion_reliable",
                    "fusion_diagnostic_path",
                ):
                    representative[field] = fused.get(field)
                if not bool(fused.get("fusion_reliable")):
                    representative["decision"] = "uncertain"
        paths: list[str] = []
        representative["composite_path"] = next(
            (row["composite_path"] for row in ordered if row.get("composite_path")),
            None,
        )
        representative["composite_overlay_path"] = next(
            (row["composite_overlay_path"] for row in ordered if row.get("composite_overlay_path")),
            None,
        )
        for row in ordered:
            for path in row.get("artifact_paths") or []:
                if path and path not in paths:
                    paths.append(path)
        representative["artifact_paths"] = paths
        representative.update(diagnostics)
        representative["source_measurement_ids"] = [
            str(row["measurement_id"]) for row in candidate_rows
        ]
        representative["measurement_count"] = len(candidate_rows)
        representative["useful_measurement_count"] = len(evidence.useful)
        representative["used_measurement_count"] = len(evidence.used)
        representative["reason"] = (
            (
                f"Fused plant ownership masks from {representative['fusion_view_count']} "
                f"images; angular coverage "
                f"{float(representative['fusion_angular_coverage']):.0%}, corroborated "
                f"{float(representative['fusion_corroborated_fraction']):.0%}"
            )
            if representative.get("fused_canopy")
            else (
                f"Estimated from {len(evidence.used)} selected image"
                f"{'s' if len(evidence.used) != 1 else ''} out of "
                f"{len(candidate_rows)} candidates ({len(evidence.useful)} contained useful "
                f"target evidence); visible outer-boundary coverage "
                f"{evidence.boundary_coverage:.0%}"
            )
        )
        return representative

    def pending_measurements(
        self,
        limit: int = 100,
        *,
        minimum_confidence: float = 0.0,
        minimum_radius_increase_mm: float = 0.0,
        minimum_radius_reduction_mm: float = 0.0,
    ) -> list[dict]:
        """Return consolidated review rows above the configured rejection floor.

        Low-confidence measurements remain persisted for diagnostics and future
        re-analysis, but callers rendering a review queue can omit them without
        destroying their evidence or manufacturing a human decision.
        """
        rows = self.connection.execute(
            """SELECT m.* FROM measurements m
            WHERE NOT EXISTS (
              SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id
              AND d.action IN
                ('applied','approved_no_change','reject','removed','keep','superseded')
            )
            ORDER BY m.image_timestamp DESC,m.created_at DESC""",
        ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            try:
                row["artifact_paths"] = json.loads(row.get("artifact_paths_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                row["artifact_paths"] = []
            try:
                row["boundary_sectors"] = json.loads(row.get("boundary_sectors_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                row["boundary_sectors"] = []
        groups: dict[tuple[object, object], list[dict]] = {}
        for row in result:
            groups.setdefault((row.get("config_entry_id"), row["plant_id"]), []).append(row)
        consolidated = []
        for group in groups.values():
            # One view per source image. Re-analysis of the same image is not
            # independent evidence and must not add weight.
            by_image: dict[int, dict] = {}
            for row in group:
                by_image.setdefault(int(row["image_id"]), row)
            consolidated.append(self._consolidate_measurement_rows(list(by_image.values())))
        for row in consolidated:
            if row.get("vegetation_absent"):
                continue
            current = float(row.get("current_radius_mm") or 0)
            recommended = float(row.get("recommended_protection_radius_mm") or 0)
            delta = recommended - current
            minimum = minimum_radius_increase_mm if delta > 0 else minimum_radius_reduction_mm
            if delta and abs(delta) < minimum:
                row["confidence"] = 0.05
                row["reason"] = f"radius change is below the configured minimum of {minimum:g} mm"
        reviewable = [
            row for row in consolidated if float(row.get("confidence") or 0) >= minimum_confidence
        ]
        return sorted(reviewable, key=lambda row: str(row["image_timestamp"]), reverse=True)[:limit]

    def pending_plant_measurements(
        self,
        config_entry_id: str,
        plant_id: int,
        image_ids: Iterable[int],
    ) -> list[Measurement]:
        """Return the newest pending result for each requested source image.

        Photo-grid uploads arrive as separate vision events. Reading their
        pending rows back as one set lets a later tile consolidate and stitch
        the current grid instead of behaving like an isolated photo.
        """
        selected_ids = sorted({int(image_id) for image_id in image_ids})
        if not selected_ids:
            return []
        selected_id_set = set(selected_ids)
        rows = self.connection.execute(
            """SELECT m.* FROM measurements m
            WHERE m.config_entry_id=? AND m.plant_id=?
            AND NOT EXISTS (
              SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id
              AND d.action IN
                ('applied','approved_no_change','reject','removed','keep','superseded')
            )
            ORDER BY m.image_timestamp DESC,m.created_at DESC""",
            (config_entry_id, plant_id),
        ).fetchall()
        by_image: dict[int, sqlite3.Row] = {}
        for row in rows:
            image_id = int(row["image_id"])
            if image_id in selected_id_set:
                by_image.setdefault(image_id, row)

        measurements: list[Measurement] = []
        for row in by_image.values():
            item = dict(row)
            try:
                item["artifact_paths"] = json.loads(item.get("artifact_paths_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                item["artifact_paths"] = []
            try:
                item["boundary_sectors"] = json.loads(item.get("boundary_sectors_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                item["boundary_sectors"] = []
            if item.get("recommended_center_px_x") is not None:
                item["recommended_center_px"] = (
                    item["recommended_center_px_x"],
                    item["recommended_center_px_y"],
                )
            if item.get("plant_center_px_x") is not None:
                item["plant_center_px"] = (
                    item["plant_center_px_x"],
                    item["plant_center_px_y"],
                )
            payload = {
                field: item[field]
                for field in Measurement.model_fields
                if field in item and item[field] is not None
            }
            measurements.append(Measurement.model_validate(payload))
        return measurements

    def has_terminal_decision(self, measurement_id: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM decisions WHERE measurement_id=? "
                "AND action IN "
                "('applied','approved_no_change','reject','removed','keep','superseded') LIMIT 1",
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
        if row is None:
            return None
        item = dict(row)
        if self.has_terminal_decision(measurement_id):
            return item
        rows = self.connection.execute(
            """SELECT m.* FROM measurements m WHERE m.config_entry_id=? AND m.plant_id=?
            AND NOT EXISTS (
              SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id
              AND d.action IN
                ('applied','approved_no_change','reject','removed','keep','superseded')
            ) ORDER BY m.image_timestamp DESC""",
            (item.get("config_entry_id"), item["plant_id"]),
        ).fetchall()
        result = [dict(candidate) for candidate in rows]
        if not result or str(result[0]["measurement_id"]) != measurement_id:
            return item
        by_image: dict[int, dict] = {}
        for candidate in result:
            by_image.setdefault(int(candidate["image_id"]), candidate)
        result = list(by_image.values())
        for candidate in result:
            try:
                candidate["artifact_paths"] = json.loads(
                    candidate.get("artifact_paths_json") or "[]"
                )
            except (TypeError, json.JSONDecodeError):
                candidate["artifact_paths"] = []
            try:
                candidate["boundary_sectors"] = json.loads(
                    candidate.get("boundary_sectors_json") or "[]"
                )
            except (TypeError, json.JSONDecodeError):
                candidate["boundary_sectors"] = []
        return self._consolidate_measurement_rows(result) if result else item

    def record_group_decision(self, measurement_id: str, action: str, details: dict) -> None:
        """Apply a terminal review decision to every pending view in the group."""
        row = self.connection.execute(
            "SELECT config_entry_id,plant_id FROM measurements WHERE measurement_id=?",
            (measurement_id,),
        ).fetchone()
        if row is None:
            return
        candidates = self.connection.execute(
            """SELECT m.measurement_id FROM measurements m
            WHERE m.config_entry_id=? AND m.plant_id=? AND NOT EXISTS (
              SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id
              AND d.action IN
                ('applied','approved_no_change','reject','removed','keep','superseded')
            )""",
            (row[0], row[1]),
        ).fetchall()
        candidate_rows = list(candidates)
        image_ids = {
            candidate[1]
            for candidate in self.connection.execute(
                """SELECT m.measurement_id,m.image_id FROM measurements m
                WHERE m.config_entry_id=? AND m.plant_id=? AND NOT EXISTS (
                  SELECT 1 FROM decisions d WHERE d.measurement_id=m.measurement_id
                  AND d.action IN
                    ('applied','approved_no_change','reject','removed','keep','superseded')
                )""",
                (row[0], row[1]),
            )
        }
        if len(image_ids) <= 1:
            candidate_rows = [(measurement_id,)]
        payload = json.dumps(details, separators=(",", ":"))
        with self.connection:
            self.connection.executemany(
                "INSERT INTO decisions(measurement_id,action,details_json) VALUES(?,?,?)",
                [(candidate[0], action, payload) for candidate in candidate_rows],
            )
        # A rejected recommendation ("reject") or a kept plant that was
        # flagged for removal ("keep") means the image(s) behind it are not
        # valid evidence for this plant going forward -- unlink them from the
        # plant without touching the image file itself (other plants, and the
        # whole-garden photo-grid mosaic, may still legitimately reference it).
        if action in ("reject", "keep") and image_ids:
            self.unlink_plant_images(row[0], row[1], image_ids, measurement_id)

    def unlink_plant_images(
        self,
        config_entry_id: str | None,
        plant_id: int,
        image_ids: Iterable[int],
        measurement_id: str | None = None,
    ) -> None:
        """Record that ``image_ids`` must no longer be shown as this plant's evidence."""
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO plant_image_unlinks"
                "(config_entry_id,plant_id,image_id,measurement_id) VALUES(?,?,?,?)",
                [
                    (config_entry_id, plant_id, int(image_id), measurement_id)
                    for image_id in image_ids
                ],
            )

    def unlinked_image_ids(self, config_entry_id: str | None, plant_id: int) -> set[int]:
        rows = self.connection.execute(
            "SELECT image_id FROM plant_image_unlinks WHERE config_entry_id IS ? AND plant_id=?",
            (config_entry_id, plant_id),
        ).fetchall()
        return {int(row[0]) for row in rows}

    def set_composite_path(
        self,
        measurement_ids: list[str],
        path: str,
        overlay_path: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.executemany(
                "UPDATE measurements SET composite_path=?,composite_overlay_path=? "
                "WHERE measurement_id=?",
                [(path, overlay_path, measurement_id) for measurement_id in measurement_ids],
            )

    def set_evidence_selection(self, measurements: list[object]) -> dict[str, object]:
        """Persist candidate/useful/used state and return structured diagnostics."""

        selection = select_measurement_evidence(measurements)
        used = {str(item.measurement_id) for item in selection.used}
        useful = {str(item.measurement_id) for item in selection.useful}
        excluded = {str(item.measurement_id): reason for item, reason in selection.excluded}
        updates = []
        for item in selection.candidates:
            item_id = str(item.measurement_id)
            if item_id in used:
                status, reason = "used", None
            elif item_id in useful:
                status = "excluded"
                reason = excluded.get(item_id, "useful evidence was not needed")
            else:
                status = "excluded"
                reason = excluded.get(item_id, item.exclusion_reason)
            updates.append((status, reason, item_id))
        with self.connection:
            self.connection.executemany(
                "UPDATE measurements SET evidence_status=?,exclusion_reason=? "
                "WHERE measurement_id=?",
                updates,
            )
        return selection_diagnostics(selection)

    def set_fused_canopy(self, measurement_ids: list[str], values: dict) -> None:
        with self.connection:
            self.connection.executemany(
                """UPDATE measurements SET fused_canopy=1,fused_typical_radius_mm=?,
                fused_maximum_radius_mm=?,fused_recommended_radius_mm=?,
                fused_confidence=?,fusion_view_count=?,fusion_angular_coverage=?,
                fusion_corroborated_fraction=?,fusion_disagreement_mm=?,
                fusion_reliable=?,fusion_diagnostic_path=? WHERE measurement_id=?""",
                [
                    (
                        values["fused_typical_radius_mm"],
                        values["fused_maximum_radius_mm"],
                        values["fused_recommended_radius_mm"],
                        values["fused_confidence"],
                        values["fusion_view_count"],
                        values["fusion_angular_coverage"],
                        values["fusion_corroborated_fraction"],
                        values["fusion_disagreement_mm"],
                        int(values["fusion_reliable"]),
                        values.get("fusion_diagnostic_path"),
                        measurement_id,
                    )
                    for measurement_id in measurement_ids
                ],
            )

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

    def update_curve_proposal(
        self, proposal_id: int, status: str, data: dict | None = None
    ) -> None:
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
        self, measurement_id: str, *, decision: str, applied: bool, reason: str | None = None
    ) -> None:
        with self.connection:
            if reason is None:
                self.connection.execute(
                    "UPDATE measurements SET decision=?,applied=? WHERE measurement_id=?",
                    (decision, int(applied), measurement_id),
                )
            else:
                self.connection.execute(
                    "UPDATE measurements SET decision=?,applied=?,reason=? WHERE measurement_id=?",
                    (decision, int(applied), reason, measurement_id),
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

    def record_change(
        self,
        *,
        entity_type: str,
        entity_id: object,
        crop_type: str,
        x: float | None,
        y: float | None,
        z: float | None,
        change_type: str,
        original_radius_mm: float,
        current_radius_mm: float,
        decision_method: str,
        confidence: float,
        details: dict | None = None,
    ) -> None:
        """Record an applied plant/weed mutation for the dashboard change log."""
        with self.connection:
            self.connection.execute(
                """INSERT INTO change_log(
                entity_type,entity_id,crop_type,x,y,z,change_type,
                original_radius_mm,current_radius_mm,decision_method,confidence,
                details_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entity_type,
                    str(entity_id),
                    crop_type,
                    x,
                    y,
                    z,
                    change_type,
                    float(original_radius_mm),
                    float(current_radius_mm),
                    decision_method,
                    float(confidence),
                    json.dumps(details or {}, separators=(",", ":")),
                ),
            )

    def radius_growth_baseline(
        self,
        *,
        entity_type: str,
        entity_id: object,
        since: datetime,
        current_radius_mm: float,
    ) -> float:
        """Return the radius before the first increase inside a rolling window."""

        row = self.connection.execute(
            """SELECT original_radius_mm FROM change_log
            WHERE entity_type=? AND entity_id=? AND change_type='radius increased'
              AND datetime(created_at)>=datetime(?)
            ORDER BY datetime(created_at),id LIMIT 1""",
            (entity_type, str(entity_id), since.astimezone(UTC).isoformat()),
        ).fetchone()
        return float(row[0]) if row is not None else float(current_radius_mm)

    def recent_changes(self, limit: int = 100) -> list[dict]:
        """Return applied changes newest first for the dashboard log."""
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM change_log ORDER BY created_at DESC,id DESC LIMIT ?",
                (limit,),
            )
        ]

    def save_soil_calibration(self, calibration: SoilStereoCalibration) -> SoilStereoCalibration:
        with self.connection:
            version = int(
                self.connection.execute(
                    """SELECT COALESCE(MAX(version),0)+1 FROM soil_calibrations
                    WHERE config_entry_id=?""",
                    (calibration.config_entry_id,),
                ).fetchone()[0]
            )
            self.connection.execute(
                "UPDATE soil_calibrations SET active=0 WHERE config_entry_id=?",
                (calibration.config_entry_id,),
            )
            cursor = self.connection.execute(
                """INSERT INTO soil_calibrations(
                config_entry_id,point_id,capture_z,baseline_mm,reference_distance_mm,
                z_direction,inverse_depth_slope,inverse_depth_intercept,residual_mm,
                processed_width,processed_height,source_width,source_height,
                source_image_ids_json,camera_signature,active,created_at,version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    calibration.config_entry_id,
                    calibration.point_id,
                    calibration.capture_z,
                    calibration.baseline_mm,
                    calibration.reference_distance_mm,
                    calibration.z_direction,
                    calibration.inverse_depth_slope,
                    calibration.inverse_depth_intercept,
                    calibration.residual_mm,
                    calibration.processed_width,
                    calibration.processed_height,
                    calibration.source_width,
                    calibration.source_height,
                    json.dumps(calibration.source_image_ids, separators=(",", ":")),
                    calibration.camera_signature,
                    int(calibration.active),
                    calibration.created_at.isoformat(),
                    version,
                ),
            )
        return calibration.model_copy(
            update={"calibration_id": int(cursor.lastrowid), "version": version}
        )

    def active_soil_calibration(self, config_entry_id: str) -> SoilStereoCalibration | None:
        row = self.connection.execute(
            """SELECT * FROM soil_calibrations WHERE config_entry_id=? AND active=1
            ORDER BY created_at DESC LIMIT 1""",
            (config_entry_id,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["calibration_id"] = data.pop("id")
        data["source_image_ids"] = json.loads(data.pop("source_image_ids_json"))
        data["active"] = bool(data["active"])
        return SoilStereoCalibration.model_validate(data)

    def save_soil_measurement(self, measurement: SoilMeasurement) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO soil_measurements(
                measurement_id,config_entry_id,point_id,point_name,expected_x,expected_y,
                old_z_mm,proposed_z_mm,confidence,uncertainty_mm,status,reason,capture_id,
                calibration_id,frame_ids_json,metrics_json,artifact_paths_json,
                algorithm_version,created_at,point_updated_at,capture_x,capture_y,
                relocation_distance_mm)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(measurement.measurement_id),
                    measurement.config_entry_id,
                    measurement.point_id,
                    measurement.point_name,
                    measurement.expected_x,
                    measurement.expected_y,
                    measurement.old_z_mm,
                    measurement.proposed_z_mm,
                    measurement.confidence,
                    measurement.uncertainty_mm,
                    measurement.status,
                    measurement.reason,
                    str(measurement.capture_id) if measurement.capture_id else None,
                    measurement.calibration_id,
                    json.dumps(measurement.frame_ids, separators=(",", ":")),
                    json.dumps(measurement.metrics, separators=(",", ":")),
                    json.dumps(measurement.artifact_paths, separators=(",", ":")),
                    measurement.algorithm_version,
                    measurement.created_at.isoformat(),
                    (
                        measurement.point_updated_at.isoformat()
                        if measurement.point_updated_at
                        else None
                    ),
                    measurement.capture_x,
                    measurement.capture_y,
                    measurement.relocation_distance_mm,
                ),
            )

    @staticmethod
    def _decode_soil_measurement(row: sqlite3.Row | dict) -> dict:
        data = dict(row)
        for source, target in (
            ("frame_ids_json", "frame_ids"),
            ("metrics_json", "metrics"),
            ("artifact_paths_json", "artifact_paths"),
        ):
            try:
                data[target] = json.loads(data.pop(source) or "[]")
            except (TypeError, json.JSONDecodeError):
                data[target] = {} if target == "metrics" else []
        return data

    def soil_measurement(self, measurement_id: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM soil_measurements WHERE measurement_id=?",
            (measurement_id,),
        ).fetchone()
        return None if row is None else self._decode_soil_measurement(row)

    def recent_soil_measurements(
        self, config_entry_id: str | None = None, limit: int = 200
    ) -> list[dict]:
        if config_entry_id:
            rows = self.connection.execute(
                """SELECT * FROM soil_measurements WHERE config_entry_id=?
                ORDER BY created_at DESC LIMIT ?""",
                (config_entry_id, limit),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM soil_measurements ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._decode_soil_measurement(row) for row in rows]

    def update_soil_measurement_status(self, measurement_id: str, status: str, reason: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE soil_measurements SET status=?,reason=? WHERE measurement_id=?",
                (status, reason, measurement_id),
            )

    def record_soil_decision(self, measurement_id: str, action: str, details: dict) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO soil_decisions(measurement_id,action,details_json)
                VALUES(?,?,?)""",
                (
                    measurement_id,
                    action,
                    json.dumps(details, separators=(",", ":")),
                ),
            )

    def start_soil_job(
        self,
        job_id: str,
        config_entry_id: str,
        kind: str,
        point_ids: list[int],
    ) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO soil_jobs(
                id,config_entry_id,kind,status,point_ids_json,started_at,message)
                VALUES(?,?,?,'running',?,?,?)""",
                (
                    job_id,
                    config_entry_id,
                    kind,
                    json.dumps(point_ids, separators=(",", ":")),
                    datetime.now(UTC).isoformat(),
                    "Starting",
                ),
            )

    def update_soil_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        current_point_id: int | None = None,
        completed_count: int | None = None,
        failed_count: int | None = None,
        stop_requested: bool | None = None,
        message: str | None = None,
        complete: bool = False,
    ) -> None:
        updates, values = [], []
        for field, value in (
            ("status", status),
            ("current_point_id", current_point_id),
            ("completed_count", completed_count),
            ("failed_count", failed_count),
            ("stop_requested", int(stop_requested) if stop_requested is not None else None),
            ("message", message),
        ):
            if value is not None:
                updates.append(f"{field}=?")
                values.append(value)
        if complete:
            updates.append("completed_at=?")
            values.append(datetime.now(UTC).isoformat())
        if not updates:
            return
        values.append(job_id)
        with self.connection:
            self.connection.execute(
                f"UPDATE soil_jobs SET {','.join(updates)} WHERE id=?",  # noqa: S608
                values,
            )

    def soil_job(self, job_id: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM soil_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["point_ids"] = json.loads(data.pop("point_ids_json"))
        data["stop_requested"] = bool(data["stop_requested"])
        return data

    def latest_soil_job(self, config_entry_id: str | None = None) -> dict | None:
        if config_entry_id:
            row = self.connection.execute(
                """SELECT * FROM soil_jobs WHERE config_entry_id=?
                ORDER BY started_at DESC LIMIT 1""",
                (config_entry_id,),
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT * FROM soil_jobs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["point_ids"] = json.loads(data.pop("point_ids_json"))
        data["stop_requested"] = bool(data["stop_requested"])
        return data

    def stats(self) -> dict[str, int]:
        return {
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "measurements": self.connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[
                0
            ],
            "soil_measurements": self.connection.execute(
                "SELECT COUNT(*) FROM soil_measurements"
            ).fetchone()[0],
        }
