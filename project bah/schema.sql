-- ============================================================
--  Cloud Removal Framework — PostGIS Database Schema
--  PostgreSQL 16 + PostGIS 3.4
-- ============================================================

-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────
-- 1. SCENES — raw input satellite scenes
-- ─────────────────────────────────────────────

CREATE TABLE scenes (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_name      TEXT        NOT NULL UNIQUE,
    sensor          TEXT        NOT NULL CHECK (sensor IN ('LISS-IV','Resourcesat','Sentinel-2')),
    acquisition_date DATE       NOT NULL,
    cloud_coverage  FLOAT       CHECK (cloud_coverage BETWEEN 0 AND 100),
    aoi             GEOMETRY(Polygon, 4326) NOT NULL,   -- footprint in WGS84
    optical_path    TEXT,       -- GeoTIFF path or S3 URI
    sar_path        TEXT,
    temporal_dir    TEXT,
    resolution_m    FLOAT       DEFAULT 10.0,
    band_count      INT         DEFAULT 4,
    epsg            INT         DEFAULT 32643,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    metadata        JSONB       DEFAULT '{}'
);

-- Spatial index on scene footprint
CREATE INDEX idx_scenes_aoi      ON scenes USING GIST (aoi);
CREATE INDEX idx_scenes_date     ON scenes (acquisition_date);
CREATE INDEX idx_scenes_sensor   ON scenes (sensor);
CREATE INDEX idx_scenes_cloud    ON scenes (cloud_coverage);


-- ─────────────────────────────────────────────
-- 2. JOBS — inference pipeline runs
-- ─────────────────────────────────────────────

CREATE TABLE jobs (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_id        UUID        REFERENCES scenes(id) ON DELETE CASCADE,
    status          TEXT        NOT NULL DEFAULT 'queued'
                                CHECK (status IN ('queued','processing','verifying',
                                                  'complete','failed')),
    progress        INT         DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    stage           TEXT        DEFAULT 'queued',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    duration_secs   FLOAT GENERATED ALWAYS AS (
        EXTRACT(EPOCH FROM (completed_at - started_at))
    ) STORED,
    error_msg       TEXT,
    config_snapshot JSONB       DEFAULT '{}'
);

CREATE INDEX idx_jobs_scene  ON jobs (scene_id);
CREATE INDEX idx_jobs_status ON jobs (status);
CREATE INDEX idx_jobs_date   ON jobs (created_at DESC);


-- ─────────────────────────────────────────────
-- 3. RECONSTRUCTIONS — outputs of each job
-- ─────────────────────────────────────────────

CREATE TABLE reconstructions (
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id              UUID        REFERENCES jobs(id) ON DELETE CASCADE,
    scene_id            UUID        REFERENCES scenes(id),
    reconstructed_path  TEXT        NOT NULL,
    cloud_mask_path     TEXT,
    confidence_path     TEXT,
    uncertainty_path    TEXT,
    report_json_path    TEXT,
    report_pdf_path     TEXT,
    overall_pass        BOOLEAN,
    confidence_mean     FLOAT,
    -- Pixel-level statistics
    cloud_px_count      INT,
    high_conf_px_pct    FLOAT,
    med_conf_px_pct     FLOAT,
    low_conf_px_pct     FLOAT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_recon_job    ON reconstructions (job_id);
CREATE INDEX idx_recon_scene  ON reconstructions (scene_id);
CREATE INDEX idx_recon_pass   ON reconstructions (overall_pass);


-- ─────────────────────────────────────────────
-- 4. VALIDATION_METRICS — per-job scientific metrics
-- ─────────────────────────────────────────────

CREATE TABLE validation_metrics (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    reconstruction_id UUID      REFERENCES reconstructions(id) ON DELETE CASCADE,
    -- Image quality metrics
    ssim            FLOAT,
    psnr            FLOAT,
    rmse            FLOAT,
    sam             FLOAT,      -- Spectral Angle Mapper (radians)
    -- Spectral index errors
    ndvi_error      FLOAT,
    ndwi_error      FLOAT,
    savi_error      FLOAT,
    ndbi_error      FLOAT,
    -- Temporal consistency
    temporal_consistency FLOAT,
    -- Verification passes (1=pass, 0=fail)
    pass_temporal   BOOLEAN,
    pass_sar        BOOLEAN,
    pass_spectral   BOOLEAN,
    pass_ai         BOOLEAN,
    pass_reference  BOOLEAN,
    pass_temporal_score   FLOAT,
    pass_sar_score        FLOAT,
    pass_spectral_score   FLOAT,
    pass_ai_score         FLOAT,
    pass_reference_score  FLOAT,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metrics_recon ON validation_metrics (reconstruction_id);
CREATE INDEX idx_metrics_ssim  ON validation_metrics (ssim);
CREATE INDEX idx_metrics_pass  ON validation_metrics (pass_temporal, pass_sar, pass_spectral);


-- ─────────────────────────────────────────────
-- 5. CLOUD_MASKS — spatial cloud extent
-- ─────────────────────────────────────────────

CREATE TABLE cloud_masks (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_id        UUID        REFERENCES scenes(id) ON DELETE CASCADE,
    cloud_class     INT         NOT NULL CHECK (cloud_class IN (0,1,2,3)),
    cloud_class_name TEXT GENERATED ALWAYS AS (
        CASE cloud_class
            WHEN 0 THEN 'clear'
            WHEN 1 THEN 'thin_cloud'
            WHEN 2 THEN 'thick_cloud'
            WHEN 3 THEN 'cloud_shadow'
        END
    ) STORED,
    geom            GEOMETRY(MultiPolygon, 4326),   -- cloud polygons
    pixel_count     INT,
    coverage_pct    FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cloud_scene ON cloud_masks (scene_id);
CREATE INDEX idx_cloud_geom  ON cloud_masks USING GIST (geom);
CREATE INDEX idx_cloud_class ON cloud_masks (cloud_class);


-- ─────────────────────────────────────────────
-- 6. ARCHIVE — historical cloud-free composites
-- ─────────────────────────────────────────────

CREATE TABLE archive (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_name      TEXT        NOT NULL,
    year            INT         NOT NULL,
    month           INT         NOT NULL CHECK (month BETWEEN 1 AND 12),
    season          TEXT GENERATED ALWAYS AS (
        CASE
            WHEN month IN (3,4,5)  THEN 'spring'
            WHEN month IN (6,7,8)  THEN 'monsoon'
            WHEN month IN (9,10,11) THEN 'autumn'
            ELSE 'winter'
        END
    ) STORED,
    aoi             GEOMETRY(Polygon, 4326),
    composite_path  TEXT        NOT NULL,
    cloud_free_pct  FLOAT       DEFAULT 100.0,
    ndvi_mean       FLOAT,
    ndwi_mean       FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (scene_name, year, month)
);

CREATE INDEX idx_archive_aoi    ON archive USING GIST (aoi);
CREATE INDEX idx_archive_date   ON archive (year, month);
CREATE INDEX idx_archive_season ON archive (season);


-- ─────────────────────────────────────────────
-- 7. SPECTRAL_SNAPSHOTS — per-scene index statistics
-- ─────────────────────────────────────────────

CREATE TABLE spectral_snapshots (
    id          UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene_id    UUID    REFERENCES scenes(id),
    recon_id    UUID    REFERENCES reconstructions(id),
    is_reconstructed BOOLEAN DEFAULT FALSE,   -- true = post-reconstruction
    ndvi_min    FLOAT, ndvi_mean FLOAT, ndvi_max FLOAT, ndvi_std FLOAT,
    ndwi_min    FLOAT, ndwi_mean FLOAT, ndwi_max FLOAT, ndwi_std FLOAT,
    savi_min    FLOAT, savi_mean FLOAT, savi_max FLOAT, savi_std FLOAT,
    ndbi_min    FLOAT, ndbi_mean FLOAT, ndbi_max FLOAT, ndbi_std FLOAT,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_spectral_scene ON spectral_snapshots (scene_id);
CREATE INDEX idx_spectral_recon ON spectral_snapshots (recon_id);


-- ─────────────────────────────────────────────
-- VIEWS
-- ─────────────────────────────────────────────

-- Summary view: all jobs with validation metrics
CREATE OR REPLACE VIEW v_job_summary AS
SELECT
    j.id              AS job_id,
    s.scene_name,
    s.sensor,
    s.acquisition_date,
    s.cloud_coverage   AS input_cloud_pct,
    j.status,
    j.progress,
    j.duration_secs,
    r.overall_pass,
    r.confidence_mean,
    r.high_conf_px_pct,
    m.ssim,
    m.psnr,
    m.rmse,
    m.ndvi_error,
    m.ndwi_error,
    m.temporal_consistency,
    j.created_at
FROM jobs j
LEFT JOIN scenes         s ON s.id = j.scene_id
LEFT JOIN reconstructions r ON r.job_id = j.id
LEFT JOIN validation_metrics m ON m.reconstruction_id = r.id
ORDER BY j.created_at DESC;


-- Monthly performance summary
CREATE OR REPLACE VIEW v_monthly_performance AS
SELECT
    DATE_TRUNC('month', j.created_at)::DATE AS month,
    COUNT(*)                                  AS total_jobs,
    SUM(CASE WHEN j.status = 'complete' THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN r.overall_pass THEN 1 ELSE 0 END)        AS passed,
    AVG(m.ssim)         AS mean_ssim,
    AVG(m.psnr)         AS mean_psnr,
    AVG(m.ndvi_error)   AS mean_ndvi_error,
    AVG(r.confidence_mean) AS mean_confidence,
    AVG(j.duration_secs)   AS mean_duration_secs
FROM jobs j
LEFT JOIN reconstructions  r ON r.job_id = j.id
LEFT JOIN validation_metrics m ON m.reconstruction_id = r.id
GROUP BY 1
ORDER BY 1 DESC;


-- ─────────────────────────────────────────────
-- SPATIAL QUERIES (as stored functions)
-- ─────────────────────────────────────────────

-- Find scenes intersecting a given bounding box
CREATE OR REPLACE FUNCTION find_scenes_in_bbox(
    lon_min FLOAT, lat_min FLOAT, lon_max FLOAT, lat_max FLOAT
) RETURNS TABLE (
    scene_id UUID, scene_name TEXT, acquisition_date DATE,
    cloud_coverage FLOAT, overall_pass BOOLEAN, ssim FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.id, s.scene_name, s.acquisition_date,
        s.cloud_coverage, r.overall_pass, m.ssim
    FROM scenes s
    LEFT JOIN reconstructions  r ON r.scene_id = s.id
    LEFT JOIN validation_metrics m ON m.reconstruction_id = r.id
    WHERE ST_Intersects(
        s.aoi,
        ST_MakeEnvelope(lon_min, lat_min, lon_max, lat_max, 4326)
    )
    ORDER BY s.acquisition_date DESC;
END;
$$ LANGUAGE plpgsql;


-- Find nearest cloud-free archive composite for a given point and date
CREATE OR REPLACE FUNCTION nearest_archive_composite(
    lon FLOAT, lat FLOAT, target_month INT
) RETURNS TABLE (
    archive_id UUID, scene_name TEXT, year INT, month INT,
    composite_path TEXT, distance_m FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        a.id, a.scene_name, a.year, a.month, a.composite_path,
        ST_Distance(
            a.aoi::GEOGRAPHY,
            ST_SetSRID(ST_MakePoint(lon, lat), 4326)::GEOGRAPHY
        ) AS distance_m
    FROM archive a
    WHERE ABS(a.month - target_month) <= 1  -- same season
    ORDER BY distance_m ASC, a.year DESC
    LIMIT 5;
END;
$$ LANGUAGE plpgsql;


-- Cloud coverage trend for a region over time
CREATE OR REPLACE FUNCTION cloud_trend(
    lon_min FLOAT, lat_min FLOAT, lon_max FLOAT, lat_max FLOAT,
    start_date DATE, end_date DATE
) RETURNS TABLE (
    acquisition_date DATE, cloud_coverage FLOAT,
    ndvi_mean FLOAT, reconstruction_pass BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.acquisition_date,
        s.cloud_coverage,
        ss.ndvi_mean,
        r.overall_pass
    FROM scenes s
    LEFT JOIN spectral_snapshots ss ON ss.scene_id = s.id AND ss.is_reconstructed = FALSE
    LEFT JOIN reconstructions     r  ON r.scene_id = s.id
    WHERE ST_Intersects(
        s.aoi,
        ST_MakeEnvelope(lon_min, lat_min, lon_max, lat_max, 4326)
    )
    AND s.acquisition_date BETWEEN start_date AND end_date
    ORDER BY s.acquisition_date;
END;
$$ LANGUAGE plpgsql;


-- ─────────────────────────────────────────────
-- TRIGGERS
-- ─────────────────────────────────────────────

-- Auto-update job started_at when status changes to 'processing'
CREATE OR REPLACE FUNCTION trg_job_started() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'processing' AND OLD.status = 'queued' THEN
        NEW.started_at = NOW();
    END IF;
    IF NEW.status IN ('complete', 'failed') AND OLD.status != NEW.status THEN
        NEW.completed_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER job_status_trigger
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION trg_job_started();


-- ─────────────────────────────────────────────
-- SEED DATA — Indian region AOI presets
-- ─────────────────────────────────────────────

CREATE TABLE aoi_presets (
    name        TEXT PRIMARY KEY,
    description TEXT,
    geom        GEOMETRY(Polygon, 4326)
);

INSERT INTO aoi_presets (name, description, geom) VALUES
  ('delhi',     'NCT Delhi region',
   ST_MakeEnvelope(76.8, 28.4, 77.4, 29.0, 4326)),
  ('mumbai',    'Greater Mumbai',
   ST_MakeEnvelope(72.7, 18.8, 73.1, 19.2, 4326)),
  ('chennai',   'Chennai and suburbs',
   ST_MakeEnvelope(79.9, 12.8, 80.4, 13.3, 4326)),
  ('bangalore', 'Bengaluru metropolitan',
   ST_MakeEnvelope(77.4, 12.8, 77.8, 13.1, 4326)),
  ('bhopal',    'Bhopal and central MP',
   ST_MakeEnvelope(77.2, 23.1, 77.6, 23.5, 4326)),
  ('assam',     'Brahmaputra floodplain',
   ST_MakeEnvelope(90.5, 25.5, 92.5, 26.5, 4326)),
  ('punjab',    'Punjab agricultural belt',
   ST_MakeEnvelope(74.5, 30.0, 76.5, 31.5, 4326));

COMMENT ON TABLE aoi_presets IS
    'Predefined AOI bounding boxes for major Indian regions';
