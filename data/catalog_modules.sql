-- RoomSpec AI catalog schema (PostgreSQL: Neon / Supabase).
-- The app creates this table automatically via SQLAlchemy; this file is for manual setup.

CREATE TABLE IF NOT EXISTS cabinet_modules (
    part_id            VARCHAR(64)  PRIMARY KEY,
    part_name          VARCHAR(255) NOT NULL,
    category           VARCHAR(64)  NOT NULL,  -- 'Base Cabinet', 'Wall Cabinet', 'Countertop', ...
    finish_style       VARCHAR(64)  NOT NULL,  -- 'Matte Walnut', 'Gloss White', ...
    material           VARCHAR(128) NOT NULL DEFAULT '',
    price_usd          NUMERIC(10, 2) NOT NULL CHECK (price_usd >= 0),
    width_cm           NUMERIC(6, 2)  NOT NULL CHECK (width_cm > 0),
    height_cm          NUMERIC(6, 2)  NOT NULL CHECK (height_cm > 0),
    depth_cm           NUMERIC(6, 2)  NOT NULL CHECK (depth_cm > 0),
    door_clearance_cm  NUMERIC(6, 2)  NOT NULL DEFAULT 0,  -- swing/drawer extension needed in front
    in_stock           BOOLEAN DEFAULT TRUE,
    description        TEXT NOT NULL DEFAULT '',
    image_url          TEXT NOT NULL,
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_modules_specs
    ON cabinet_modules (category, width_cm, finish_style, in_stock);
