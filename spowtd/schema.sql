-- Data schema for Spowtd

PRAGMA foreign_keys = 1;


-- Staging tables, prior to regridding
CREATE TABLE rainfall_intensity_staging (
  epoch integer NOT NULL PRIMARY KEY,
  rainfall_intensity_mm_h double precision NOT NULL
);


CREATE TABLE water_level_staging (
  epoch integer NOT NULL PRIMARY KEY,
  zeta_mm double precision NOT NULL
);


CREATE TABLE evapotranspiration_staging (
  epoch integer NOT NULL PRIMARY KEY,
  evapotranspiration_mm_h double precision NOT NULL
);

-- Gridded tables

CREATE TABLE time_grid (
  time_step_s integer NOT NULL,
  -- Singleton
  is_valid integer NOT NULL PRIMARY KEY
    CHECK (is_valid = 1)
    DEFAULT 1
);


CREATE TABLE grid_time (
  epoch integer NOT NULL PRIMARY KEY
);


CREATE TABLE grid_time_flags (
  start_epoch integer NOT NULL PRIMARY KEY
    REFERENCES grid_time(epoch),
  is_raining boolean NOT NULL,
  is_jump boolean NOT NULL,
  is_mystery_jump boolean NOT NULL,
  is_interstorm boolean NOT NULL
);


CREATE TABLE rainfall_intensity (
  from_epoch integer NOT NULL
    REFERENCES grid_time (epoch)
    CHECK (from_epoch < thru_epoch),
  thru_epoch integer NOT NULL
    REFERENCES grid_time (epoch)
    CHECK (from_epoch < thru_epoch),
  rainfall_intensity_mm_h double precision NOT NULL,
  PRIMARY KEY (from_epoch)
);


CREATE TABLE water_level (
  epoch integer NOT NULL
    REFERENCES grid_time (epoch),
  zeta_mm double precision NOT NULL,
  PRIMARY KEY (epoch)
);


CREATE TABLE storm (
  start_epoch integer NOT NULL
    REFERENCES grid_time (epoch)
    CHECK (start_epoch < thru_epoch),
  thru_epoch integer NOT NULL
    REFERENCES grid_time (epoch)
    CHECK (start_epoch < thru_epoch),
  PRIMARY KEY (start_epoch)
);


CREATE TABLE zeta_interval (
  start_epoch integer NOT NULL PRIMARY KEY
    REFERENCES water_level (epoch),
  interval_type text NOT NULL
    CHECK (interval_type in ('storm', 'interstorm')),
  thru_epoch integer NOT NULL
    CHECK (start_epoch < thru_epoch)
    REFERENCES water_level (epoch),
  -- For FK
  UNIQUE (start_epoch, interval_type)
);


CREATE TABLE zeta_interval_storm (
  interval_start_epoch integer NOT NULL PRIMARY KEY,
  interval_type text NOT NULL CHECK (interval_type = 'storm'),
  storm_start_epoch integer NOT NULL,
  FOREIGN KEY (interval_start_epoch, interval_type)
    REFERENCES zeta_interval (start_epoch,
                              interval_type),
  FOREIGN KEY (storm_start_epoch)
    REFERENCES storm (start_epoch)
);


CREATE TABLE zeta_grid (
  -- Enforce singleton
  id boolean PRIMARY KEY DEFAULT TRUE CHECK (id = TRUE),
  grid_interval_mm double precision NOT NULL
);


CREATE TABLE discrete_zeta (
  zeta_number integer PRIMARY KEY,
  zeta_grid boolean NOT NULL DEFAULT TRUE REFERENCES zeta_grid(id)
);


CREATE TABLE rising_interval (
  start_epoch integer NOT NULL PRIMARY KEY
    REFERENCES zeta_interval_storm (interval_start_epoch),
  interval_type text NOT NULL DEFAULT 'storm'
    CHECK (interval_type = 'storm'),
  rain_depth_offset_mm double precision NOT NULL
);


CREATE TABLE recession_interval (
  start_epoch integer NOT NULL PRIMARY KEY,
  interval_type text NOT NULL DEFAULT 'interstorm'
    CHECK (interval_type = 'interstorm'),
  time_offset interval NOT NULL,
  FOREIGN KEY (start_epoch, interval_type)
    REFERENCES zeta_interval (start_epoch, interval_type)
);


CREATE TABLE rising_interval_zeta (
  start_epoch integer NOT NULL
    REFERENCES rising_interval (start_epoch),
  zeta_number integer NOT NULL REFERENCES discrete_zeta(zeta_number),
  mean_crossing_depth_mm double precision NOT NULL,
  PRIMARY KEY (start_epoch, zeta_number)
);


CREATE TABLE recession_interval_zeta (
  start_epoch integer NOT NULL
    REFERENCES recession_interval (start_epoch),
  zeta_number integer NOT NULL REFERENCES discrete_zeta(zeta_number),
  mean_crossing_time interval NOT NULL,
  PRIMARY KEY (start_epoch, zeta_number)
);


-- Views

CREATE VIEW storm_total_rain_depth AS
SELECT s.start_epoch AS storm_start_epoch,
       SUM(ri.rainfall_intensity_mm_h *
           (ri.thru_epoch - ri.from_epoch)
	   / 3600.
           ) AS total_depth_mm
FROM storm AS s
JOIN rainfall_intensity AS ri
  ON ri.from_epoch >= s.start_epoch
  AND ri.thru_epoch <= s.thru_epoch
GROUP BY s.start_epoch;
