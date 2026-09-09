# BEAD Funding Predictor — Project Memory

## Project Overview
Random forest model predicting BEAD funding per location. Target: `log1p(funding_per_location)`, winsorized 2.5–97.5%. All notebooks in `random_forest/`.

## BigQuery Setup
- Project: `broadband-data`
- Key tables: `fp_approved.deployment_projects`, `fp_approved.locations`, `fp_approved.loc_cai`, `fp_approved.competition_analysis`, `fcc_bdc.nbm_hive` (period='2025_06'), `fcc_block_level_pop.us2020`
- **fcc_nbm is GONE** — use `fcc_bdc.nbm_hive` instead
- Competition columns are **BOOL** — use `IF(col, 1.0, 0.0)` not `CAST(col AS FLOAT64)`
- RUCC CSV is **long format** — must `pivot_table(columns='Attribute')` before use

## Local Data Files (in repo root `../` from `random_forest/`)
- `Ruralurbancontinuumcodes2023.csv` — long format, pivot on Attribute/Value; RUCC_2023 col
- `natamenf.xls` — skip 103 rows, use row 104 as header; col 'TOPOG - Z' → topography_z
- `ruggedness-scales-2020-tracts.csv` — CountyFIPS23 → county_id (zfill 5), aggregate RRS/ARS to county
- `ACSDT5Y2024.B17001_2026-07-15T133739/ACSDT5Y2024.B17001-Data.csv` — skip row 0 (metadata), GEO_ID[-5:] = county_id, B17001_002E/B17001_001E = poverty_rate
- `2020_UA_COUNTY.xlsx` — POPDEN_RUR, POPDEN_COU; county_id = STATE.zfill(2)+COUNTY.zfill(3)
- `xlrd` must be installed (`pip install xlrd`) to read .xls files

## Technology Codes
40=Cable/HFC, 50=Fiber, 61=Satellite/LEO, 71=Fixed Wireless

## Active Notebooks (most recent work)
- `bead_rf_final.ipynb` — 6-feature final model (baseline), R²≈0.754
- `bead_rf_drop_jobs.ipynb` — drops `jobs_per_location` (reviewer #23), adds `pct_cai` from `loc_cai`, compares 3 models
- `bead_rf_by_technology.ipynb` — separate RF per tech subset, fiber gets terrain features (pct_aerial, topography_z_wmean, RRS_wmean, rucc_code_wmean, poverty_rate_wmean)
- `bead_rf_by_tech_locnorm.ipynb` — same but state_population/state_num_providers normalized by state total funded locations
- `bead_rf_no_state_features.ipynb` — **IN PROGRESS** single model, NO state features; replaces state_num_providers with competition fracs from competition_analysis, replaces state_population with POPDEN_RUR/COU_wmean

## Key Findings
- `jobs_per_location` NOT correlated with fiber (point-biserial r=-0.022, p=0.10 — not significant). Reviewer's mechanism was wrong; variable is noise from inconsistent self-reporting. Drop it.
- Fiber model fits worse than satellite/FWA because cost varies 10-100x with terrain/aerial-vs-buried; satellite is near-flat per location
- `pct_aerial` = aerial_miles/total_fiber_miles is the single biggest expected improvement for fiber model

## County→Project Geographic Pipeline
```python
# BQ: get (project_id, county_id, location_count)
df_loc_county = client.query("""
SELECT loc.project_id, SUBSTR(nbm.block_geoid,1,5) AS county_id, COUNT(*) AS location_count
FROM `broadband-data.fp_approved.locations` loc
JOIN `broadband-data.fcc_bdc.nbm_hive` nbm ON loc.location_id = nbm.location_id
WHERE nbm.period = '2025_06'
GROUP BY loc.project_id, county_id
""").to_dataframe()
# Then: merge county features, compute weighted means per project
```

## Reviewer Comments Addressed
- #23: drop jobs_per_location (not motivated, not in Table 1, not correlated with fiber)
- #25: separate models by technology; mixed projects: use dominant tech if ≥95% purity, omit rest; state fiber stats computed per state
