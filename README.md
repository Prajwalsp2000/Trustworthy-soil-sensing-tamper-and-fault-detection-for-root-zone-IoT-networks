# Trustworthy soil sensing: tamper and fault detection for root-zone IoT networks

Code and reference results for the book chapter *Trustworthy Soil Sensing: Lightweight Anomaly Detection for
Tamper and Fault Detection in Root-Zone IoT Networks* (Prajwal S P, Goutham T R, Bhagyalaxmi S Banakar;
Bentham Science, *Rhizosphere Intelligence for Agriculture 5.0*).

Archived on Zenodo: https://doi.org/10.5281/zenodo.24199451

## What this does
1. Loads two open soil-sensor records (water content, soil temperature, electrical conductivity; 15-minute data).
2. Injects labelled faults (spike, stuck-at, drift) and attacks (spoofing, replay, slow poisoning) into the real series.
3. Runs generic detectors (rolling z-score, cross-sensor rules, Isolation Forest, autoencoder) and the attack-specific
   checks (exact-repeat replay check, per-sensor flat-run check), alone and combined.
4. Compares all detectors at the same realised false-alarm rate and writes tables, CSV files and figures.

All faults and attacks are **injected into recorded data**; the results are a feasibility study, not evidence of
performance against a real adversary. Do not use this code as an operational security control.

## Data (not included)
Download the two files from HydroShare (Creative Commons Attribution) and place them in a folder `data/`:
- Metompkin Farm: https://doi.org/10.4211/hs.c414b4b08e2647da9ae5d70dc1aae60c  (`MetompkinFarm_WC_ST_SC_220719-240723.csv`)
- Crisfield Farm: https://doi.org/10.4211/hs.ffb417acb8bb4fdda660d8756b38b4cd  (`CrisfieldFarm_WC_ST_SC_210714-240711.csv`)

`DATA_CHECKSUMS.txt` lists the SHA-256 of the files used here, so you can confirm you have the same version.
Please cite the data providers (Pratt et al., 2025) if you reuse the data. Notes: the documented malfunction flag 999
does not occur in these files; the shallow-only unit `VFMS-SP-S` of Metompkin is excluded (no depth partner).

## Install
```
pip install -r requirements.txt
```
Python 3.12 and 3.13 were used. Tested versions are listed in `requirements.txt`.

## Run (about 10-15 minutes on one CPU core)
```
python soil_trust_v2c.py --met data/MetompkinFarm_WC_ST_SC_220719-240723.csv \
                         --cris data/CrisfieldFarm_WC_ST_SC_210714-240711.csv \
                         --out results --seeds 3 --severities 0.5 1.0 2.0
```
Quick test (2-4 minutes): add `--seeds 1 --severities 1.0` (and drop the other severities).
Windows: use `^` instead of `\` at line ends, or put the command on one line.

Rebuild the chapter tables from the output: `python make_tables.py results` (writes `tables_auto.tex`).

## Files
- `soil_anomaly.py` - data loader, fault/attack injector, generic detectors (first version of the pipeline).
- `soil_trust_v2c.py` - replay and flat-run checks, v2 tiers, combinations, evaluation at matched false-alarm
  rates, figures. Needs `soil_anomaly.py` in the same folder.
- `make_tables.py` - turns the result CSVs into LaTeX tables.
- `results_reference_run/` - output of the run behind the chapter (CSV files, run information, figures).

## Reproducibility
Random seeds are fixed (derived from the sensor-unit name and the seed index). Independent runs on Windows 11
(Python 3.13) and Linux (Python 3.12) gave identical results for the reported tables.

## Limitations
Injected events only; one sensor modified at a time with the others assumed clean; two farms in one coastal
region with one sensor model; the design was refined during the study (see the chapter's limitations section).

## License and citation
Code: MIT (see `LICENSE`). Cite via `CITATION.cff` (GitHub shows a "Cite this repository" button) and cite the
accompanying chapter.
