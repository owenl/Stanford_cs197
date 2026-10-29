# Week 1 — Results (eICU DEMO data, demo=True)

Notebook: `week1_explore_eicu_data.ipynb` — ran end-to-end after 2 fixes (see Findings).
Produced `data/clean_dataset.parquet` + `data/feature_names.pkl`.

## Pipeline numbers (from notebook outputs)
- Raw patient table: 2,520 unit stays
- Removed 200 patients discharged before 24h; removed 378 ICU visits starting after 48h
- Final `patient_agg`: **1,722 hospitalizations × 1,099 columns**
- Feature sets: admissiondx 220 · icd10_before24h 363 · icd10_after24h 318 · vitals 52 · labs 135
- Mortality 24–48h rate: **3.31%** (57 / 1,722)
- Hospitals: 186 in hospital table; **185 distinct hospitalids** in patient_agg
- Top diagnoses after 24h (label prevalence): J96.00 acute respiratory failure 19.7%, I10 hypertension 18.2%, J18.9 pneumonia 16.4%, I48.0 AFib 14.7%, N17.9 AKI 12.6%
- Top admission diagnoses: missing 7.2%, Sepsis (pulmonary) 5.3%, DKA 5.2%, CHF 4.5%, Pneumonia bacterial 3.2%

## Section 3 TODOs (completed in `work/week1_section3.py`)
1. **Medication** (75,604 orders): top = potassium chloride, acetaminophen, sodium chloride 0.9%,
   furosemide — but drug names are *not normalized*: potassium chloride alone appears under ≥5
   different strings (e.g. 'potassium chloride' 364 + 'POTASSIUM CHLORIDE' 304 + 'POTASSIUM
   CHLORIDE CRYS ER 20 MEQ PO TBCR' 228 + ...). First-24h orders: 15,557; top = potassium
   chloride variants, acetaminophen, Dextrose 50%, lorazepam, insulin-lispro.
2. **Demographics**: Male 1,037 / Female 684 / 1 missing. Caucasian 1,357 (78.8%), African
   American 169, Hispanic 84, Asian 25, Native American 11, Other/Unknown 51, missing 25.
   Age mean 62.7, median 66, max 90 ('> 89' → 90), 3 missing.
   **Data errors**: admissionheight max = 600 cm; admissionweight min = 0.5 kg. Height missing 2.26%, weight 2.21%.
3. **Hospitals**: per region — Midwest 559, South 523, West 371, region missing 160, Northeast 109.
   Largest single hospital has only **20 stays** (demo is very thin per hospital).
4. **Lab missingness**: median lab feature is **93.1% missing**; 74 of 135 labs >90% missing;
   'Site', 'ANF/ANA', '24 h urine urea nitrogen' are 100% missing in the demo.
5/6. **Baseline logistic regression** (9 covariates from the notebook, median imputation +
   StandardScaler, stratified 80/20 split, seed 42), Y = mortality_at48h:
   **AUROC 0.553 · AUPRC 0.207** (test n=345, only 11 deaths; prevalence baseline AUPRC 0.032).

## Findings / bugs
- **F1. pandas version**: notebook cell 0 imports `SettingWithCopyWarning` from `pandas.errors`
  — removed in pandas 3.x. `requirements.txt` pins nothing. Notebook runs on pandas 2.2.3.
- **F2. Filename case bug**: one cell reads `vitalperiodic.csv.gz` / `vitalaperiodic.csv.gz`
  (lowercase) but the files are `vitalPeriodic.csv.gz` / `vitalAperiodic.csv.gz`. Works on
  macOS (case-insensitive), fails on Linux.
- **F3.** Notebook defaults `demo = False` (full dataset path); demo runs need `demo = True`.
