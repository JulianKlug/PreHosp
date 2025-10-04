# Pre-hospital Analgesia Recommender

This project trains an offline reinforcement learning agent that recommends pre-hospital fentanyl and ketamine dosing to achieve an arrival VAS (Visual Analogue Scale) score below 3. Only variables observable in the field are used as context.

## Data
- Source workbook: `../temp_data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx`
- Each row represents a mission with field assessments, interventions, and the recorded VAS on scene and on hospital arrival.
- Medication administrations are parsed from the `Alle Medikamente detailliert` column. The parser extracts fentanyl and (es)ketamine doses, converts them to milligrams, and sums repeated administrations per mission.

### Feature set
Numeric features:
- `age`, `vas_on_scene`, `gcs`, `heart_rate`, `spo2`, `nibp_systolic`, `nibp_diastolic`, `resp_rate`, `naca_score`

Categorical features:
- `gender`, `mechanism`, `dispatch_type`, `consciousness`, `breathing_assessment`, `airway_status`, `weekday`, `month`, `time_of_day`

### Action space
Actions are discrete combinations of fentanyl and ketamine dose bands:
- Fentanyl bands: `0 mg`, `≤0.10 mg`, `0.10-0.20 mg`, `0.20-0.30 mg`, `>0.30 mg`
- Ketamine bands: `0 mg`, `≤25 mg`, `25-50 mg`, `50-75 mg`, `>75 mg`

Each action corresponds to one fentanyl band × one ketamine band. Historical medians per band pair are used as representative doses during recommendation.

### Reward
Binary reward: `1` if `VAS_on_arrival < 3`, else `0`.

Rows with missing VAS on arrival or scene values are excluded. Only missions with `VAS_on_scene ≥ 3` are retained to focus on cases where analgesia is clinically indicated.

## Training pipeline
1. **Parsing** – `rl_recommender.data.prepare_analgesia_dataset` reimplements minimal XLSX parsing (avoiding `openpyxl`), extracts medication totals, discretises doses, and returns a clean dataset plus metadata.
2. **Agent** – `AnalgesiaRecommender` (in `rl_recommender/model.py`) fits a contextual bandit via fitted Q-estimation with a `HistGradientBoostingClassifier`. States are encoded through a `ColumnTransformer`; actions are one-hot encoded.
3. **Evaluation** – After training the agent reports baseline success, expected policy value, coverage of on-policy matches, and historical best-action references.

## Usage
```bash
python train.py \
  --data-path "../temp_data/trauma_categories_Rega Pain Study15.09.2025_v2.xlsx" \
  --output-dir artifacts
```

### Outputs (stored in `artifacts/` by default)
- `metrics.json` – offline evaluation metrics.
- `action_mapping.csv` – all action bands with historical support and suggested doses.
- `dataset_summary.json` – simple stats about the filtered dataset.
- `sample_recommendations.csv` – preview of recommendations for 25 held-out missions.

Adjust `--test-size` and `--seed` as needed for experimentation.
