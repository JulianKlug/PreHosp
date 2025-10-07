# Feature Catalog

This catalog documents the feature inputs used by the current analgesia predictive models (logistic regression and XGBoost) trained via `analgesia/predictor/train_predictor.py`. Both models share the same feature set, preprocessing logic, and encoders.

## Encoding Overview
- Numeric features are imputed with the median and standardised.
- Categorical features are imputed with the most frequent value and one-hot encoded (unknown levels are ignored at inference).
- Multi-label text columns are tokenised on `;` and `,`, lower-cased, and the 25 most frequent tokens per column are encoded as binary indicators via `MultiLabelTopKEncoder`.

## Numeric Features
| Feature | Definition | Notes |
| --- | --- | --- |
| `Alter ` | Patient age in years at mission start. | Trailing space retained from source column naming. |
| `GCS` | Glasgow Coma Scale documented on scene. | |
| `Temperatur` | Patient temperature recorded by EMS (°C). | |
| `VAS_on_scene` | Visual analogue pain score assessed on scene. | Target cohort requires `VAS_on_scene` > 3. |
| `etCO2` | End-tidal CO₂ measurement. | |
| `AF` | Atemfrequenz / respiratory rate per minute. | |
| `SPO2` | Peripheral oxygen saturation; sourced from `SPO2 Einsatzart`. | Alias resolved through `COLUMN_ALIASES`. |
| `Körperfläche` | Documented body surface area. | |
| `doctor_age` | Age of the attending physician. | Computed as `reference_year` (2025) minus doctor birth year from metadata. |
| `heart_rate` | Patient heart rate in beats per minute. | Coalesced from `heart_rate`, `HF`, `HF `, and `HR` columns. |
| `diastolic` | Diastolic blood pressure. | Coalesced from invasive/non-invasive diastolic readings (`IBD*`, `NIBD*`). |
| `systolic` | Systolic blood pressure. | Coalesced from invasive/non-invasive systolic readings. |
| `naca` | Numeric NACA severity score. | Filled from `NACA (nummerisch)` with fallback to `NACA`. |
| `venous_access_count` | Number of venous access entries recorded. | Parsed from `Zugänge`; counts distinct access lines per mission. |
| `oxygen_delivery_rate_lpm` | Oxygen flow rate in litres per minute. | Extracted from numeric patterns in `Sauerstoffabgaben`. |

## Categorical Features
| Feature | Definition | Notes |
| --- | --- | --- |
| `Lebensbedrohliche Blutung` | Documented presence and location of life-threatening bleeding. | |
| `Befund` | Primary clinical findings recorded by the crew. | |
| `Hautbefund` | Documented skin assessment. | |
| `psychischer Erstbefund` | Initial psychological status / mental assessment. | |
| `Geschlecht` | Patient sex recorded in the chart. | |
| `Herzrhytmusstörungen` | Structured heart rhythm assessment. | |
| `Weitere Massnahmen` | Additional interventions noted in free text. | Normalised (duplicates removed, delimiters standardised) before encoding. |
| `Wochentag` | Day of week of the mission. | |
| `Monat` | Month of the mission. | |
| `Tag oder Nacht` | Day/night flag for mission start. | |
| `Ort` | Municipality / locality of the mission. | |
| `Kanton` | Canton where the mission occurred. | |
| `Atemwegbefund` | Airway assessment findings. | |
| `Gesicht` | Findings related to the patient's face. | |
| `Strasse` | Street recorded for the mission location. | |
| `Bewusstseinlage` | Level of consciousness on scene. | |
| `EKG 3-Kanal` | Utilisation of a three-lead ECG. | |
| `Ist venös bestehend` | Indicates the presence of a pre-existing venous access. | |
| `Cincinnati FAST` | Result of the Cincinnati FAST stroke screen. | |
| `EKG Telemetrie` | Telemetry ECG usage. | |
| `EKG 12-Kanal` | Utilisation of a 12-lead ECG. | |
| `Ist ossär bestehend` | Indicates an existing intraosseous access. | |
| `Aktuelles Ereignis` | Categorised incident type. | Sourced from `Aktuelles Ereignis Hauptdiagnose` via column alias. |
| `Einteilung` | Mission classification recorded by dispatch/crew. | |
| `Körperregion` | Documented primary body region affected. | |
| `Kategorie` | Detailed injury/incident category. | |
| `Lagerungen` | Patient positioning interventions. | |
| `Atemgeräusche` | Breath sound assessment. | |
| `Pupillenweite (re)` | Right pupil size. | |
| `Arm (Empfindlichkeit re)` | Sensory response of the right arm. | |
| `Bein (Empfindlichkeit li)` | Sensory response of the left leg. | |
| `Arm (Bewegung re)` | Motor response of the right arm. | |
| `Bein (Bewegung li)` | Motor response of the left leg. | |
| `Pupillenweite (li)` | Left pupil size. | |
| `Lichtreaktion (re)` | Right pupil light reactivity. | |
| `Bein (Bewegung re)` | Motor response of the right leg. | |
| `Bein (Empfindlichkeit re)` | Sensory response of the right leg. | |
| `Arm (Bewegung li)` | Motor response of the left arm. | |
| `Arm (Empfindlichkeit li)` | Sensory response of the left arm. | |
| `doctor_sex` | Sex of the attending physician. | Derived from roster; values: `male`, `female`, `unknown`. |
| `doctor_specialist_qualifications` | Consolidated list of physician specialist qualifications. | Pulled from doctor metadata; empty string when unspecified. |
| `venous_access_types` | Access type descriptors (e.g., venous, IO). | Extracted from `Zugänge`; unique tokens joined. |
| `venous_access_locations` | Access anatomical locations. | Extracted from `Zugänge`. |
| `venous_access_sides` | Laterality of venous access placements. | Extracted from `Zugänge`. |
| `venous_access_sizes` | Cannula sizes recorded for accesses. | Extracted from `Zugänge`. |
| `venous_access_preexisting` | Notes on whether the line pre-existed. | Extracted from `Zugänge`. |
| `venous_access_inserted_by` | Personnel recorded as placing the access. | Extracted from `Zugänge`. |
| `oxygen_delivery_mode` | Description of oxygen delivery interface (e.g., mask, nasal). | Cleaned from `Sauerstoffabgaben` after removing rate and bracket text. |

## Multi-label Text Features
| Feature | Definition | Notes |
| --- | --- | --- |
| `Puls tastbar` | Documented palpable pulse locations. | Tokens derived from delimited entries; top 25 indicators retained. |
| `Bergungen` | Rescue/extrication techniques applied. | Multi-label token encoding. |
| `Ereignisort` | Detailed incident settings (e.g., home, road). | Multi-label token encoding. |
| `Blutstillung` | Hemorrhage control techniques recorded. | Multi-label token encoding. |
| `Thoraxdrainage` | Chest drain details (type, site, side). | Multi-label token encoding. |

## Engineered Feature Sources
- Physician attributes (`doctor_age`, `doctor_sex`, `doctor_specialist_qualifications`) are merged from roster (`Liste Notärzte-1.xlsx`) and metadata (`final_complete_extractions_*.xlsx`) after cleaning names; missing values default to `unknown`/empty.
- Venous access fields and counts originate from structured parsing of the `Zugänge` column; tokens are deduplicated per mission before encoding.
- Oxygen delivery fields originate from the `Sauerstoffabgaben` free-text column, separating numeric flow rates from descriptive mode text.
- Alias resolution combines redundant numeric vitals into canonical columns (`heart_rate`, `diastolic`, `systolic`, `naca`) to reduce sparsity.
