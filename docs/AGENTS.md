# Repository Guidelines

## Project Structure & Module Organization
- `analgesia/`: clinical pain-management analyses; notebooks live alongside the `extract_medreg_data/` Selenium tooling and interim Excel exports under `temp_data/`.
- `intubation_trigger_gcs/` and `pediatric_trauma/`: focused exploratory notebooks for airway and trauma cohorts; keep derived figures in-place to preserve provenance.
- `utils/utils.py`: lightweight shared helpers—extend here before duplicating notebook logic.
- Treat large Excel inputs under `analgesia/temp_data/` as read-only; place new raw data in a sibling directory with a README describing provenance.

## Build, Test, and Development Commands
- Create an isolated environment: `python3 -m venv .venv && source .venv/bin/activate`.
- Install core tooling: `pip install pandas selenium openpyxl webdriver-manager jupyter pytest` (match notebook imports when adding libraries).
- Run the MedReg extractor: `python analgesia/extract_medreg_data/run_extraction.py data/physicians.xlsx output/ --headless`.
- Execute notebooks reproducibly: `jupyter nbconvert --execute analgesia/table1.ipynb --to notebook --output table1.out.ipynb`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, descriptive snake_case for functions, and PascalCase for classes (`MedRegDataExtractor`).
- Prefer type hints and docstrings for new Python modules; keep logging consistent with the existing extractor scripts.
- Notebook filenames should summarize purpose (`table1.ipynb`, `fig_mechanism.ipynb`); include markdown cells that explain each analytical step.

## Testing Guidelines
- Add `pytest` suites under `tests/` when extending `utils/` or introducing reusable modules; aim for coverage of parsing and matching utilities.
- For notebooks, validate critical pipelines by executing them end-to-end before committing; capture key figures/tables in the repo so reviewers can diff outputs.
- When touching Selenium flows, smoke-test with a short doctor subset (`--max-doctors 5`) to confirm selectors still resolve.

## Commit & Pull Request Guidelines
- Match the existing history: concise, present-tense summaries (e.g., `improve medreg matching`).
- Rebase onto main before opening a PR; include a short rationale, data sources touched, and screenshots or sample table previews when visual output changes.
- Link tracking issues where applicable and call out any data files that must remain local.

## Data Handling & Security
- Medical registry extracts may include sensitive identifiers—store raw exports outside version control and mask personal data before sharing.
- Scraper credentials should rely on environment variables or `.env` files excluded via `.gitignore`; never commit API keys or session cookies.
