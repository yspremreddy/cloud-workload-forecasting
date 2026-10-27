# Local environment

This project is a Python command-line forecasting pipeline and Jupyter notebook analysis.
The local environment uses Python 3.13 in `.venv` and an editable installation of
`cloud-workload-forecasting` with the `dev` and `notebooks` extras.

No `.env` file, API keys, database, or external service configuration is required.
The CLI accepts the dataset and output paths as arguments.

Run from this directory in PowerShell (activation is optional):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m cwfc.predict --data challenge_data --output .venv/predictions.local.csv
.\.venv\Scripts\python.exe -m jupyter lab --no-browser
```

The local verification output uses `.venv/predictions.local.csv` to preserve the
existing submission file. Use `--output predictions.csv` to regenerate the submission.
The official `data-v1` dataset is extracted into `challenge_data/`; the included
SHA256 checksums were verified for both data archives and `DATA_CARD.md`.

To recreate the environment using any installed Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
```

Verified on 2026-10-27:

- `pip check`: no broken requirements.
- `pytest -q`: 41 tests passed.
- Real dataset CLI run: 24 forecast rows generated successfully.
- Output schema, five-minute cadence, finite values, and target ranges passed validation.
- Installed package versions are recorded in `.venv/installed-packages.txt`.

The first scientific-library imports were slow on this machine; subsequent
verification completed successfully. The application is a batch CLI: it exits
after writing its forecast.
