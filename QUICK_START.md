# Quick start

Run these commands from the project root.

## 1. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

If activation is blocked, use `venv\Scripts\python.exe` instead of `python`.

## 2. Install dependencies

```powershell
python -m pip install -e ".[dev,preprocessing]"
```

## 3. Collect candidates

```powershell
python scripts\collect_videos.py --search
```

Review `data\manifest.csv` before downloading.

## 4. Download reviewed URLs

```powershell
python scripts\collect_videos.py --download
```

Videos and original metadata go to `data\raw\videos\`.

## 5. Process downloaded videos

```powershell
python scripts\process_videos.py --model tiny
```

Audio goes to `data\processed\audio\`. Transcript JSON goes to
`data\processed\transcripts\`. Video-only files are skipped.

## 6. Check transcripts

```powershell
Get-ChildItem data\processed\transcripts
Get-Content data\processed\transcripts\nutrition-01.transcript.json
```

## 7. Run tests

```powershell
pytest -q
```

Claim extraction, evidence retrieval, verification, and final verdict commands
do not exist yet and are intentionally not documented as runnable commands.
