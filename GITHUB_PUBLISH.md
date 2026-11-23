# Publish this repository to GitHub (macOS)

## Before upload

1. Read `docs/ATTRIBUTION_AND_LICENSES.md`: verify **public redistribution permission** for the bundled four trained `.tflite` files and relevant dataset-derived materials.
2. Select a license for your original code if you wish others to reuse it. **No license has been selected by the packaging process.**
3. Inspect `git status`, the file list and notebook contents for credentials. No model `.pte` or tokenizer is included, by design.
4. Run `python3 scripts/verify_models.py`; keep the report to include in your submission evidence.

## Option A — Publish with GitHub website + Git command line

1. Sign in to GitHub and create a new repository named **`CropDetector-Offline-AI-ARM-SoC`** with visibility **Public**. When creating it, do **not** initialize the remote with a README, `.gitignore`, or license, because this local project already has a README.
2. Extract the prepared source ZIP on your Mac. Open Terminal **inside** the extracted `CropDetector-Offline-AI-ARM-SoC` folder:

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "Initial ARM SoC Challenge prototype submission"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/CropDetector-Offline-AI-ARM-SoC.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your actual account name, and authenticate with the supported GitHub flow (Git Credential Manager / browser / SSH or a token). **Do not paste access tokens into repository files or chats.**

3. Refresh the repository URL, inspect the README, and check `Settings -> General -> Danger Zone -> Change repository visibility` if it is not Public.
4. Copy the actual link to the challenge submission form.

## Option B — GitHub CLI (if installed and authenticated)

From inside the extracted source directory:

```bash
git init
git branch -M main
git add .
git commit -m "Initial ARM SoC Challenge prototype submission"
gh repo create CropDetector-Offline-AI-ARM-SoC --public --source=. --remote=origin --push
```

Only use Option A **or** Option B, not both. `gh` requires a GitHub login (`gh auth login`), completed by you.

## Follow-up commits

After publishing, commit updates to the final report or video link:

```bash
git add .
git commit -m "Add final project report and demo link"
git push
```

For the final report, add `docs/Final_Project_Report.pdf` after editing and verifying its content. For the video, publish on YouTube/Google Drive as **Public** and add its URL to `demo/README.md` and the repository README.

## Large files

GitHub regular Git rejects files over 100 MiB. The external Llama `.pte` is over 1 GiB and is purposely excluded; **do not put it into this repo**. The four bundled `.tflite` models are each below the limit. If a future release needs binary distribution, assess permissions and consider official upstream links or GitHub Releases/Git LFS rather than ordinary Git history.
