# Git Cleanup Guide

Use this when you are ready to publish a recruiter-facing version of the project.

## Recommended Safe Path

Create a new polished branch and make one clear commit:

```bash
git checkout -b codex/recruiter-ready
git add .gitignore Makefile README.md configs docs main.py requirements.txt src training_history.json training_model.ipynb testing_model.ipynb home_page.jpeg
git add -u requirement.txt
git commit -m "Prepare recruiter-ready ML project"
git push origin codex/recruiter-ready
```

Then open a pull request into `main`.

## What Not To Commit

Keep these out of Git:

- `dataset-2/`
- `ml_env/`
- `trained_model.keras`
- `Trained_model.h5`
- `.DS_Store`
- `reports/`
- `__pycache__/`

## Publishing Model Weights

Because the model file is large, publish it separately through one of these:

- GitHub Releases
- Hugging Face Hub
- Google Drive
- Kaggle dataset artifact

Then add the download link to `README.md`.

## History Rewrite Option

If you want the public repository to show only one clean commit, use a squash merge or create a fresh repository. Do not rewrite shared history until you are sure no one else depends on the current branch.
