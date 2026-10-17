#!/usr/bin/env bash
# Puts this project on GitHub as a series of small commits.
# Before running: create an EMPTY repository called portfolio-optimization on
# github.com (no README, no .gitignore, no licence).
# Usage (from the project folder, in WSL):  bash publish.sh <your-github-username>
set -euo pipefail

GH_USER="${1:?Usage: bash publish.sh <your-github-username>}"

if [ -d .git ]; then
    echo "This folder is already a git repository. Nothing done." >&2
    exit 1
fi
if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
    echo "Set your git name and email first, for example:" >&2
    echo '  git config --global user.name "Your Name"' >&2
    echo '  git config --global user.email "you@example.com"' >&2
    exit 1
fi

sed -i "s/<your-username>/${GH_USER}/g" README.md

git init -q
git config core.fileMode false   # WSL shows Windows files as executable; ignore that

git add src tests conftest.py requirements.txt .gitignore .github
git commit -q -m "Add portfolio maths, optimisers and tests"

git add analysis.ipynb
git commit -q -m "Add analysis notebook"

git add data figures
git commit -q -m "Add price data and figures"

git add README.md
git commit -q -m "Write README with results and conclusion"

git branch -M main
git remote add origin "https://github.com/${GH_USER}/portfolio-optimization.git"
git log --oneline
echo
echo "Pushing to https://github.com/${GH_USER}/portfolio-optimization ..."
git push -u origin main
