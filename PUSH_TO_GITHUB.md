# Pushing EPISIM v0.1.0-alpha to GitHub — final user step

The Week-1 build is complete locally. To make it public, run these three commands from the `EPISIM/` directory in a shell where you are authenticated to GitHub (either via `gh auth login` or via a configured SSH key / token):

## 1. Create the GitHub repo and add the remote

```bash
# Option A — using GitHub CLI (recommended)
cd "D:/Anti ageing research/EPISIM"
gh repo create hssling/episim --public \
    --description "Epidemiology Platform for In Silico Methods" \
    --source=. --remote=origin

# Option B — manual (if no gh CLI)
# 1. Create repo at https://github.com/new (name: episim, public, no init files)
# 2. git remote add origin git@github.com:hssling/episim.git   (or https://...)
```

## 2. Push branch and tag

```bash
git push -u origin main
git push --tags
```

## 3. Create the GitHub release from the tag

```bash
gh release create v0.1.0a1 \
    --title "EPISIM v0.1.0-alpha (foundation)" \
    --notes-file CHANGELOG.md
```

## 4. Verify

After ~3 minutes:
- Repository: https://github.com/hssling/episim
- CI run: https://github.com/hssling/episim/actions  → both `CI` and `Docs` workflows must be green
- Docs site (after `Docs` workflow finishes): https://hssling.github.io/episim
- Release: https://github.com/hssling/episim/releases/tag/v0.1.0a1

Once all four URLs resolve cleanly, EPISIM v0.1.0-alpha is publicly live.

## Optional next steps (still Week 1, not blocking)

- Enable GitHub Pages in repo Settings → Pages → Source = "GitHub Actions" (the docs workflow needs this once).
- Connect Zenodo to the repository at https://zenodo.org/account/settings/github/ to get a DOI on the next tag.
- Reserve the PyPI name `episim` (only needed before v0.8 PyPI publish): `python -m pip install --upgrade twine && python -m build && twine register dist/*.whl` (optional).
