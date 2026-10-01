# Contributing to DetectZoo

Thank you for helping improve DetectZoo. This project uses a **pull-request workflow**: all changes land on `main` through reviewed PRs, not direct pushes.

## Before you start

1. **Open an issue first** for large features (new detector families, API changes, new dependencies). Small fixes and docs typos can skip this.
2. **Verify your Git identity** matches a [verified email](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address) on your GitHub account:

   ```bash
   git config user.name "Your Full Name"
   git config user.email "you@example.com"
   ```

## Development setup

```bash
git clone https://github.com/sadjadeb/DetectZoo.git
cd DetectZoo
pip install -e ".[dev]"
```

Run the test suite (fast tests only; skip model downloads):

```bash
pytest -m "not slow"
```

Lint:

```bash
ruff check .
```

Fix lint issues:

```bash
ruff check --fix .
```

CI runs on every pull request and every push to `main` (see `.github/workflows/ci.yml`). Fast tests and `ruff check` must be green before merge.

## Workflow

### 1. Sync `main`

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
```

Use `--ff-only` so Git refuses to create accidental merge commits on `main`.

If you ever see a message about diverged history after a force-push on `main`, **do not merge**. Reset to the remote:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

Re-apply any unmerged work on a new branch and open a fresh PR.

### 2. Create a feature branch

Use a short, descriptive name:

```bash
git checkout -b feature/add-foo-detector
# or: fix/itw-loader-partition, docs/update-readme
```

### 3. Make focused commits

- One logical change per PR when possible; smaller PRs are reviewed faster.
- Write clear commit messages in the imperative mood (`Add …`, `Fix …`, `Update …`).
- Do not add `Co-authored-by` trailers unless a human co-author actually contributed.

### 4. Push and open a PR

```bash
git push -u origin feature/add-foo-detector
```

Open a pull request on GitHub: **base `main` ← compare your branch**.

Fill in the PR template checklist. Link related issues (`Fixes #123`).

### 5. Keep your branch up to date

If `main` moves while your PR is open:

```bash
git fetch origin
git rebase origin/main
# resolve conflicts if any
git push --force-with-lease
```

Prefer **rebase** over merging `main` into your branch; it keeps history linear.

### 6. After merge

```bash
git checkout main
git pull --ff-only origin main
git branch -d feature/add-foo-detector
```

Delete the remote branch when GitHub prompts you.

## What to include in a PR

| Change type | Expectations |
|-------------|--------------|
| New detector | Register in the appropriate `__init__.py`, add tests, update README tables if public-facing |
| New dataset | Follow existing loader conventions in `detectzoo/datasets/`, add tests |
| Bug fix | Add or extend a test when feasible |
| Docs only | No tests required; keep examples runnable |

See `examples/custom_detector.py` for the detector extension pattern and the **Extensibility** section in `README.md`.

## Review and merge

- At least **one approval** from a maintainer is required before merge.
- CI checks must pass.
- Maintainers may squash or rebase-merge depending on commit quality; either way, the final commit on `main` should be clean.

**Only maintainers merge PRs.** Contributors should not push to `main` directly.

## Branch naming conventions

| Prefix | Use for |
|--------|---------|
| `feature/` | New detectors, datasets, benchmarks |
| `fix/` | Bug fixes |
| `docs/` | Documentation and website |
| `refactor/` | Internal restructuring without behavior change |
| `test/` | Test-only changes |

## Code style

- Python 3.9+ compatible (`requires-python` in `pyproject.toml`). CI tests 3.11 and 3.12.
- Line length 100 (Ruff)
- Follow patterns in neighboring modules; match naming and structure of existing detectors/datasets
- Avoid unrelated drive-by refactors in the same PR

## Reporting bugs

Open a [GitHub issue](https://github.com/sadjadeb/DetectZoo/issues) with:

- DetectZoo version (`pip show detectzoo` or `detectzoo.__version__`)
- Python version
- Minimal code snippet or command to reproduce
- Full error traceback

## Questions

Open an issue with the `question` label or ask in your team channel before spending time on a large PR that might not fit project direction.
