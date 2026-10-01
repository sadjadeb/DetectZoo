## Summary

<!-- What changed and why? Link issues: Fixes #123 -->

## Type of change

- [ ] New detector or dataset
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactor (no behavior change)
- [ ] Tests
- [ ] Other (describe below)

## Checklist

- [ ] Commit author name/email matches my verified GitHub account
- [ ] No unintended `Co-authored-by` trailers (Cursor attribution disabled)
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest -m "not slow"` passes (or N/A — explain why)
- [ ] New public detectors/datasets are registered (decorator / `__init__.py`), tested, and listed in README if applicable
- [ ] I added/updated tests for behavior changes

## How to test

<!-- Commands or steps a reviewer can run -->

```bash
# example
pytest tests/test_audio_detectors.py -m "not slow" -k "in_the_wild"
```

## Notes for reviewers

<!-- Anything non-obvious: design trade-offs, follow-ups, known limitations -->
