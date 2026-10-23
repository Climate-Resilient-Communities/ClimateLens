# Error Log

## 2026-10-22 — Python 3.12 install failure (multiple stages)

| Field | Detail |
|---|---|
| **Date** | 2026-10-22 |
| **Reported by** | karim |
| **Fixed by** | karim |
| **Environment** | Windows amd64, Python 3.12 → 3.11, `climatelens==0.2.0`, `pip install -e ".[ml,dev]"` in venv |
| **Status** | Resolved |

### Problem
Installing `climatelens` from source on Python 3.12 fails in three separate places because the pinned dependencies predate Python 3.12 support. Switching to Python 3.11 removed the first two failures but the third persisted until `hdbscan` was bumped.

### Symptoms

1. **`numpy==1.24.3`** — pip downloads the source tarball (no `cp312` wheel) and fails:
   ```
   ModuleNotFoundError: No module named 'distutils'
   ```
   NumPy 1.24.x supports Python 3.8–3.11 only; `distutils` was removed in 3.12.

2. **`wordcloud==1.9.2`** — fails during `get_requires_for_build_wheel`:
   ```
   AttributeError: module 'configparser' has no attribute 'SafeConfigParser'.
   Did you mean: 'RawConfigParser'?
   ```
   Legacy `versioneer.py` uses an API removed in Python 3.12.

3. **`hdbscan==0.8.38.post1`** — pip resolves to the source distribution and attempts a Cython/C extension build:
   ```
   error: Microsoft Visual C++ 14.0 or greater is required.
   ```
   This failure persisted even after moving to Python 3.11.

### Root Cause
- Errors 1 and 2: dependency pins are incompatible with Python 3.12 — no `cp312` wheels exist, forcing source builds that hit removed stdlib APIs.
- Error 3: `hdbscan==0.8.38.post1` had no usable prebuilt wheel resolved for the target environment, so pip fell back to compiling from source, which requires MSVC build tools.

### Solution
1. Switched to Python 3.11 (`py -3.11 -m venv venv`), which still ships `distutils`/legacy `configparser` and has prebuilt `cp311` Windows wheels for `numpy` 1.24.3 and `wordcloud`. This resolved errors 1 and 2.
2. Bumped the pin in `pyproject.toml` from `hdbscan==0.8.38.post1` to `hdbscan==0.8.40`, which provides a prebuilt `cp311` Windows wheel and avoids the MSVC compile step entirely. This resolved error 3.

```bash
py -3.11 -m venv venv
source venv/Scripts/activate
pip install -e ".[ml,dev]"
```

### Notes
- Errors appeared sequentially as each blocker was cleared; relaxing `numpy==1.24.3` to `numpy>=1.24.3` (similarly with `wordcloud`) made pip resolve NumPy 2.5.4 (wheel), removing error #1 but exposing #2.
- Renaming the venv folder after creation is not supported — the venv was recreated cleanly under the final name instead.
- No application code changes were required; the only project change was the `hdbscan` version pin.