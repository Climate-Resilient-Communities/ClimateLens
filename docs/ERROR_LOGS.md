# Error Log

## 2026-10-22 — Python 3.12 install failure (multiple stages)

| Field | Detail |
|---|---|
| **Date** | 2026-10-22 |
| **Reported by** | karim |
| **Fixed by** | karim (workaround); maintainers TBD |
| **Environment** | Windows amd64, Python 3.12, `climatelens==0.2.0`, `pip install -e .` in venv |
| **Status** | Resolved via workaround (Python 3.11) |

### Problem
Installing `climatelens` from source on Python 3.12 fails in three separate places because the pinned dependencies predate Python 3.12 support.

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

3. **`hdbscan`** — needs Cython/C extension compilation:
   ```
   error: Microsoft Visual C++ 14.0 or greater is required.
   ```
   No suitable wheel resolved, so pip attempted a source build.

### Root Cause
Dependency pins are incompatible with Python 3.12: no prebuilt wheels exist for these versions on `cp312`, forcing source builds that hit removed stdlib APIs or require a C++ toolchain.

### Solution (Workaround)
Use Python 3.11, which still ships `distutils`/legacy `configparser` and has prebuilt `cp311` Windows wheels for `numpy` 1.24.3 and `hdbscan`:

```bash
py -3.11 -m venv venv311
venv311\Scripts\activate
pip install -e .
```

### Recommended Fix (Maintainers)
- Bump pins: `numpy>=1.26.4`, `wordcloud>=1.9.3` (first with Python 3.12 support).
- Verify `hdbscan` resolves to a wheel on 3.12 (0.8.44+ provides `cp312` wheels).
- Add `Programming Language :: Python :: 3.11` classifier only after a clean install is confirmed.

### Notes
- Errors appeared sequentially as each blocker was cleared; relaxing `numpy==1.24.3` to `numpy>=1.24.3` made pip resolve NumPy 2.5.4 (wheel), removing error #1 but exposing #2.
- No code changes were required — environment/dependency issue only.