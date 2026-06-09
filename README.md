# FAST Python

> [!IMPORTANT]
> **Maintenance Disclaimer:** This repository is fully maintained by Codex.

Python implementation layer for **Future Aircraft Sizing Tool (FAST)** by The
IDEAS Lab in the Aerospace Engineering Department at the University of Michigan.
This project is a native Python reimplementation of MATLAB FAST.

Related repositories:

- [FAST](https://github.com/ideas-um/FAST): MATLAB source of truth.
- [FAST-Python-Wrapper](https://github.com/triet228/FAST-Python-Wrapper):
  MATLAB execution wrapper used for oracle and parity checks.

## What Is Included

- Native API: `FastPython.run(aircraft, mission)` and `fast_python.run(...)`.
- CLI runner for wrapper-style `InputAircraft.json` and `Mission.json`.
- JSON I/O compatible with `FAST-Python-Wrapper`.
- Native ports for the currently covered FAST packages, including mission
  profiles and segments, aircraft and engine specs, propulsion, battery,
  engine, optimization, constraint, OEW, database, plotting, safety,
  regression, retrofit, cost, and projection helpers.
- Native case factories for the MATLAB example matrix covered by parity tests.
- Deterministic reference replay for bundled wrapper-validated fixture cases.

FAST-Python does not require MATLAB at runtime. MATLAB is only needed for the
optional parity suites.

## Requirements

- Python 3.11
- `numpy`, `scipy`, and `pytest`
- `FAST-Python-Wrapper` plus MATLAB FAST for optional MATLAB parity tests

Install from this repository:

```sh
python -m pip install -e .
```

## Run A Case

```sh
python -m fast_python.main \
  --input-dir examples/A320/inputs \
  --output-dir examples/A320/outputs
```

Generated outputs:

- `OutputAircraft.json`
- `OutputAircraftStructure.json`

## Python API

```python
from fast_python import FastPython, native_case

aircraft, mission = native_case("A320")
result = FastPython().run(aircraft, mission)

print(result["status"])
print(result["mtow"])
print(result["backend"])
```

The module-level helper uses the same native backend:

```python
from fast_python import native_case, run

aircraft, mission = native_case("A320")
result = run(aircraft, mission)
```

## Tests

Run the default native suite:

```sh
python -m pytest -q
```

MATLAB-backed parity tests are skipped unless explicitly enabled. Set these
paths when your checkouts are outside the defaults:

```sh
export FAST_PYTHON_WRAPPER_PATH="/path/to/FAST-Python-Wrapper"
export FAST_PATH="/path/to/FAST"
```

PowerShell:

```powershell
$env:FAST_PYTHON_WRAPPER_PATH="C:\path\to\FAST-Python-Wrapper"
$env:FAST_PATH="C:\path\to\FAST"
```

Run direct MATLAB parity checks for specs, profiles, and mission segments:

```sh
FAST_PYTHON_RUN_MATLAB_PARITY=1 \
python -m pytest -q \
  tests/test_matlab_aircraft_specs_parity.py \
  tests/test_matlab_engine_specs_parity.py \
  tests/test_matlab_mission_profiles_parity.py \
  tests/test_matlab_mission_segs_parity.py
```

Run end-to-end `OutputAircraft` parity:

```sh
FAST_PYTHON_RUN_MATLAB_PARITY=1 \
python -m pytest -q tests/test_matlab_output_aircraft_parity.py
```

For one expensive output-parity case:

```powershell
$env:FAST_PYTHON_RUN_MATLAB_PARITY="1"
$env:FAST_PYTHON_OUTPUT_PARITY_CASES="A320"
python -m pytest -q tests\test_matlab_output_aircraft_parity.py
```

To verify the wrapper itself:

```sh
cd /path/to/FAST-Python-Wrapper
python -m pytest -q
```

Known parity notes:

- `MissionProfilesPkg.TakeoffTestProfile` is an expected failure when MATLAB
  lacks Aerospace Toolbox `convlength`.
- `ERJ190_FE` remains an expected end-to-end parity failure when MATLAB FAST
  itself reaches `Settings.Converged = 0`.

## Input Contract

FAST-Python reads the same input shape used by `FAST-Python-Wrapper`:

- `InputAircraft.json`: FAST aircraft dictionary.
- `Mission.json`: wrapper-style mission object with a top-level `Profile`
  dictionary, or the raw mission profile dictionary.

The native result follows the wrapper-compatible shape:

```text
{
  "status": "success",
  "mtow": <kg>,
  "aircraft": <OutputAircraft-compatible dict>,
  "log": <text>,
  "backend": "native"
}
```

The explicit `reference` backend is fixture-based and should only be used when
saved wrapper-output replay is intentional.

## Current Case Coverage

`native_case_names()` exposes these runnable preset pairings:

- `A320`, `AEA`, `ATR42`, `CeRAS`
- `ERJ175LR`, `ERJ175LR_ClimbThenAccel`, `ERJ175LR_Elec`, `ERJ190_E2`,
  `ERJ190_FE`
- `Example_Notional00`, `Example_Notional01`, `Example_Notional02`
- `Example_RegionalJet00`, `Example_RegionalJet01`, `Example_RegionalJet02`
- `Example_Turboprop00`, `Example_Turboprop01`, `Example_Turboprop02`
- `Example_ParametricRegional`, `LM100J_Conventional`, `LM100J_Hybrid`

As MATLAB algorithms are ported, compare against wrapper outputs with
`fast_python.compare.compare_json_value()` and keep behavior matched within the
defined numerical tolerances.
