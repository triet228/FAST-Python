# FAST Python

> [!IMPORTANT]
> **Maintenance Disclaimer:** This repository is fully maintained by Codex.

Python implementation layer for **Future Aircraft Sizing Tool (FAST)** by The
IDEAS Lab in the Aerospace Engineering Department at the University of Michigan.

This repository is the Python-side home for converting the MATLAB FAST project
into Python.

Related repositories:

- [FAST](https://github.com/ideas-um/FAST): MATLAB source project.
- [FAST-Python-Wrapper](https://github.com/triet228/FAST-Python-Wrapper):
  wrapper and example fixture project.

The current package provides:

- A wrapper-compatible Python API: `FastPython.run(aircraft, mission)`.
- A command-line runner that reads `InputAircraft.json` and `Mission.json`.
- JSON loaders, validators, and output writers compatible with
  `FAST-Python-Wrapper`.
- Bundled reference inputs and outputs for the wrapper-validated cases.
- Native Python ports of foundational FAST utilities:
  unit conversions, standard atmosphere, gravity, mission profile segment
  indexing, flight-condition conversion, and takeoff/detailed-takeoff/climb/
  cruise/CruiseBRE/descent/landing segment evaluation with FlyMission-style
  target iteration.
- Native Python port of `MissionHistTable` as a lightweight row/column mission
  history table helper.
- Native Python ports of PlotPkg's `PlotPerfParam` and `PlotMission` data
  formatting for state, instantaneous, and mission-history plot series.
- Native Python ports of all MissionProfilesPkg preset profiles, including
  A320, AEA, ATR42-600, CeRAS, notional/regional/turboprop/BRECruise
  families, DiversionMission, ParametricRegional, LM100J/LM100J_NoRsrv,
  ERJ variants, ATRMissionBRE, ATRMissionEPASS, and TakeoffTestProfile.
- Native Python ports of all AircraftSpecsPkg and EngineSpecsPkg presets:
  A320Neo, AEA, ATR42, CeRAS aircraft, AircraftSpecsPkg Example,
  ERJ175LR, ERJ175LR_Elec, ERJ190_E2, ERJ190_FE,
  LM100J_Conventional, LM100J_Hybrid, plus the full engine preset catalog and
  ExampleTF/ExampleTP engine templates.
- Native case factories that pair those specs and mission profiles for
  `A320`, `AEA`, `ATR42`, and `CeRAS`.
- Native Python port of RetrofitPkg's example retrofit options initializer.
- Native Python ports of core data-structure and regression utilities:
  missing-field defaulting, SpecProcessing-style regression/default filling,
  mission-history allocation/reset, database path searches, regression
  preprocessing, and FAST's Gaussian-process kernel.
- SciPy-backed loading for FAST's `IDEAS_DB.mat` historical aircraft and
  engine databases, plus DatabasePkg tree-flattening and training/validation
  randomization helpers, turbofan/turboprop unit-metadata population, and
  derived aircraft-value population.
- Native Python port of CostPkg's battery replacement cost model.
- Native Python ports of BatteryPkg's Lithium-ion charge/discharge dynamics,
  ground charging, cycling-aging SOH/FEC estimates, and post-mission battery
  resizing logic for simple and detailed battery models.
- Native Python port of FAST's key-performance-parameter projection curves
  for default technology assumptions.
- Native Python port of SafetyPkg's fault-tree analysis, Boolean cut-set
  simplification, failure probability, and component failure-rate helpers.
- Native Python ports of OptimizationPkg helper routines for flags,
  feasible step limits, Gaussian elimination, Hessian updates, merit
  functions, phase-I golden-section line search, simplex tableau setup/solve
  post-processing, optimized split retrieval, power-management objective
  bookkeeping, design/operation sizing constraints, design split optimization
  orchestration, operational split optimization orchestration, and the
  constrained interior-point optimizer.
- Native Python ports of initial ConstraintDiagramPkg helpers:
  PsLoss sigmoid, OEI multiplier, approach-speed, takeoff-field-length,
  landing-field-length, cruise, diversion, service-ceiling, all-engines
  climb, FAR 25 climb residuals, and the constraint-diagram grid evaluator.
- Native Python port of OEWPkg's operating-empty-weight iteration for the
  coupled airframe/propulsion/wing-area sizing loop.
- Native Python port of EAPAnalysis's aircraft sizing/performance loop for
  workflows covered by the currently ported native packages.
- Native Python ports of EngineModelPkg's SimpleOffDesign turbofan fuel-flow
  model, isentropic/spec-heat helpers, turbofan/turboprop linear sizing
  estimates, on-design diffuser, burner, compressor, turbine, nozzle, and fan
  system components, the off-design nozzle helper, plus turbofan/turboprop
  on-design cycles and nonlinear sizing loops.
- Native Python ports of core propulsion architecture utilities:
  split evaluation, power-flow propagation, propulsion sizing, engine lapse,
  power availability, full-throttle split recomputation,
  conventional, electric, parallel-hybrid, series-hybrid, turboelectric, and
  partially turboelectric architecture construction,
  custom-architecture validation, wrapper constant-matrix split expression
  parsing, parallel-connection bookkeeping, detailed battery discharge
  integration, turbofan nonlinear engine sizing, SimpleOffDesign turbofan
  fuel-flow integration, and turboprop nonlinear fuel-flow integration.
- A deterministic reference backend for the wrapper-validated fixture cases:
  `A320`, `AEA`, `ATR42`, and `CeRAS`.
- A native workflow runner for aircraft/mission combinations covered by the
  currently ported Python packages.

The reference backend is intentionally MATLAB-free at runtime. It uses the
saved `OutputAircraft.json` fixture outputs from `FAST-Python-Wrapper` while
the numerical MATLAB algorithms are ported module by module.

## Repository Layout

```text
src/fast_python/
  aircraft.py       wrapper-compatible aircraft normalization
  analysis.py       EAPAnalysis sizing/performance loop
  atmosphere.py     StdAtm and Gravity ports
  cases.py          native baseline aircraft/mission case factories
  battery.py        BatteryPkg charge/discharge, aging, and resizing
  compare.py        FAST output parity comparison helpers
  constraint.py     ConstraintDiagramPkg helper constraints
  cost.py           CostPkg battery replacement cost model
  core.py           FastPython API
  database.py       IDEAS_DB.mat loader and DatabasePkg helpers
  data_struct.py    DataStructPkg defaults, SpecProcessing, mission history
  engine.py         EngineModelPkg cycle, component, and fuel-flow helpers
  history.py        MissionHistTable mission-history table helper
  io.py             JSON loading, validation, and output writing
  main.py           command-line entry point
  markers.py        _matlab_expression and _matlab_row marker objects
  mission.py        Mission profile processing and segment evaluators
  native.py         top-level native workflow runner
  oew.py            OEWPkg operating-empty-weight iteration
  optimization.py   OptimizationPkg numerical helper routines
  plotting.py       PlotPkg plot-data formatting helpers
  profiles.py       MissionProfilesPkg preset mission definitions
  projection.py     ProjectionPkg KPP projection curves
  propulsion.py     PropulsionPkg matrix and architecture utilities
  regression.py     RegressionPkg search and GPR utilities
  reference.py      FAST-Python-Wrapper fixture backend
  retrofit.py       RetrofitPkg option helpers
  safety.py         SafetyPkg fault-tree analysis helpers
  specs.py          AircraftSpecsPkg and EngineSpecsPkg preset definitions
tests/
  test_cases.py
  test_analysis.py
  test_battery.py
  test_compare.py
  test_constraint.py
  test_cost.py
  test_data_struct.py
  test_database.py
  test_engine.py
  test_history.py
  test_io.py
  test_mission_segments.py
  test_native_utilities.py
  test_oew.py
  test_optimization.py
  test_plotting.py
  test_profiles.py
  test_projection.py
  test_propulsion.py
  test_reference_backend.py
  test_regression.py
  test_retrofit.py
  test_safety.py
  test_specs.py
```

## Requirements

- Python 3.11
- A Python 3.11 virtual environment, such as `venv` or conda
- SciPy for reading FAST's historical MATLAB database (`IDEAS_DB.mat`)
- A local MATLAB FAST checkout when running database-backed workflows or tests
  that need `+DatabasePkg/IDEAS_DB.mat`
- A `FAST-Python-Wrapper` checkout with example fixtures when running parity
  tests against the external wrapper oracle

The package can run bundled cases without the wrapper checkout. If your FAST
or wrapper checkout is outside the repository, set the relevant environment
variables:

```sh
export FAST_PATH="/path/to/FAST"
export FAST_PYTHON_WRAPPER_PATH="/path/to/FAST-Python-Wrapper"
```

PowerShell equivalent:

```powershell
$env:FAST_PATH="C:\path\to\FAST"
$env:FAST_PYTHON_WRAPPER_PATH="C:\path\to\FAST-Python-Wrapper"
```

## Install

From this repo:

```sh
python -m pip install -e .
```

## Run Tests

```sh
python -m pytest -q
```

The tests compare the Python backend against the saved wrapper fixture outputs
for `A320`, `AEA`, `ATR42`, and `CeRAS`.

To also re-test the MATLAB wrapper oracle itself:

```sh
cd /path/to/FAST-Python-Wrapper
python -m pytest -q
```

To run the optional direct MATLAB parity checks for the AircraftSpecsPkg,
MissionProfilesPkg, and MissionSegsPkg Python ports:

```sh
FAST_PYTHON_RUN_MATLAB_PARITY=1 \
FAST_PYTHON_WRAPPER_PATH=/path/to/FAST-Python-Wrapper \
FAST_PATH=/path/to/FAST \
python -m pytest -q \
  tests/test_matlab_aircraft_specs_parity.py \
  tests/test_matlab_mission_profiles_parity.py \
  tests/test_matlab_mission_segs_parity.py
```

PowerShell equivalent:

```powershell
$env:FAST_PYTHON_RUN_MATLAB_PARITY="1"
$env:FAST_PYTHON_WRAPPER_PATH="C:\path\to\FAST-Python-Wrapper"
$env:FAST_PATH="C:\path\to\FAST"
python -m pytest -q `
  tests\test_matlab_aircraft_specs_parity.py `
  tests\test_matlab_mission_profiles_parity.py `
  tests\test_matlab_mission_segs_parity.py
```

These parity tests start MATLAB through `FAST-Python-Wrapper` and are skipped
unless `FAST_PYTHON_RUN_MATLAB_PARITY=1`.
`MissionProfilesPkg.TakeoffTestProfile` is marked as an expected failure when
MATLAB lacks Aerospace Toolbox `convlength`; the native Python constants remain
covered by the ordinary profile tests.

## Input Expectations

FAST Python uses the same input file contract as `FAST-Python-Wrapper`:

- `InputAircraft.json` contains the FAST aircraft dictionary.
- `Mission.json` contains either the wrapper-style mission object with a
  top-level `Profile` dictionary, or the raw mission profile dictionary itself.
- Output is written in the same wrapper-compatible shape as
  `OutputAircraft.json`, with `OutputAircraftStructure.json` alongside it.

The default `reference` backend is fixture-based: it matches the input against
the bundled wrapper-validated cases (`A320`, `AEA`, `ATR42`, and `CeRAS`) and
returns the saved wrapper output. Those inputs must match one of the bundled
cases.

The `native` backend is the pure Python run path. It reads the same
`InputAircraft.json` and `Mission.json` shape used by `FAST-Python-Wrapper` and
does not require MATLAB or `FAST-Python-Wrapper` at runtime. It still requires
the aircraft and mission fields covered by the currently ported Python modules;
database-backed preprocessing also needs a local MATLAB FAST checkout so
`+DatabasePkg/IDEAS_DB.mat` can be found through `FAST_PATH` or
`FAST_MATLAB_PATH`.

## Run A Case

Run a bundled supported case:

```sh
python -m fast_python.main --case A320 --output-dir outputs/A320
```

You can also use any supported wrapper example input directory:

```sh
fast-python \
  --input-dir /path/to/FAST-Python-Wrapper/examples/A320/inputs \
  --output-dir outputs/A320
```

Or without installing the console script:

```sh
python -m fast_python.main \
  --input-dir /path/to/FAST-Python-Wrapper/examples/A320/inputs \
  --output-dir outputs/A320
```

For aircraft and mission inputs covered by the native port, opt into the
native backend:

```sh
python -m fast_python.main \
  --backend native \
  --input-dir inputs \
  --output-dir outputs/native
```

Or run a native Python preset pair directly:

```sh
python -m fast_python.main \
  --native-case A320 \
  --output-dir outputs/native-a320
```

Generated files:

- `outputs/A320/OutputAircraft.json`
- `outputs/A320/OutputAircraftStructure.json`

## Python API

```python
from fast_python import FastPython
from fast_python.reference import load_bundled_case_inputs

aircraft, mission = load_bundled_case_inputs("A320")

result = FastPython().run(aircraft, mission)
print(result["status"])
print(result["mtow"])
print(result["case"])
```

For workflows covered by the native port, call:

```python
from fast_python import native_case, run_native

aircraft, mission = native_case("A320")
result = run_native(aircraft, mission)
print(result["backend"])
print(result["mtow"])
```

Mission histories can be converted to PlotPkg-style data without opening
MATLAB figures:

```python
from fast_python import PlotMission

plot_data = PlotMission(result["aircraft"])
print(plot_data["figures"][0]["name"])
```

## Current Backend Coverage

The default reference backend supports the wrapper fixture cases exactly:

- `A320`
- `AEA`
- `ATR42`
- `CeRAS`

For other aircraft or missions, `FastPython.run()` raises
`UnsupportedCaseError` with a message explaining that the requested case is
outside current backend coverage.

The opt-in native backend (`run_native()` or `--backend native`) now executes
those same wrapper input cases through the ported Python modules. The native
backend has been checked against the saved `FAST-Python-Wrapper` outputs for
`A320`, `AEA`, `ATR42`, and `CeRAS` with zero comparable-field differences.
The reference backend remains the deterministic fixture backend; the native
backend is the MATLAB-free execution path for the ported workflow.

## Porting Notes

The stable public contract is:

```text
aircraft dict + mission dict -> result dict
```

The result dictionary follows the wrapper shape:

```text
{
  "status": "success",
  "mtow": <kg>,
  "aircraft": <OutputAircraft-compatible dict>,
  "log": <text>,
  "backend": "reference",
  "case": <case name>
}
```

As MATLAB algorithms are ported, new native modules should keep this contract
and use `fast_python.compare.compare_json_value()` for parity checks against
wrapper outputs.
