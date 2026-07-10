# QICK 200 ps XY Pi-Count Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a QICK-DAWG program that sweeps the exact integer number of individual microwave pi pulses, emits `X, Y, X, Y, ...`, supports `N = 0`, and maintains a configurable active-end-to-active-start gap on the existing approximately 200 ps timing grid.

**Architecture:** Keep the integer sweep, fine-timing geometry, and QICK pulse program in one focused `arqick` module. Test pure validation independently, then construct and compile the real program against QICK 0.2.302 with a minimal common-clock `QickConfig`; a small test VM executes the relevant `prog_list` instructions to verify counts, phases, and timing rather than relying on text matching.

**Tech Stack:** Python 3, NumPy, pytest, QICK 0.2.302 tProcessor-v1 assembly, QICK-DAWG `NVAveragerProgram`.

## Global Constraints

- Work only on branch `weitao/pi-count-sweep`, whose history must descend from `eab636e4d5a54155e03080e79e06dbec7bb22830`.
- This plan document is committed before Task 1 begins; clean-tree gates include the design and planning documentation commits.
- Keep `qick==0.2.302`; do not change package metadata or unrelated pulse programs.
- The public classes are exactly `IntegerRegisterSweep` and `PiPulseNumberSweep` in `src/qickdawg/arqick/arqick_pi_count_sweep.py`.
- `N = 0` emits the PMOD marker and no microwave pulse.
- Every shot restarts at X; pulse phases are `X, Y, X, Y, ...` for individual pulse indices.
- `pi_to_pi_delay_tdds` means active pulse end to next active pulse start, not start-to-start and not padded-envelope end to start.
- Fine timing must use the generator's `samps_per_clk`; require it to be a positive power of two and require generator/tProcessor common-clock compatibility.
- Do not put `phase` in both `default_pulse_registers()` and `set_pulse_registers()`; QICK 0.2.302 rejects overlapping keys.
- User-input failures use `TypeError` or `ValueError`, not removable `assert` statements.
- Do not run `test/local/test_nv_config.py`; it contacts laboratory hardware during import.
- The local fallback test command uses `uv --no-project` because this checkout's legacy package metadata has a case-mismatched README and must not be repaired as part of this feature.

Reference design: `docs/superpowers/specs/2026-07-10-pi-count-sweep-design.md`.

---

### Task 1: Exact Integer Register Sweep

**Files:**
- Create: `src/qickdawg/arqick/arqick_pi_count_sweep.py`
- Create: `test/unit/conftest.py`
- Create: `test/unit/test_arqick_pi_count_sweep.py`

**Interfaces:**
- Consumes: `qick.averager_program.AbsQickSweep`; a register exposing `name`, `init_val`, `reset()`, and `set_to()`.
- Produces: `IntegerRegisterSweep(prog, reg, start: int, stop: int, expts: int, label: str | None = None)` with integer `step`, `get_sweep_pts() -> numpy.ndarray`, `reset()`, and `update()`.

- [ ] **Step 1: Add the fake register and failing integer-sweep tests**

Create `test/unit/conftest.py` with:

```python
class FakeRegister:
    def __init__(self, name="n_pi"):
        self.name = name
        self.init_val = None
        self.reset_calls = 0
        self.set_calls = []

    def reset(self):
        self.reset_calls += 1

    def set_to(self, *args, **kwargs):
        self.set_calls.append((args, kwargs))
```

Create `test/unit/test_arqick_pi_count_sweep.py` with:

```python
import numpy as np
import pytest

from conftest import FakeRegister
from qickdawg.arqick.arqick_pi_count_sweep import IntegerRegisterSweep


@pytest.mark.parametrize(
    ("start", "stop", "expts", "expected"),
    [
        (0, 6, 4, [0, 2, 4, 6]),
        (1, 9, 3, [1, 5, 9]),
    ],
)
def test_integer_register_sweep_points(start, stop, expts, expected):
    register = FakeRegister()
    sweep = IntegerRegisterSweep(object(), register, start, stop, expts)

    points = sweep.get_sweep_pts()

    assert points.tolist() == expected
    assert np.issubdtype(points.dtype, np.integer)
    assert sweep.step == expected[1] - expected[0]
    assert register.init_val == start
    assert sweep.label == "n_pi"


def test_integer_register_sweep_reset_and_raw_update():
    register = FakeRegister()
    sweep = IntegerRegisterSweep(
        object(), register, 0, 6, 4, label="pulse_count"
    )

    sweep.reset()
    sweep.update()

    assert register.reset_calls == 1
    assert register.set_calls == [
        ((register, "+", 2), {"physical_unit": False})
    ]
    assert sweep.label == "pulse_count"


@pytest.mark.parametrize(
    ("start", "stop", "expts", "exception"),
    [
        (0.0, 4, 5, TypeError),
        (False, 4, 5, TypeError),
        (0, 4.0, 5, TypeError),
        (0, 4, 5.0, TypeError),
        (-1, 4, 6, ValueError),
        (4, 0, 5, ValueError),
        (0, 5, 4, ValueError),
        (0, 4, 1, ValueError),
        (2, 2, 2, ValueError),
    ],
)
def test_integer_register_sweep_rejects_invalid_ranges(
    start, stop, expts, exception
):
    with pytest.raises(exception):
        IntegerRegisterSweep(
            object(), FakeRegister(), start, stop, expts
        )
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest test/unit/test_arqick_pi_count_sweep.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'qickdawg.arqick.arqick_pi_count_sweep'`.

- [ ] **Step 3: Implement the integer-only sweep**

Create `src/qickdawg/arqick/arqick_pi_count_sweep.py` with:

```python
"""Integer pi-count sweep with fine-resolution XY microwave timing."""

import numpy as np

from qick.averager_program import AbsQickSweep


def _require_python_int(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    return value


class IntegerRegisterSweep(AbsQickSweep):
    """Sweep a raw QICK register over an exact nonnegative integer axis."""

    def __init__(self, prog, reg, start, stop, expts, label=None):
        start = _require_python_int("start", start)
        stop = _require_python_int("stop", stop)
        expts = _require_python_int("expts", expts)

        if start < 0:
            raise ValueError("start must be nonnegative")
        if stop < start:
            raise ValueError("stop must be greater than or equal to start")
        if expts < 2:
            raise ValueError("expts must be at least 2")

        span = stop - start
        intervals = expts - 1
        if span % intervals:
            raise ValueError(
                "integer sweep endpoints must be divisible by expts - 1"
            )
        step = span // intervals
        if step <= 0:
            raise ValueError("integer sweep step must be positive")

        resolved_label = reg.name if label is None else label
        super().__init__(prog, resolved_label)
        self.reg = reg
        self.start = start
        self.stop = stop
        self.expts = expts
        self.step = step
        self.label = resolved_label
        self.reg.init_val = start

    def reset(self):
        self.reg.reset()

    def update(self):
        self.reg.set_to(
            self.reg, "+", self.step, physical_unit=False
        )

    def get_sweep_pts(self):
        return self.start + self.step * np.arange(
            self.expts, dtype=np.int64
        )
```

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest test/unit/test_arqick_pi_count_sweep.py -q
```

Expected: all integer-sweep tests pass.

- [ ] **Step 5: Commit the independently testable sweep**

```bash
git add \
  src/qickdawg/arqick/arqick_pi_count_sweep.py \
  test/unit/conftest.py \
  test/unit/test_arqick_pi_count_sweep.py
git commit -m "feat: add exact integer QICK sweep"
```

---

### Task 2: Fine-Timing Geometry and Clock Guards

**Files:**
- Modify: `src/qickdawg/arqick/arqick_pi_count_sweep.py`
- Modify: `test/unit/test_arqick_pi_count_sweep.py`

**Interfaces:**
- Consumes: exact Python integer pulse duration, gap, and `samps_per_clk`; a QICK configuration mapping containing `gens[mw_channel].f_fabric` and `tprocs[0].f_time`.
- Produces: `_FineTimingLayout`, `_build_fine_timing_layout(mw_pi_tdds: int, gap_tdds: int, samps_per_clk: int) -> _FineTimingLayout`, and `_validate_common_clock(soccfg, mw_channel: int) -> None`.

- [ ] **Step 1: Add failing layout and clock-validation tests**

Append these imports and tests to `test/unit/test_arqick_pi_count_sweep.py`:

```python
from qickdawg.arqick.arqick_pi_count_sweep import (
    _build_fine_timing_layout,
    _validate_common_clock,
)


def test_fine_timing_layout_accounts_for_maximum_shift():
    layout = _build_fine_timing_layout(190, 500, 16)

    assert layout.samps_per_clk == 16
    assert layout.log2_samps_per_clk == 4
    assert layout.waveform_len_treg == 13
    assert layout.waveform_len_tdds == 208
    assert layout.unused_tail_tdds == 18
    assert layout.interpulse_stride_tdds == 482


@pytest.mark.parametrize(
    ("duration", "gap", "samps", "exception"),
    [
        (0, 500, 16, ValueError),
        (190, -1, 16, ValueError),
        (190, 17, 16, ValueError),
        (190, 500, 0, ValueError),
        (190, 500, 12, ValueError),
        (190.0, 500, 16, TypeError),
    ],
)
def test_fine_timing_layout_rejects_invalid_geometry(
    duration, gap, samps, exception
):
    with pytest.raises(exception):
        _build_fine_timing_layout(duration, gap, samps)


def test_common_clock_validation_accepts_lab_clock():
    soccfg = {
        "gens": [{"f_fabric": 307.2}],
        "tprocs": [{"f_time": 307.2}],
    }

    _validate_common_clock(soccfg, 0)


@pytest.mark.parametrize(
    "soccfg",
    [
        {
            "gens": [{"f_fabric": 250.0}],
            "tprocs": [{"f_time": 307.2}],
        },
        {
            "gens": [{"f_fabric": 0.0}],
            "tprocs": [{"f_time": 0.0}],
        },
    ],
)
def test_common_clock_validation_rejects_incompatible_firmware(soccfg):
    with pytest.raises(ValueError, match="common clock"):
        _validate_common_clock(soccfg, 0)
```

- [ ] **Step 2: Run the focused tests and confirm the missing-symbol failure**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest test/unit/test_arqick_pi_count_sweep.py -q
```

Expected: collection fails because `_build_fine_timing_layout` and `_validate_common_clock` do not exist.

- [ ] **Step 3: Implement exact integer timing geometry**

Add `math` and `dataclass` imports near the top of the production module:

```python
import math
from dataclasses import dataclass
```

Add the following immediately after `_require_python_int`:

```python
@dataclass(frozen=True)
class _FineTimingLayout:
    samps_per_clk: int
    log2_samps_per_clk: int
    waveform_len_treg: int
    waveform_len_tdds: int
    unused_tail_tdds: int
    interpulse_stride_tdds: int


def _build_fine_timing_layout(
    mw_pi_tdds, gap_tdds, samps_per_clk
):
    mw_pi_tdds = _require_python_int("mw_pi_tdds", mw_pi_tdds)
    gap_tdds = _require_python_int("pi_to_pi_delay_tdds", gap_tdds)
    samps_per_clk = _require_python_int(
        "samps_per_clk", samps_per_clk
    )

    if mw_pi_tdds <= 0:
        raise ValueError("mw_pi_tdds must be positive")
    if gap_tdds < 0:
        raise ValueError("pi_to_pi_delay_tdds must be nonnegative")
    if samps_per_clk <= 0 or samps_per_clk & (samps_per_clk - 1):
        raise ValueError("samps_per_clk must be a positive power of two")

    waveform_len_treg = max(
        (mw_pi_tdds + 2 * samps_per_clk - 2) // samps_per_clk,
        3,
    )
    waveform_len_tdds = waveform_len_treg * samps_per_clk
    unused_tail_tdds = waveform_len_tdds - mw_pi_tdds
    if gap_tdds < unused_tail_tdds:
        raise ValueError(
            "pi_to_pi_delay_tdds is too short for the padded waveform: "
            f"got {gap_tdds}, need at least {unused_tail_tdds}"
        )

    return _FineTimingLayout(
        samps_per_clk=samps_per_clk,
        log2_samps_per_clk=samps_per_clk.bit_length() - 1,
        waveform_len_treg=waveform_len_treg,
        waveform_len_tdds=waveform_len_tdds,
        unused_tail_tdds=unused_tail_tdds,
        interpulse_stride_tdds=gap_tdds - unused_tail_tdds,
    )


def _validate_common_clock(soccfg, mw_channel):
    mw_channel = _require_python_int("mw_channel", mw_channel)
    try:
        generator_mhz = float(soccfg["gens"][mw_channel]["f_fabric"])
        tprocessor_mhz = float(soccfg["tprocs"][0]["f_time"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "soccfg does not expose generator and tProcessor clocks"
        ) from exc

    if (
        generator_mhz <= 0
        or tprocessor_mhz <= 0
        or not math.isclose(
            generator_mhz,
            tprocessor_mhz,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "the 200 ps pi-count program requires a common clock: "
            f"generator={generator_mhz} MHz, "
            f"tProcessor={tprocessor_mhz} MHz"
        )
```

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest test/unit/test_arqick_pi_count_sweep.py -q
```

Expected: all integer-sweep, layout, and clock tests pass.

- [ ] **Step 5: Commit the timing boundary**

```bash
git add \
  src/qickdawg/arqick/arqick_pi_count_sweep.py \
  test/unit/test_arqick_pi_count_sweep.py
git commit -m "feat: validate pi-count fine timing"
```

---

### Task 3: Compile and Execute the XY Pi-Count Assembly Model

**Files:**
- Modify: `src/qickdawg/arqick/arqick_pi_count_sweep.py`
- Modify: `test/unit/conftest.py`
- Modify: `test/unit/test_arqick_pi_count_sweep.py`

**Interfaces:**
- Consumes: `IntegerRegisterSweep`, `_build_fine_timing_layout`, `_validate_common_clock`, QICK 0.2.302 `QickConfig`, and QICK-DAWG `NVAveragerProgram`.
- Produces: `PiPulseNumberSweep(cfg)` with one integer sweep axis; `pi_0` through `pi_{samps_per_clk - 1}` envelopes; one PMOD marker per point; exactly N alternating XY pulses; and no inter-pulse delay after the final pulse.

- [ ] **Step 1: Expand the host fixtures for a real QICK 0.2.302 compile**

Replace `test/unit/conftest.py` with:

```python
import pytest
from qick import QickConfig


class FakeRegister:
    def __init__(self, name="n_pi"):
        self.name = name
        self.init_val = None
        self.reset_calls = 0
        self.set_calls = []

    def reset(self):
        self.reset_calls += 1

    def set_to(self, *args, **kwargs):
        self.set_calls.append((args, kwargs))


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def soccfg_factory():
    def make(*, samps_per_clk=16, f_fabric=307.2, f_time=307.2):
        return QickConfig(
            {
                "board": "RFSoC4x2",
                "sw_version": "0.2.302",
                "fw_timestamp": "host-test",
                "gens": [
                    {
                        "type": "axis_signal_gen_v6",
                        "tproc_ch": 0,
                        "samps_per_clk": samps_per_clk,
                        "f_fabric": f_fabric,
                        "has_mixer": False,
                        "has_dds": True,
                        "complex_env": True,
                        "maxv": 32767,
                        "maxv_scale": 1.0,
                        "b_phase": 32,
                        "maxlen": 65536,
                    }
                ],
                "readouts": [],
                "tprocs": [
                    {
                        "type": "axis_tproc64x32_x8",
                        "revision": 1,
                        "pmem_size": 4096,
                        "dmem_size": 4096,
                        "f_time": f_time,
                        "output_pins": [
                            ("output", 7, 0, "PMOD0_0")
                        ],
                        "start_pin": "external",
                    }
                ],
                "iqs": [],
                "extra_description": [],
            }
        )

    return make


@pytest.fixture
def config_factory():
    def make(**overrides):
        values = {
            "mw_pi_tdds": 32,
            "pi_to_pi_delay_tdds": 25,
            "n_pi_start": 0,
            "n_pi_end": 3,
            "nsweep_points": 4,
            "freq_freg": 0,
            "mw_channel": 0,
            "mw_nqz": 1,
            "mw_gain": 1000,
            "reps": 1,
            "pmod_out_pin": 0,
            "pmod_out_pulse_width_treg": 10,
            "pmod_out_trig_delay_treg": 0,
            "inherent_trigger_to_pulses_delay_treg": 0,
            "pulse_seq_delay_treg": 0,
        }
        values.update(overrides)
        return AttrDict(values)

    return make
```

- [ ] **Step 2: Add failing compile, envelope, validation, and assembly-execution tests**

Append the following to `test/unit/test_arqick_pi_count_sweep.py`:

```python
from collections import defaultdict
import operator

import qickdawg

from qickdawg.arqick.arqick_pi_count_sweep import PiPulseNumberSweep


def _make_program(monkeypatch, soccfg, config):
    monkeypatch.setattr(qickdawg, "soccfg", soccfg, raising=False)
    return PiPulseNumberSweep(config)


def _run_pi_body(program, n_pi):
    labels = {
        inst["label"]: index
        for index, inst in enumerate(program.prog_list)
        if "label" in inst
    }
    start_index = labels["LOOP_n_pi"]
    end_index = labels["PI_LOOP_END"]
    registers = defaultdict(int)
    registers[(program.n_pi_register.page, program.n_pi_register.addr)] = n_pi
    registers[
        (program.phase_step_register.page, program.phase_step_register.addr)
    ] = 1
    registers[
        (program.tdds_offset_register.page, program.tdds_offset_register.addr)
    ] = 7
    registers[
        (program.treg_offset_register.page, program.treg_offset_register.addr)
    ] = 3
    registers[
        (program.address_register.page, program.address_register.addr)
    ] = 5 * program.pi_waveform_len_treg
    time_tdds = 0
    pulses = []
    dynamic_syncs = 0
    pc = start_index
    steps = 0

    comparisons = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
    }
    arithmetic = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
    }
    bitwise = {
        "&": operator.and_,
        "|": operator.or_,
        "^": operator.xor,
        "<<": operator.lshift,
        ">>": operator.rshift,
    }

    while True:
        steps += 1
        if steps > 500:
            raise AssertionError("pi loop did not terminate")

        inst = program.prog_list[pc]
        name = inst["name"]
        args = inst["args"]
        next_pc = pc + 1

        if name == "regwi":
            page, addr, value = args[:3]
            registers[(page, addr)] = value
        elif name == "mathi":
            page, dst, src, op, value = args[:5]
            registers[(page, dst)] = arithmetic[op](
                registers[(page, src)], value
            )
        elif name == "bitwi":
            page, dst, src, op, value = args[:5]
            registers[(page, dst)] = bitwise[op](
                registers[(page, src)], value
            )
        elif name == "condj":
            page, left, op, right, label = args[:5]
            if comparisons[op](
                registers[(page, left)], registers[(page, right)]
            ):
                next_pc = labels[label]
        elif name == "set":
            page = args[1]
            phase = registers[(page, args[3])]
            address = registers[(page, args[4])]
            fine_offset = address // program.pi_waveform_len_treg
            pulses.append((time_tdds + fine_offset, phase))
        elif name == "synci":
            time_tdds += args[0] * program.samps_per_clk
        elif name == "sync":
            page, addr = args[:2]
            time_tdds += (
                registers[(page, addr)] * program.samps_per_clk
            )
            dynamic_syncs += 1
        elif name in {"seti", "comment"}:
            pass
        else:
            raise AssertionError(
                f"unexpected instruction inside pi body: {name}"
            )

        if pc == end_index:
            break
        pc = next_pc

    return pulses, dynamic_syncs


def test_program_compiles_on_qick_0_2_302(
    monkeypatch, soccfg_factory, config_factory
):
    program = _make_program(
        monkeypatch, soccfg_factory(), config_factory()
    )

    program.compile()

    assert program.binprog
    assert program.get_expt_pts()[0].tolist() == [0, 1, 2, 3]
    assert np.issubdtype(program.get_expt_pts()[0].dtype, np.integer)

    manager = program._gen_mgrs[0]
    assert set(manager.defaults).isdisjoint(manager.last_set_regs)
    assert "phase" not in manager.defaults
    assert manager.last_set_regs["waveform"] == "pi_0"


def test_shifted_envelopes_cover_every_fine_offset(
    monkeypatch, soccfg_factory, config_factory
):
    program = _make_program(
        monkeypatch,
        soccfg_factory(),
        config_factory(mw_pi_tdds=190, pi_to_pi_delay_tdds=500),
    )

    envelopes = program.envelopes[0]["envs"]
    assert set(envelopes) == {f"pi_{index}" for index in range(16)}
    for offset in range(16):
        envelope = envelopes[f"pi_{offset}"]
        data = envelope["data"]
        assert data.shape == (208, 2)
        assert np.flatnonzero(data[:, 0]).tolist() == list(
            range(offset, offset + 190)
        )
        assert np.flatnonzero(data[:, 1]).tolist() == list(
            range(offset, offset + 190)
        )
        assert envelope["addr"] // 16 == offset * 13


@pytest.mark.parametrize(
    ("samps", "f_fabric", "f_time", "gap", "message"),
    [
        (12, 307.2, 307.2, 25, "power of two"),
        (16, 250.0, 307.2, 25, "common clock"),
        (16, 307.2, 307.2, 15, "too short"),
    ],
)
def test_program_rejects_unsupported_firmware_or_gap(
    monkeypatch,
    soccfg_factory,
    config_factory,
    samps,
    f_fabric,
    f_time,
    gap,
    message,
):
    with pytest.raises(ValueError, match=message):
        _make_program(
            monkeypatch,
            soccfg_factory(
                samps_per_clk=samps,
                f_fabric=f_fabric,
                f_time=f_time,
            ),
            config_factory(pi_to_pi_delay_tdds=gap),
        )


@pytest.mark.parametrize(
    ("n_pi", "expected_phases"),
    [
        (0, []),
        (1, [0]),
        (2, [0, 1 << 30]),
        (3, [0, 1 << 30, 0]),
        (4, [0, 1 << 30, 0, 1 << 30]),
    ],
)
def test_assembly_emits_exact_count_xy_phase_and_no_final_gap(
    monkeypatch,
    soccfg_factory,
    config_factory,
    n_pi,
    expected_phases,
):
    program = _make_program(
        monkeypatch,
        soccfg_factory(),
        config_factory(n_pi_end=4, nsweep_points=5),
    )

    pulses, dynamic_syncs = _run_pi_body(program, n_pi)

    assert [phase for _, phase in pulses] == expected_phases
    assert len(pulses) == n_pi
    assert dynamic_syncs == max(n_pi - 1, 0)


def test_assembly_gap_is_active_end_to_active_start_across_carry(
    monkeypatch, soccfg_factory, config_factory
):
    program = _make_program(
        monkeypatch,
        soccfg_factory(),
        config_factory(
            n_pi_end=4,
            nsweep_points=5,
            mw_pi_tdds=32,
            pi_to_pi_delay_tdds=25,
        ),
    )

    pulses, _ = _run_pi_body(program, 4)
    starts = [start for start, _ in pulses]

    assert starts == [starts[0] + 57 * index for index in range(4)]


def test_body_contains_one_pmod_marker_pair(
    monkeypatch, soccfg_factory, config_factory
):
    program = _make_program(
        monkeypatch, soccfg_factory(), config_factory()
    )
    labels = {
        inst["label"]: index
        for index, inst in enumerate(program.prog_list)
        if "label" in inst
    }
    body = program.prog_list[
        labels["LOOP_n_pi"] : labels["PI_LOOP_END"] + 1
    ]

    assert sum(inst["name"] == "seti" for inst in body) == 2
```

- [ ] **Step 3: Run the tests and confirm the missing-class failure**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest test/unit/test_arqick_pi_count_sweep.py -q
```

Expected: collection fails because `PiPulseNumberSweep` is not defined.

- [ ] **Step 4: Implement the full QICK program**

Add this import beside the existing QICK import:

```python
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
```

Append this class to `src/qickdawg/arqick/arqick_pi_count_sweep.py`:

```python
class PiPulseNumberSweep(NVAveragerProgram):
    """Sweep a bare XYXY pi-pulse train on the fine DAC timing grid."""

    required_cfg = [
        "mw_pi_tdds",
        "pi_to_pi_delay_tdds",
        "n_pi_start",
        "n_pi_end",
        "nsweep_points",
        "freq_freg",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "reps",
        "pmod_out_pin",
        "pmod_out_pulse_width_treg",
        "pmod_out_trig_delay_treg",
        "inherent_trigger_to_pulses_delay_treg",
        "pulse_seq_delay_treg",
    ]

    def initialize(self):
        missing = [name for name in self.required_cfg if name not in self.cfg]
        if missing:
            raise ValueError(
                "missing required pi-count configuration fields: "
                + ", ".join(missing)
            )

        mw_channel = _require_python_int(
            "mw_channel", self.cfg.mw_channel
        )
        _validate_common_clock(self.soccfg, mw_channel)
        generator_cfg = self.soccfg["gens"][mw_channel]
        layout = _build_fine_timing_layout(
            self.cfg.mw_pi_tdds,
            self.cfg.pi_to_pi_delay_tdds,
            generator_cfg["samps_per_clk"],
        )
        phase_bits = _require_python_int(
            "b_phase", generator_cfg["b_phase"]
        )
        if phase_bits < 2:
            raise ValueError("generator b_phase must be at least 2")

        self.declare_gen(ch=mw_channel, nqz=self.cfg.mw_nqz)
        self.samps_per_clk = layout.samps_per_clk
        self.log2_samps_per_clk = layout.log2_samps_per_clk
        self.pi_waveform_len_treg = layout.waveform_len_treg
        self.pi_waveform_len_tdds = layout.waveform_len_tdds
        self.pi_len_unused_tdds = layout.unused_tail_tdds
        self.interpulse_stride_tdds = layout.interpulse_stride_tdds
        self.phase_shift = phase_bits - 2

        maxv = self.soccfg.get_maxv(mw_channel)
        for offset in range(self.samps_per_clk):
            idata = np.zeros(self.pi_waveform_len_tdds)
            qdata = np.zeros(self.pi_waveform_len_tdds)
            active = slice(offset, offset + self.cfg.mw_pi_tdds)
            idata[active] = maxv
            qdata[active] = maxv
            self.add_envelope(
                ch=mw_channel,
                name=f"pi_{offset}",
                idata=idata,
                qdata=qdata,
            )

        self.default_pulse_registers(
            ch=mw_channel,
            style="arb",
            freq=self.cfg.freq_freg,
            gain=self.cfg.mw_gain,
        )
        self.set_pulse_registers(
            ch=mw_channel,
            waveform="pi_0",
            phase=0,
        )

        self.address_register = self.get_gen_reg(mw_channel, name="addr")
        self.phase_register = self.get_gen_reg(mw_channel, name="phase")
        self.n_pi_register = self.new_gen_reg(
            mw_channel, name="n_pi", init_val=self.cfg.n_pi_start
        )
        self.pi_counter_register = self.new_gen_reg(
            mw_channel, name="pi_counter", init_val=0
        )
        self.tdds_offset_register = self.new_gen_reg(
            mw_channel, name="tdds_offset", init_val=0
        )
        self.treg_offset_register = self.new_gen_reg(
            mw_channel, name="treg_offset", init_val=0
        )
        self.phase_step_register = self.new_gen_reg(
            mw_channel, name="phase_step", init_val=0
        )

        self.add_sweep(
            IntegerRegisterSweep(
                self,
                self.n_pi_register,
                self.cfg.n_pi_start,
                self.cfg.n_pi_end,
                self.cfg.nsweep_points,
                label="n_pi",
            )
        )
        self.synci(200)

    def body(self):
        self.sync_all(
            self.cfg.inherent_trigger_to_pulses_delay_treg
        )
        self.trigger(
            pins=[self.cfg.pmod_out_pin],
            width=self.cfg.pmod_out_pulse_width_treg,
        )
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.pi_counter_register.set_to(
            self.n_pi_register, "+", 0, physical_unit=False
        )
        self.tdds_offset_register.reset()
        self.treg_offset_register.reset()
        self.phase_step_register.reset()
        self.address_register.set_to(0, physical_unit=False)

        self.condj(
            self.pi_counter_register.page,
            self.pi_counter_register.addr,
            "==",
            0,
            "PI_LOOP_END",
        )

        self.label("PI_LOOP")
        self.bitwi(
            self.phase_register.page,
            self.phase_register.addr,
            self.phase_step_register.addr,
            "<<",
            self.phase_shift,
        )
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.pi_counter_register.set_to(
            self.pi_counter_register, "-", 1, physical_unit=False
        )
        self.condj(
            self.pi_counter_register.page,
            self.pi_counter_register.addr,
            "==",
            0,
            "PI_LOOP_END",
        )

        self._configure_next_pi_start()
        self.phase_step_register.set_to(
            self.phase_step_register, "+", 1, physical_unit=False
        )
        self.bitwi(
            self.phase_step_register.page,
            self.phase_step_register.addr,
            self.phase_step_register.addr,
            "&",
            1,
        )
        self.condj(
            self.pi_counter_register.page,
            self.pi_counter_register.addr,
            ">",
            0,
            "PI_LOOP",
        )

        self.label("PI_LOOP_END")
        self.synci(self.cfg.pulse_seq_delay_treg)

    def _configure_next_pi_start(self):
        self.tdds_offset_register.set_to(
            self.tdds_offset_register,
            "+",
            self.interpulse_stride_tdds,
            physical_unit=False,
        )
        self.bitwi(
            self.tdds_offset_register.page,
            self.treg_offset_register.addr,
            self.tdds_offset_register.addr,
            ">>",
            self.log2_samps_per_clk,
        )
        self.bitwi(
            self.tdds_offset_register.page,
            self.tdds_offset_register.addr,
            self.tdds_offset_register.addr,
            "&",
            self.samps_per_clk - 1,
        )
        self.address_register.set_to(
            self.tdds_offset_register,
            "*",
            self.pi_waveform_len_treg,
            physical_unit=False,
        )
        self.sync(
            self.treg_offset_register.page,
            self.treg_offset_register.addr,
        )
```

- [ ] **Step 5: Run the real compile and assembly-model tests**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest test/unit/test_arqick_pi_count_sweep.py -q
```

Expected: all tests pass; `program.compile()` populates `binprog`; the model reports 0, 1, 2, 3, and 4 pulses with phases `[]`, `X`, `XY`, `XYX`, and `XYXY`; every start-to-start delta is `mw_pi_tdds + pi_to_pi_delay_tdds`.

- [ ] **Step 6: Inspect the generated assembly once for readable labels and register operations**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest \
  test/unit/test_arqick_pi_count_sweep.py::test_program_compiles_on_qick_0_2_302 \
  -q
```

Expected: one test passes. If debugging is required, print `program.asm()` locally and confirm labels `LOOP_n_pi`, `PI_LOOP`, and `PI_LOOP_END`; do not commit diagnostic printing.

- [ ] **Step 7: Commit the pulse program**

```bash
git add \
  src/qickdawg/arqick/arqick_pi_count_sweep.py \
  test/unit/conftest.py \
  test/unit/test_arqick_pi_count_sweep.py
git commit -m "feat: add 200 ps XY pi-count sweep"
```

---

### Task 4: QICK Branch Acceptance and Handoff

**Files:**
- Verify: `src/qickdawg/arqick/arqick_pi_count_sweep.py`
- Verify: `test/unit/conftest.py`
- Verify: `test/unit/test_arqick_pi_count_sweep.py`

**Interfaces:**
- Consumes: the three commits from Tasks 1-3.
- Produces: a clean, locally verified QICK feature commit that the dqpcontrol integration plan can pin after it is published to `userwhe/qick-dawg`.

- [ ] **Step 1: Verify branch ancestry and exact dependency version**

```bash
test "$(git branch --show-current)" = "weitao/pi-count-sweep"
git merge-base --is-ancestor \
  eab636e4d5a54155e03080e79e06dbec7bb22830 HEAD
rg -n 'qick==0\.2\.302' pyproject.toml
```

Expected: both `test`/`merge-base` commands return zero and `rg` shows the existing QICK pin.

- [ ] **Step 2: Run the complete isolated host suite**

Run:

```bash
PYTHONPATH=src uv run --no-project \
  --with 'qick==0.2.302' --with numpy --with pytest \
  --with tqdm --with itemattribute --with matplotlib \
  --with scipy --with pyro4 --with ipython \
  -- python -m pytest test/unit/test_arqick_pi_count_sweep.py -q
```

Expected: all tests in `test/unit/test_arqick_pi_count_sweep.py` pass. No test attempts a network or hardware connection.

- [ ] **Step 3: Run static repository checks**

```bash
python3 -m compileall -q \
  src/qickdawg/arqick/arqick_pi_count_sweep.py \
  test/unit/conftest.py \
  test/unit/test_arqick_pi_count_sweep.py
git diff --check
git status --short --branch
```

Expected: compilation and `git diff --check` succeed; status shows branch `weitao/pi-count-sweep` and no uncommitted files.

- [ ] **Step 4: Record the immutable handoff commit**

```bash
git rev-parse HEAD
git log --oneline \
  eab636e4d5a54155e03080e79e06dbec7bb22830..HEAD
```

Expected: output includes the design commit and the three feature commits. Copy the full `git rev-parse HEAD` value into the dqpcontrol submodule task; the gitlink, not the branch name, is the reproducible dependency.

## Hardware Acceptance Gate

Host tests cannot establish analog pulse shape or real trigger latency. Before deployment, use the linked dqpcontrol wrapper with a short `N = 0, 1, 2, 3, 4` sweep and verify on a scope or phase-sensitive loopback:

1. one PMOD marker at every point;
2. no microwave output for `N = 0`;
3. exactly N active microwave pulses;
4. phase prefixes `X`, `XY`, `XYX`, and `XYXY`;
5. measured active-end-to-active-start gaps at a value that crosses a coarse-cycle boundary; and
6. optical readout alignment at the first and final sweep points.
