import numpy as np
import pytest

from conftest import FakeRegister
from qickdawg.arqick.arqick_pi_count_sweep import IntegerRegisterSweep
from qickdawg.arqick.arqick_pi_count_sweep import (
    _build_fine_timing_layout,
    _validate_common_clock,
)


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
