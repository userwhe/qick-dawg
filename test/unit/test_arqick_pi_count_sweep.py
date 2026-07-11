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
