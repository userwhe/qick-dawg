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
