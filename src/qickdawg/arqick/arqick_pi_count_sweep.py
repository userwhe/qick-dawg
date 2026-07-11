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
