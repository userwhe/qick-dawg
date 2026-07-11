"""Integer pi-count sweep with fine-resolution XY microwave timing."""

import math
from dataclasses import dataclass

import numpy as np

from qick.averager_program import AbsQickSweep


def _require_python_int(name, value):
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    return value


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
