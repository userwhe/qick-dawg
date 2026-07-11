"""Integer pi-count sweep with fine-resolution XY microwave timing."""

import math
from dataclasses import dataclass

import numpy as np

from qick.averager_program import AbsQickSweep
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram


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
