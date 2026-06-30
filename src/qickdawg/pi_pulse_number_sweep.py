'''
Fine-Resolution Pi Pulse Number Sweep
=======================================================================
Sweeps the number of pi pulses in a CPMG-like train without the initial
or final pi/2 pulses.

Sequence structure:
    pi_X -> tau -> pi_Y -> tau -> pi_X -> ...

The microwave timing uses the fine-resolution ftsamp method from the
fine timing suite: a coarse tProc delay plus waveform-addressed sample
offsets for sub-clock pulse placement.
'''

from .nvpulsing.nvaverageprogram import NVAveragerProgram
from .nvpulsing.nvqicksweep import NVQickSweep
from .finetimingsuite.readout_helpers import ReadoutHelpers

import numpy as np


class _IntegerNVQickSweep(NVQickSweep):
    """NVQickSweep variant with integer sweep points and integer updates."""

    def __init__(self, prog, reg, start, stop, expts, label=None):
        super().__init__(prog, reg, start, stop, expts, label=label)
        self.step_val = int((stop - start) // (expts - 1))

    def get_sweep_pts(self):
        return self.start + self.step_val * np.arange(self.expts, dtype=int)


class PiPulseNumberSweep(ReadoutHelpers, NVAveragerProgram):
    '''
    Fine-resolution pi-pulse-number sweep.

    This program sweeps the number of individual pi pulses, alternating
    each pi pulse between X and Y phase. It intentionally omits the pi/2
    pulses used by CPMG/XY coherence programs.
    '''
    required_cfg = [
        # Hardware channels
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod",

        # MW pulse parameters
        "mw_pi2_ftsamp",
        "mw_nqz",
        "mw_freg",
        "mw_gain",

        # Pi-pulse count sweep
        "n_start",
        "n_end",
        "nsweep_points",

        # Fine inter-pulse timing
        "tau_ftsamp",

        # Timing delays
        "mw_to_laser_delay_treg",
        "relax_delay_treg",

        # Readout timing
        "laser_on_treg",
        "readout_reference_start_treg",
        "readout_integration_treg",
        "laser_readout_offset_treg",

        # Experiment control
        "reps",
        "pre_init",
        "get_reference",
    ]

    def initialize(self):
        """
        Prepare pi waveforms, registers, phase cycling, and the n sweep.
        """
        self.check_cfg()
        self.validate_cfg()

        if self.cfg.mw_gain < 0:
            raise ValueError("Smallest Microwave gain must be positive")
        if self.cfg.mw_gain > 32767:
            raise ValueError("Largest Microwave gain exceeds maximum value")

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.setup_helper_registers(self.cfg.mw_channel)
        self.setup_readout()

        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        if self.samps_per_clk & (self.samps_per_clk - 1):
            raise ValueError("Fine timing requires samps_per_clk to be a power of two")

        # Waveform buffer length must be >= 3 treg.
        self.pi_waveform_len_treg = max(
            int(np.ceil((self.cfg.mw_pi2_ftsamp * 2 + (self.samps_per_clk - 1)) / self.samps_per_clk)),
            3,
        )
        self.pi_waveform_len_ftsamp = self.pi_waveform_len_treg * self.samps_per_clk

        for i in range(self.samps_per_clk):
            data = np.zeros(self.pi_waveform_len_ftsamp)
            data[i: i + self.cfg.mw_pi2_ftsamp * 2] = 1
            data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=data, qdata=data)

        self.pi_len_unused = self.pi_waveform_len_ftsamp - self.cfg.mw_pi2_ftsamp * 2
        if max(self.cfg.n_start, self.cfg.n_end) > 1 and self.cfg.tau_ftsamp < self.pi_len_unused:
            raise ValueError(
                "tau_ftsamp is too short for fine-resolution pi pulse spacing: "
                f"tau_ftsamp={self.cfg.tau_ftsamp} must be >= "
                f"pi waveform deadtime {self.pi_len_unused}"
            )

        self.delay_coarse_cycles = self.new_gen_reg(
            self.cfg.mw_channel,
            name='treg_offset',
            init_val=0,
        )
        self.delay_fine_samples = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ftsamp_offset',
            init_val=0,
        )
        self.tau = self.new_gen_reg(
            self.cfg.mw_channel,
            name='tau',
            init_val=self.cfg.tau_ftsamp,
        )
        self.n_total_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='n_total',
            init_val=self.cfg.n_start,
        )
        self.n_loop_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='n_loop',
            init_val=0,
        )
        self.pulse_index_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='pulse_index',
            init_val=0,
        )

        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style='arb',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            phase=0,
        )

        self.add_sweep(_IntegerNVQickSweep(
            self,
            self.n_total_register,
            self.cfg.n_start,
            self.cfg.n_end,
            self.cfg.nsweep_points,
            label='n_pulses',
        ))

        self.address_reg = self.get_gen_reg(self.cfg.mw_channel, name='addr')
        self.phase_reg = self.get_gen_reg(self.cfg.mw_channel, name='phase')

        self.pre_init()

    def validate_cfg(self):
        """
        Validate the unitless n sweep before constructing QICK sweep objects.
        """
        for key in ("n_start", "n_end", "nsweep_points", "mw_pi2_ftsamp", "tau_ftsamp"):
            if not isinstance(self.cfg[key], (int, np.integer)):
                raise TypeError(f"config.{key} must be an integer")

        if self.cfg.n_start < 0 or self.cfg.n_end < 0:
            raise ValueError("config.n_start and config.n_end must be >= 0")
        if self.cfg.nsweep_points < 2:
            raise ValueError("config.nsweep_points must be >= 2")

        span = self.cfg.n_end - self.cfg.n_start
        steps = self.cfg.nsweep_points - 1
        if span % steps != 0:
            raise ValueError(
                "Pi-pulse number sweep must have an integer step: "
                f"({self.cfg.n_end} - {self.cfg.n_start}) / {steps}"
            )
        if span == 0:
            raise ValueError("config.n_start and config.n_end must differ when nsweep_points >= 2")

        if self.cfg.mw_pi2_ftsamp <= 0:
            raise ValueError("config.mw_pi2_ftsamp must be > 0")
        if self.cfg.tau_ftsamp < 0:
            raise ValueError("config.tau_ftsamp must be >= 0")

    def body(self):
        """
        One repetition: initialize spin, run the pi train, then read out.
        """
        self.initialize_spin()
        self.program_pulses(1)
        self.readout_and_reference(lambda: self.program_pulses(2))

    def program_pulses(self, label_id):
        """
        Program one pi-pulse train for the current swept n value.
        """
        self.delay_fine_samples.reset()
        self.pulse_index_register.reset()
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0")

        self.condj(
            self.n_total_register.page,
            self.n_total_register.addr,
            "==",
            0,
            f"JUMP_END_PI_{label_id}",
        )

        # First pi pulse starts immediately, with X phase.
        self.select_phase()
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
        self.mathi(
            self.pulse_index_register.page,
            self.pulse_index_register.addr,
            self.pulse_index_register.addr,
            "+",
            1,
        )

        self.condj(
            self.n_total_register.page,
            self.n_total_register.addr,
            "==",
            1,
            f"JUMP_END_PI_{label_id}",
        )

        self.mathi(
            self.n_loop_register.page,
            self.n_loop_register.addr,
            self.n_total_register.addr,
            "-",
            2,
        )

        # Remaining pi pulses: tau spacing from the previous pi pulse.
        self.label(f"LOOP_pi_{label_id}")
        self.set_waveform()
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
        self.mathi(
            self.pulse_index_register.page,
            self.pulse_index_register.addr,
            self.pulse_index_register.addr,
            "+",
            1,
        )
        self.loopnz(
            self.n_loop_register.page,
            self.n_loop_register.addr,
            f"LOOP_pi_{label_id}",
        )

        self.label(f"JUMP_END_PI_{label_id}")
        self.delay_fine_samples.reset()
        self.pulse_index_register.reset()

    def set_waveform(self):
        """
        Configure the delay, waveform address, and phase for the next pi pulse.
        """
        self.offset_computations()
        self.sync(self.delay_coarse_cycles.page, self.delay_coarse_cycles.addr)

        self.address_reg.set_to(
            self.delay_fine_samples,
            '*',
            self.pi_waveform_len_treg,
            physical_unit=False,
        )
        self.select_phase()

    def select_phase(self):
        """
        Set the next pi-pulse phase from the alternating X/Y pattern.
        """
        self.bitwi(
            self.phase_reg.page,
            self.phase_reg.addr,
            self.pulse_index_register.addr,
            "&",
            1,
        )
        self.bitwi(
            self.phase_reg.page,
            self.phase_reg.addr,
            self.phase_reg.addr,
            "<<",
            30,
        )

    def offset_computations(self):
        """
        Decompose the next pi-pulse delay into coarse and fine timing.
        """
        self.math(
            self.delay_fine_samples.page,
            self.delay_fine_samples.addr,
            self.delay_fine_samples.addr,
            "+",
            self.tau.addr,
        )
        self.mathi(
            self.delay_fine_samples.page,
            self.delay_fine_samples.addr,
            self.delay_fine_samples.addr,
            "-",
            self.pi_len_unused,
        )

        self.bitwi(
            self.delay_fine_samples.page,
            self.delay_coarse_cycles.addr,
            self.delay_fine_samples.addr,
            ">>",
            int(np.log2(self.samps_per_clk)),
        )
        self.bitwi(
            self.delay_fine_samples.page,
            self.delay_fine_samples.addr,
            self.delay_fine_samples.addr,
            "&",
            self.samps_per_clk - 1,
        )

    def acquire(self, raw_data=False, *arg, **kwarg):
        """
        Run the experiment and tag the sweep axis as n_pulses.
        """
        data = super().acquire(raw_data=raw_data, sweep_param='n_pulses', *arg, **kwarg)
        return data
