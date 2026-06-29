"""
Sweep number of pi pulses
=========================

For each sweep point, this program sends a PMOD trigger and then plays
N microwave pi pulses with alternating XY phases, where N is swept from
n_pi_start to n_pi_end.

The delay between pi pulses is specified in tdds units, giving the same
approximately 200 ps timing granularity used by the other arqick 200 ps
programs.

This version does not include laser/ADC readout. It is meant to be used with
external readout/triggering, similar to the arqick_*_200ps pulse-only programs.
"""

from qick.averager_program import AbsQickSweep
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram

import numpy as np


class IntRegisterSweep(AbsQickSweep):
    """
    Integer-valued register sweep.

    This is safer than using NVQickSweep directly for pulse count, because
    the number of pi pulses must stay an integer.
    """

    def __init__(self, prog, reg, start, stop, expts, label=None):
        super().__init__(prog)

        assert isinstance(start, int), "n_pi_start must be an integer"
        assert isinstance(stop, int), "n_pi_end must be an integer"
        assert isinstance(expts, int), "nsweep_points must be an integer"
        assert expts >= 2, "Need at least 2 sweep points"
        assert stop >= start, "n_pi_end must be >= n_pi_start"

        numerator = stop - start
        denominator = expts - 1
        assert numerator % denominator == 0, (
            "The pi-pulse-count sweep must have integer spacing. "
            "For example, use n_pi_start=0, n_pi_end=16, nsweep_points=17."
        )

        self.prog = prog
        self.reg = reg
        self.start = start
        self.stop = stop
        self.expts = expts
        self.step = numerator // denominator
        self.label = label if label is not None else reg.name

        self.reg.init_val = self.start

    def reset(self):
        self.reg.reset()

    def update(self):
        self.reg.set_to(
            self.reg,
            "+",
            self.step,
            physical_unit=False,
        )

    def get_sweep_pts(self):
        return self.start + self.step * np.arange(self.expts, dtype=int)


class PiPulseNumberSweep(NVAveragerProgram):
    """
    Sweep the number of microwave pi pulses.

    One shot is:

        PMOD trigger
        wait trigger-to-pulse delay
        play pi_X, delay, pi_Y, delay, ... for N total pi pulses
        wait pulse-sequence delay

    where N is swept by a hardware register. The sweep axis returned by
    get_expt_pts() is the actual number of pi pulses N.

    The inter-pi delay is cfg.pi_to_pi_delay_tdds in DAC-sample units. Each
    successive pi pulse uses a shifted waveform selected by a fine-delay
    register, while the coarse part of the delay is handled by a tProc sync.
    """

    required_cfg = [
        # Pi pulse waveform
        "mw_pi_tdds",          # pi-pulse duration in fine DDS/sample units

        # Number-of-pulses sweep
        "n_pi_start",          # integer, e.g. 0
        "n_pi_end",            # integer, e.g. 16
        "nsweep_points",       # e.g. 17 for 0,1,2,...,16

        # Microwave generator
        "freq_freg",
        "mw_channel",
        "mw_nqz",
        "mw_gain",

        # Repetition
        "reps",

        # External trigger
        "pmod_out_pin",
        "pmod_out_pulse_width_treg",
        "pmod_out_trig_delay_treg",
        "inherent_trigger_to_pulses_delay_treg",

        # Pulse train timing
        "pi_to_pi_delay_tdds",     # active pi end to next active pi start
        "pulse_seq_delay_treg",    # delay after the pi-pulse train
    ]

    def initialize(self):
        self.check_cfg()

        assert self.cfg.n_pi_start >= 0, "n_pi_start must be nonnegative"
        assert self.cfg.n_pi_end >= self.cfg.n_pi_start, (
            "n_pi_end must be >= n_pi_start"
        )

        # Declare microwave generator.
        self.declare_gen(
            ch=self.cfg.mw_channel,
            nqz=self.cfg.mw_nqz,
        )

        # Fine waveform timing.
        # In the arqick 200 ps files, this is expected to be 16 for the MW channel.
        self.samps_per_clk = self.soccfg["gens"][self.cfg.mw_channel]["samps_per_clk"]
        assert self.samps_per_clk > 0
        assert self.samps_per_clk & (self.samps_per_clk - 1) == 0, (
            "samps_per_clk must be a power of two for bitwise fine timing."
        )
        self.log2_samps_per_clk = int(np.log2(self.samps_per_clk))

        # Arbitrary waveforms need a minimum length of a few treg units.
        # The extra samps_per_clk - 1 samples allow all fine start offsets.
        self.pi_waveform_len_treg = max(
            int(
                np.ceil(
                    (self.cfg.mw_pi_tdds + self.samps_per_clk - 1)
                    / self.samps_per_clk
                )
            ),
            3,
        )
        self.pi_waveform_len_tdds = self.pi_waveform_len_treg * self.samps_per_clk
        self.pi_len_unused_tdds = self.pi_waveform_len_tdds - self.cfg.mw_pi_tdds

        assert self.cfg.pi_to_pi_delay_tdds >= self.pi_len_unused_tdds, (
            "pi_to_pi_delay_tdds is too short for this pi waveform. "
            f"Use at least {self.pi_len_unused_tdds} tdds."
        )

        # Build rectangular pi pulses shifted by each DAC sample offset.
        for offset in range(self.samps_per_clk):
            i_data = np.zeros(self.pi_waveform_len_tdds)
            q_data = np.zeros(self.pi_waveform_len_tdds)

            i_data[offset: offset + self.cfg.mw_pi_tdds] = 1
            q_data[offset: offset + self.cfg.mw_pi_tdds] = 1

            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)

            self.add_envelope(
                ch=self.cfg.mw_channel,
                name=f"pi_{offset}",
                idata=i_data,
                qdata=q_data,
            )

        # Configure the MW pulse registers.
        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style="arb",
            freq=self.cfg.freq_freg,
            gain=self.cfg.mw_gain,
            phase=0,
        )

        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name="addr")

        # Swept register: current number of pi pulses.
        self.n_pi_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name="n_pi",
            init_val=self.cfg.n_pi_start,
        )

        # Runtime loop counter. We copy n_pi_register into this each shot.
        self.pi_counter_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name="pi_counter",
            init_val=0,
        )

        self.phase_register = self.get_gen_reg(self.cfg.mw_channel, name="phase")

        self.tdds_offset_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name="tdds_offset",
            init_val=0,
        )

        self.treg_offset_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name="treg_offset",
            init_val=0,
        )

        # 0 -> X phase, 1 -> Y phase. This toggles once per pi pulse.
        self.phase_step_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name="phase_step",
            init_val=0,
        )

        self.add_sweep(
            IntRegisterSweep(
                self,
                reg=self.n_pi_register,
                start=self.cfg.n_pi_start,
                stop=self.cfg.n_pi_end,
                expts=self.cfg.nsweep_points,
                label="n_pi",
            )
        )

        self.synci(200)

    def body(self):
        # Align timing with external trigger/pulse sequence convention.
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)

        # PMOD trigger to external equipment.
        self.trigger(
            pins=[self.cfg.pmod_out_pin],
            width=self.cfg.pmod_out_pulse_width_treg,
        )

        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        # Configure the pi pulse once for this shot.
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            waveform="pi_0",
            phase=self.deg2reg(0),
        )

        # Copy swept n_pi value into a loop counter.
        self.pi_counter_register.set_to(
            self.n_pi_register,
            "+",
            0,
            physical_unit=False,
        )

        # Start every train on X.
        self.phase_step_register.reset()
        self.tdds_offset_register.reset()

        # If n_pi == 0, skip the pulse loop.
        self.condj(
            self.pi_counter_register.page,
            self.pi_counter_register.addr,
            "==",
            0,
            "PI_LOOP_END",
        )

        self.label("PI_LOOP")

        # Play one pi pulse. DDS phase register uses 90 deg == 1 << 30,
        # so phase_step 0/1 maps directly to X/Y.
        self.bitwi(
            self.phase_register.page,
            self.phase_register.addr,
            self.phase_step_register.addr,
            "<<",
            30,
        )
        self.pulse(ch=self.cfg.mw_channel)

        # Wait until the pulse is done.
        self.sync_all()

        # Decrement pulse counter.
        self.pi_counter_register.set_to(
            self.pi_counter_register,
            "-",
            1,
            physical_unit=False,
        )

        # If that was the last pi pulse, exit.
        self.condj(
            self.pi_counter_register.page,
            self.pi_counter_register.addr,
            "==",
            0,
            "PI_LOOP_END",
        )

        # Delay before the next pi pulse.
        self.configure_next_pi_pulse_delay()

        # Toggle X <-> Y for the next pi pulse.
        self.phase_step_register.set_to(
            self.phase_step_register,
            "+",
            1,
            physical_unit=False,
        )
        self.bitwi(
            self.phase_step_register.page,
            self.phase_step_register.addr,
            self.phase_step_register.addr,
            "&",
            1,
        )

        # If counter is still positive, loop.
        self.condj(
            self.pi_counter_register.page,
            self.pi_counter_register.addr,
            ">",
            0,
            "PI_LOOP",
        )

        self.label("PI_LOOP_END")

        # Delay before the next shot.
        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def configure_next_pi_pulse_delay(self):
        """
        Select the shifted pi waveform and coarse sync for the next pulse.

        tdds_offset_register holds the fine offset of the current pi pulse.
        After sync_all(), the remaining unused waveform time depends on that
        offset. Adding the current fine offset back into the next delay keeps
        the active pulse-to-pulse gap equal to pi_to_pi_delay_tdds.
        """
        self.tdds_offset_register.set_to(
            self.tdds_offset_register,
            "+",
            self.cfg.pi_to_pi_delay_tdds - self.pi_len_unused_tdds,
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

        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
