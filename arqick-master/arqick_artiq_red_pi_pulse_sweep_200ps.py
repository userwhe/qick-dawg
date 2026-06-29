import numpy as np
from artiq.experiment import *
from qickdawg.arqick.arqick_pi_count_sweep import PiPulseNumberSweep
from repository.std_sequences.photoncounting import *
import qickdawg as qd
from copy import copy
from repository.arqick_sequences.arqick_artiq_red_config import ARQICK_DoPulses_Red


class ARQICK_red_pi_pulse_sweep_200ps(EnvExperiment, ARQICK_DoPulses_Red):
    def build(self):
        self.setattr_argument("pi_duration_tdds", NumberValue(190, precision=0, step=1, min=1))
        self.setattr_argument("pi_to_pi_delay_tdds", NumberValue(500, precision=0, step=1, min=0))
        self.setattr_argument("n_pi_low", NumberValue(1, precision=0, step=1, min=0))
        self.setattr_argument("n_pi_high", NumberValue(16, precision=0, step=1, min=0))
        self.setattr_argument("n_pi_step", NumberValue(1, precision=0, step=1, min=1))
        self.build_config()

    def prepare(self):
        self.prepare_config(Fineres=True)

    def run(self):
        self.run_config_with_calibration()

    @rpc
    def config_qick(self, default_config):
        config = copy(default_config)
        config.mw_gain = self.mw_gain
        config.freq_fMHz = self.freq_resonant
        config.mw_pi_tdds = int(self.pi_duration_tdds)
        config.pi_to_pi_delay_tdds = int(self.pi_to_pi_delay_tdds)

        n_pi_low = int(self.n_pi_low)
        n_pi_high = int(self.n_pi_high)
        n_pi_step = int(self.n_pi_step)

        # The sweep is inclusive of the start and end values. The plotted
        # x-axis is the actual number of pi pulses N.
        config.add_unitless_linear_sweep("n_pi", n_pi_low, n_pi_high, delta=n_pi_step)
        self.tau_list = np.arange(
            config.n_pi_start,
            config.n_pi_end + config.n_pi_delta,
            config.n_pi_delta,
            dtype=int,
        )
        self.data_size = len(self.tau_list)

        n_pi_list = self.tau_list
        pulse_train_duration = (
            n_pi_list * config.mw_pi_tdds * self.qick_tdds_ns
            + np.maximum(n_pi_list - 1, 0) * config.pi_to_pi_delay_tdds * self.qick_tdds_ns
        )
        self.tau_list2 = (
            pulse_train_duration
            + self.after_red_op_to_mw_buffer
            + self.after_mw_to_spin_readout_buffer
        )

        config.pulse_seq_delay_tus = round(
            self.after_mw_to_spin_readout_buffer * 1e6
            + self.read_to_green * 1e6
            + self.ex_spin_readout * 1e6
            + self.wait_time * 1e6
            + self.charge_readout * 1e6
            + self.qick_experiment_padding * 1e6,
            5,
        )
        config.reps = 1
        config.pmod_out_pin = 0
        config.pmod_out_pulse_width_tns = 300
        config.inherent_trigger_to_pulses_delay_tns = self.inherent_qick_delay_ns * 1e9
        config.pmod_out_trig_delay_tus = round(
            (
                self.t_buffer_us * 1e6
                + self.green_to_red1 * 1e6
                + self.green_init_duration * 1e6
                + self.a1_optical_pump * 1e6
                + self.after_red_op_to_mw_buffer * 1e6
                - config.inherent_trigger_to_pulses_delay_tns / 1000
            ),
            5,
        )
        return config

    @rpc
    def pulse_qick(self, config):
        soc = qd.soc
        prog = PiPulseNumberSweep(config)
        prog.run_rounds(soc, rounds=0, start_src="external")
