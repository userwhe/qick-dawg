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
