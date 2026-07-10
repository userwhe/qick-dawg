# 200 ps XY Pi-Count Sweep Design

## Purpose

Build a bare repeated-pulse error experiment for an NV center. The experiment sweeps the number of individual microwave pi pulses, alternates their phases as `X, Y, X, Y, ...`, and leaves a configurable idle gap between the active end of one pulse and the active start of the next. QICK-DAWG controls the microwave train. ARTIQ controls optical initialization, optical readout, photon counting, and synchronization.

This design covers the coordinated changes required in QICK-DAWG and `dqpcontrol`. It deliberately preserves the laboratory's existing ARTIQ/QICK trigger protocol and 200 ps fine-timing framework.

## Repository and Branch Strategy

| Repository | Base | Feature branch | Purpose |
| --- | --- | --- | --- |
| `userwhe/qick-dawg` | `eab636e4d5a54155e03080e79e06dbec7bb22830` | `weitao/pi-count-sweep` | Microwave program and QICK-side tests |
| `qt3uw/dqpcontrol` | `dqpcomputer-pr` at `7a8e7dad7c3727e728f3f2fe915b9ccfba859ca2` | `weitao/pi-count-artiq-wrapper` | ARTIQ experiment wrapper and lab integration |

The QICK branch starts from the exact commit currently pinned by the `dqpcontrol` submodule. The local QICK checkout uses `userwhe/qick-dawg` as `origin` and `NnguyenHTommy/qick-dawg` as `upstream`.

During feature validation, the dqpcontrol feature branch will point its `qick-dawg` submodule URL and gitlink at the matching commit in `userwhe/qick-dawg`. After the QICK change is accepted or cherry-picked into `NnguyenHTommy/qick-dawg`, the dqpcontrol branch will restore the upstream submodule URL and point at the accepted upstream commit.

## Scope

### Included

- A hardware integer sweep over the number of individual pi pulses.
- A zero-pulse baseline (`N = 0`).
- An `X, Y, X, Y, ...` phase pattern that restarts at X for every shot.
- A configurable end-to-start idle gap between adjacent pi pulses.
- Approximately 200 ps timing resolution using the existing `tdds` shifted-waveform framework.
- The existing external-start and PMOD-marker handshake with ARTIQ.
- A red optical initialization/readout wrapper using `ARQICK_DoPulses_Red`.
- Host-side, compile-time, and hardware validation.

### Excluded

- Initial or final pi/2 pulses.
- CPMG, XY4, or XY8 dynamical-decoupling semantics.
- QICK-controlled laser or ADC readout.
- A general-purpose configurable phase-pattern engine.
- Changes to shared laboratory timing calibrations or unrelated experiments.
- Immediate publication to either remote while GitHub CLI authentication is invalid.

## QICK Program

### Location and public interface

Create:

`src/qickdawg/arqick/arqick_pi_count_sweep.py`

The file will expose:

- `IntegerRegisterSweep`: an integer-only `AbsQickSweep` implementation.
- `PiPulseNumberSweep`: an `NVAveragerProgram` subclass that emits the microwave train and PMOD marker.

Required configuration fields:

- `mw_pi_tdds`: active pi-pulse duration in DAC-sample (`tdds`) units.
- `pi_to_pi_delay_tdds`: idle time from the active end of one pi pulse to the active start of the next, in `tdds` units.
- `n_pi_start`, `n_pi_end`, `nsweep_points`: the exact integer sweep definition.
- `freq_freg`, `mw_channel`, `mw_nqz`, `mw_gain`: microwave generator configuration.
- `reps`: QICK repetition count.
- `pmod_out_pin`, `pmod_out_pulse_width_treg`, `pmod_out_trig_delay_treg`, `inherent_trigger_to_pulses_delay_treg`: ARTIQ handshake timing.
- `pulse_seq_delay_treg`: delay after each microwave train.

### Sweep semantics

The wrapper defines the count sweep by inclusive low and high bounds and a positive step. It computes:

`nsweep_points = (n_pi_end - n_pi_start) // n_pi_step + 1`

The range is accepted only when:

- `n_pi_start >= 0`;
- `n_pi_end >= n_pi_start`;
- `n_pi_step > 0`;
- `(n_pi_end - n_pi_start) % n_pi_step == 0`;
- at least two sweep points are requested for this sweep experiment.

`IntegerRegisterSweep` updates the QICK register with integer arithmetic and returns an integer-valued sweep axis. It will not use `NVQickSweep`'s floating-point step calculation.

### Shot sequence

For each hardware sweep point, `body()` will:

1. Apply the calibrated inherent external-trigger delay.
2. emit one PMOD marker for ARTIQ;
3. apply the configured PMOD-to-microwave delay;
4. copy the swept pulse count into a runtime loop counter;
5. reset phase and fine-timing state so every shot starts at X and at offset zero;
6. skip the microwave loop when `N = 0`;
7. otherwise play exactly `N` pi pulses, using X for even pulse indices and Y for odd pulse indices;
8. insert the configured idle gap only when another pulse remains; and
9. apply `pulse_seq_delay_treg` after the train.

The phase register is set dynamically by the loop. `default_pulse_registers()` will therefore contain only parameters that remain constant and will not also define `phase`. This avoids the QICK 0.2.302 runtime error caused by defining the same parameter in both `default_pulse_registers()` and `set_pulse_registers()`.

### 200 ps timing framework

The program will follow the existing `arqick` fine-timing pattern rather than rounding pulse placement to ordinary tProcessor cycles:

1. Read `samps_per_clk` from `soccfg["gens"][mw_channel]`.
2. Require a positive power-of-two `samps_per_clk` so coarse/fine decomposition can use shifts and masks.
3. Build one padded arbitrary envelope for every fine start offset from zero through `samps_per_clk - 1`.
4. Use a running fine-offset accumulator to select the next shifted envelope.
5. Use the coarse portion of the accumulator for the tProcessor sync.
6. Account for the inactive tail of the padded envelope so the requested interval is the active-end-to-active-start gap.

The direct coarse-cycle calculation is valid for the laboratory common-clock firmware, where generator fabric cycles and tProcessor timing cycles are compatible. Program construction will compare the relevant `soccfg` clock rates and raise a descriptive error for incompatible firmware instead of silently producing an incorrect physical delay.

Because a generator cannot begin a new envelope while the previous padded envelope is still active, `pi_to_pi_delay_tdds` must be at least the padded inactive tail required by the fine-timing implementation. Invalid gaps will fail during program construction.

## ARTIQ Experiment

### Location and inheritance

Create:

`artiq-master/repository/arqick_sequences/arqick_artiq_red_pi_pulse_sweep_200ps.py`

The experiment class will inherit from `EnvExperiment` and `ARQICK_DoPulses_Red`, matching the modern red ARQICK experiments in `dqpcomputer-pr`.

### Dashboard arguments

The wrapper will expose integer-valued controls for:

- pi-pulse duration in `tdds` units;
- inter-pulse end-to-start gap in `tdds` units;
- pulse-count low bound;
- pulse-count high bound; and
- pulse-count step.

It will also inherit the existing red optical preparation, readout, frequency, gain, cycle-count, and calibration arguments.

### Configuration and data flow

`prepare()` calls `prepare_config(Fineres=True)`.

`config_qick()` will:

1. copy the shared `NVConfiguration`;
2. set microwave gain, frequency, pi duration, and inter-pulse gap;
3. validate and add the exact integer count sweep using the delta-based unitless-sweep path;
4. construct an integer count vector and set `data_size` from its length;
5. compute the corresponding microwave duration vector;
6. add the existing optical pre- and post-microwave buffers; and
7. configure the PMOD marker, external-start latency, and post-sequence delay using the same values as neighboring 200 ps red experiments.

For pulse count `N`, the active microwave-train duration is:

`N * mw_pi_tdds + max(N - 1, 0) * pi_to_pi_delay_tdds`

The result is converted with the existing fine-time period and combined with optical buffers to produce `tau_list2`, the ARTIQ scheduling-duration vector. The integer pulse counts remain in `tau_list` for compatibility with `ARQICK_DoPulses_Red`; the wrapper will also publish an `n_pi` dataset so the physical sweep axis is explicit.

`pulse_qick()` constructs `PiPulseNumberSweep` and uses the laboratory's established nonblocking external-start convention:

`run_rounds(soc, rounds=0, start_src="external")`

This convention is intentionally retained because it is used throughout the active dqpcomputer branch to configure the tProcessor, select the external start source, return control to ARTIQ, and allow the TTL input to start execution.

### Synchronization contract

ARTIQ and QICK must agree on all of the following:

- one PMOD marker per sweep point;
- identical sweep length and point order;
- identical meaning of `N` as an individual pi-pulse count;
- identical active microwave duration for every `N`; and
- a zero-pulse point that still emits its marker and reserves only optical buffers.

Only the first point of a loaded QICK sweep receives the ARTIQ start pulse; QICK advances the remaining points internally. A point-count mismatch would therefore deadlock or misalign the optical readout and is treated as a correctness failure.

## Validation and Error Handling

### Host and compile-time tests

QICK-side tests will cover:

- integer sweep points for representative ranges;
- zero, negative, reversed, non-divisible, and single-point range handling;
- constructor compilation against QICK 0.2.302 without default/per-pulse register collisions;
- `N = 0, 1, 2, 3` loop setup;
- phase selection corresponding to `X`, `XY`, and `XYX` prefixes;
- minimum valid and invalid inter-pulse gaps;
- power-of-two samples-per-clock validation; and
- compatible and incompatible clock configurations.

The ARTIQ-side validation will cover:

- Python syntax/import structure using controlled stubs where laboratory-only modules are unavailable;
- exact count-vector construction;
- exact equality between QICK sweep length and ARTIQ `data_size`;
- the duration formula for `N = 0, 1, 2, 3`;
- endpoint divisibility and range errors; and
- the final path and import names expected by ARTIQ repository discovery.

Input errors will use descriptive exceptions for user-facing configuration problems rather than removable `assert` statements. Internal invariants may use assertions only when they cannot be reached from dashboard input.

### Hardware validation checklist

Before treating the experiment as laboratory-ready:

1. Load a short sweep over `N = 0, 1, 2, 3, 4`.
2. Capture ARTIQ start TTL, QICK PMOD marker, and microwave output on an oscilloscope or logic analyzer.
3. Verify one marker per point and no microwave output for `N = 0`.
4. Verify the exact pulse count for each point.
5. Verify alternating orthogonal X/Y phases with IQ-capable measurement or an established phase-sensitive loopback.
6. Measure active-end-to-active-start gaps for values that exercise fine offsets and coarse carries.
7. Check the first and final sweep points for microwave-to-optical-readout alignment.
8. Run the optical experiment and confirm the expected even/odd population parity before interpreting accumulated pulse error.

## Deployment

Feature validation uses the following sequence:

1. Commit the QICK implementation and tests on `weitao/pi-count-sweep`.
2. Publish that branch to `userwhe/qick-dawg` after GitHub authentication is repaired.
3. On `weitao/pi-count-artiq-wrapper`, change the submodule URL to `userwhe/qick-dawg` and the gitlink to the tested QICK commit.
4. Add the ARTIQ wrapper at the final dqpcomputer repository path.
5. Validate a fresh recursive checkout of the dqpcontrol feature branch.
6. Deploy the feature branch to dqpcomputer with a branch checkout/pull and recursive submodule update.

Long-term integration uses:

1. a pull request or cherry-pick from `weitao/pi-count-sweep` into `NnguyenHTommy/qick-dawg:artiq_rfsoc`;
2. restoration of the dqpcontrol submodule URL to `NnguyenHTommy/qick-dawg`; and
3. a gitlink update to the accepted upstream commit.

This leaves the laboratory on its canonical QICK-DAWG repository while keeping feature testing reproducible and avoiding manual script copying.

## Acceptance Criteria

The work is complete when:

- both feature branches descend from the agreed exact bases;
- QICK 0.2.302 constructs the new program without a register-parameter conflict;
- the hardware sweep axis is the exact requested integer sequence;
- each point emits one PMOD marker and exactly `N` microwave pi pulses;
- phases follow `X, Y, X, Y, ...` and restart at X for each point;
- the measured end-to-start gap follows the requested 200 ps-grid value;
- `N = 0` emits no microwave pulse but remains synchronized;
- ARTIQ and QICK agree on point count, order, and duration;
- the ARTIQ file is discoverable at the final dqpcomputer path;
- a recursive checkout of the dqpcontrol feature branch resolves the matching QICK commit; and
- the hardware validation checklist passes or any hardware-only exceptions are recorded explicitly before deployment.
