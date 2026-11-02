"""Experimental RLBot v4 wrapper for a small RLGym-PPO policy.

This bot is an experiment, not a promise of competitive play. It recreates the
training observation from RLBot v4 packets, using approximations for internal
RocketSim-only jump/flip state fields that RLBot does not expose directly.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


# The paths match the folders set up in this project. Keep the training folder
# and the selected checkpoint in place while using this bot.
TRAINING_ROOT = Path.home() / "OneDrive" / "Desktop" / "RocketLeagueAITraining"
CHECKPOINT_DIR = (
    TRAINING_ROOT
    / "data"
    / "checkpoints"
    / "rlgym-ppo-shaped-run-1792300946107747500"
    / "4921344"
)
COMMON_VALUES_FILE = (
    TRAINING_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "rlgym"
    / "rocket_league"
    / "common_values.py"
)

OBS_SIZE = 92
N_ACTIONS = 90
ACTION_REPEAT = 8  # The training environment used RepeatAction(..., repeats=8).

POS_COEF = np.asarray(
    [1.0 / 4096.0, 1.0 / 6000.0, 1.0 / 2044.0], dtype=np.float32
)
LIN_VEL_COEF = np.float32(1.0 / 2300.0)
ANG_VEL_COEF = np.float32(1.0 / 5.5)
PAD_TIMER_COEF = np.float32(1.0 / 10.0)
BOOST_COEF = np.float32(1.0 / 100.0)
INVERT_VEC = np.asarray([-1.0, -1.0, 1.0], dtype=np.float32)
INVERT_MTX = np.asarray(
    [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)


class DiscretePolicy(nn.Module):
    """Same parameter names and layer layout as rlgym_ppo's DiscreteFF."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(OBS_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
            nn.Softmax(dim=-1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


def make_lookup_table() -> np.ndarray:
    """Replicate RLGym LookupTableAction.make_lookup_table() exactly."""
    actions: List[List[float]] = []

    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )

    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

    table = np.asarray(actions, dtype=np.float32)
    if table.shape != (N_ACTIONS, 8):
        raise RuntimeError(f"Expected 90 lookup-table actions, got {table.shape}")
    return table


def vec3(value) -> np.ndarray:
    return np.asarray([value.x, value.y, value.z], dtype=np.float32)


def rotation_matrix(rotator) -> np.ndarray:
    """RLGym's euler_to_rotation convention for pitch/yaw/roll in radians."""
    pitch = float(rotator.pitch)
    yaw = float(rotator.yaw)
    roll = float(rotator.roll)

    cp, cy, cr = math.cos(pitch), math.cos(yaw), math.cos(roll)
    sp, sy, sr = math.sin(pitch), math.sin(yaw), math.sin(roll)

    matrix = np.zeros((3, 3), dtype=np.float32)
    # Forward vector (column 0)
    matrix[0, 0] = cp * cy
    matrix[1, 0] = cp * sy
    matrix[2, 0] = sp
    # Left vector (column 1)
    matrix[0, 1] = cy * sp * sr - cr * sy
    matrix[1, 1] = sy * sp * sr + cr * cy
    matrix[2, 1] = -cp * sr
    # Up vector (column 2)
    matrix[0, 2] = -cr * cy * sp - sr * sy
    matrix[1, 2] = -cr * sy * sp + sr * cy
    matrix[2, 2] = cp * cr
    return matrix


def load_common_values(path: Path):
    """Load only RLGym's constants file without importing its other packages."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find RLGym common_values.py at {path}. "
            "Please make sure the RocketLeagueAITraining .venv is still present."
        )
    spec = importlib.util.spec_from_file_location("rlgym_common_values_for_ai_bot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load RLGym constants from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainedPolicyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.policy = None
        self.obs_mean = None
        self.obs_std = None
        self.actions = make_lookup_table()
        self.pad_map: List[int] = []

        self.last_action = np.zeros(8, dtype=np.float32)
        self.last_controls = SimpleControllerState()
        self.last_decision_frame = None
        self.last_frame = None
        self.was_on_ground = None
        self.last_native_double_jumped = False

        self.is_jumping = False
        self.jump_time = 0.0
        self.air_time_since_jump = 0.0
        self.has_flipped = False
        self.has_double_jumped = False
        self.flip_time = 0.65

        self.previous_boost: Dict[int, float] = {}
        self.boost_drop_seen: Dict[int, bool] = {}
        self.demo_started: Dict[int, float] = {}
        self.printed_error = False
        self.last_debug_frame = None

    def initialize_agent(self):
        self._load_policy_and_normalizer()
        self._map_boost_pads()
        print(
            f"TrainedPolicyBot ready: checkpoint {self.checkpoint_steps:,}, "
            f"observation size {OBS_SIZE}, actions {N_ACTIONS}."
        )

    def _load_policy_and_normalizer(self):
        policy_path = CHECKPOINT_DIR / "PPO_POLICY.pt"
        book_path = CHECKPOINT_DIR / "BOOK_KEEPING_VARS.json"
        if not policy_path.is_file() or not book_path.is_file():
            raise FileNotFoundError(
                "The AI checkpoint is missing files. Expected PPO_POLICY.pt and "
                f"BOOK_KEEPING_VARS.json in: {CHECKPOINT_DIR}"
            )

        with book_path.open("r", encoding="utf-8") as file:
            book = json.load(file)
        self.checkpoint_steps = int(book.get("cumulative_timesteps", 0))

        stats = book.get("obs_running_stats")
        if not isinstance(stats, dict):
            raise RuntimeError("Checkpoint is missing obs_running_stats.")
        self.obs_mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
        accumulated_variance = np.asarray(stats["var"], dtype=np.float32).reshape(-1)
        count = int(stats.get("count", 0))
        if len(self.obs_mean) != OBS_SIZE or len(accumulated_variance) != OBS_SIZE:
            raise RuntimeError(
                "Unexpected observation-normalizer dimensions: "
                f"mean={len(self.obs_mean)}, variance={len(accumulated_variance)}"
            )

        # WelfordRunningStat stores the accumulated squared deviation in 'var';
        # rlgym-ppo divides by count-1, sets zero variances to 1, then takes sqrt.
        if count < 2:
            self.obs_std = np.ones(OBS_SIZE, dtype=np.float32)
        else:
            variance = accumulated_variance / float(count - 1)
            variance = np.where(variance == 0.0, 1.0, variance)
            self.obs_std = np.sqrt(variance).astype(np.float32)

        self.policy = DiscretePolicy().cpu()
        state_dict = torch.load(str(policy_path), map_location="cpu")
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def _map_boost_pads(self):
        """Map RLGym's canonical 34 pad locations to RLBot FieldInfo indexes."""
        common_values = load_common_values(COMMON_VALUES_FILE)
        training_locations = np.asarray(common_values.BOOST_LOCATIONS, dtype=np.float32)
        field = self.get_field_info()
        live_locations = [vec3(field.boost_pads[i].location) for i in range(field.num_boosts)]

        if len(training_locations) != 34 or len(live_locations) < 34:
            raise RuntimeError(
                "Expected Soccar boost pads (34). "
                f"RLGym={len(training_locations)}, RLBot={len(live_locations)}."
            )

        unused = set(range(len(live_locations)))
        mapping: List[int] = []
        for location in training_locations:
            if not unused:
                raise RuntimeError("Could not map all boost pad locations.")
            closest = min(unused, key=lambda i: float(np.linalg.norm(live_locations[i] - location)))
            distance = float(np.linalg.norm(live_locations[closest] - location))
            # RLBot v4 field info can differ slightly from RocketSim's canonical
            # coordinates for some stadiums. The closest mismatch observed in the
            # user's Mannfield test was 64 units, so allow up to 100 units while
            # still matching pads one-to-one by nearest location.
            if distance > 100.0:
                raise RuntimeError(
                    "Could not match RLGym boost pad to RLBot field info; "
                    f"closest distance was {distance:.1f} units (limit 100)."
                )
            if distance > 25.0:
                print(
                    "Warning: boost-pad coordinate mismatch accepted: "
                    f"{distance:.1f} units; mapping nearest pad."
                )
            mapping.append(closest)
            unused.remove(closest)
        self.pad_map = mapping

    def _track_packet(self, packet: GameTickPacket, dt: float):
        now = float(packet.game_info.seconds_elapsed)
        for i in range(int(packet.num_cars)):
            car = packet.game_cars[i]
            boost = float(car.boost)
            previous = self.previous_boost.get(i)
            if previous is not None and boost < previous - 0.01:
                self.boost_drop_seen[i] = True
            self.previous_boost[i] = boost
            self.boost_drop_seen.setdefault(i, False)

            if bool(car.is_demolished):
                self.demo_started.setdefault(i, now)
            else:
                self.demo_started.pop(i, None)

    def _demo_timer(self, car_index: int, now: float) -> float:
        started = self.demo_started.get(car_index)
        if started is None:
            return 0.0
        return float(np.clip(3.0 - (now - started), 0.0, 3.0))

    def _update_jump_features(self, packet: GameTickPacket, dt: float):
        me = packet.game_cars[self.index]
        on_ground = bool(me.has_wheel_contact)
        native_double_jumped = bool(me.double_jumped)
        native_jumped = bool(me.jumped)
        holding_jump = bool(self.last_action[5] > 0.5)

        if on_ground:
            self.is_jumping = False
            self.jump_time = 0.0
            self.air_time_since_jump = 0.0
            self.has_flipped = False
            self.has_double_jumped = False
            self.flip_time = 0.65
            self.last_native_double_jumped = False
        else:
            # Approximate RocketSim's transient is_jumping flag: it is true
            # during the initial jump while jump is held, for at most ~0.2 s.
            if self.was_on_ground and holding_jump and native_jumped:
                self.is_jumping = True
                self.jump_time = 0.0
                self.air_time_since_jump = 0.0
            elif self.is_jumping:
                if holding_jump:
                    self.jump_time += dt
                    if self.jump_time >= 0.2:
                        self.is_jumping = False
                else:
                    self.is_jumping = False

            if native_double_jumped and not self.last_native_double_jumped:
                directional_jump = (
                    abs(float(self.last_action[2]))
                    + abs(float(self.last_action[3]))
                    + abs(float(self.last_action[4]))
                ) >= 0.5
                if directional_jump:
                    self.has_flipped = True
                    self.flip_time = 0.0
                else:
                    self.has_double_jumped = True
            elif self.has_flipped:
                self.flip_time += dt

            if not native_jumped or self.is_jumping:
                self.air_time_since_jump = 0.0
            else:
                self.air_time_since_jump = min(1.25, self.air_time_since_jump + dt)

            self.last_native_double_jumped = native_double_jumped

        self.was_on_ground = on_ground

    def _car_obs(self, car, car_index: int, inverted: bool, now: float, own_car: bool) -> np.ndarray:
        physics = car.physics
        position = vec3(physics.location)
        linear_velocity = vec3(physics.velocity)
        angular_velocity = vec3(physics.angular_velocity)
        matrix = rotation_matrix(physics.rotation)

        if inverted:
            position *= INVERT_VEC
            linear_velocity *= INVERT_VEC
            angular_velocity *= INVERT_VEC
            matrix *= INVERT_MTX

        forward = matrix[:, 0]
        up = matrix[:, 2]
        on_ground = float(bool(car.has_wheel_contact))
        if own_car:
            is_boosting = float(bool(self.last_action[6] > 0.5) and float(car.boost) > 0.0)
        else:
            is_boosting = float(self.boost_drop_seen.get(car_index, False))

        return np.concatenate(
            [
                position * POS_COEF,
                forward,
                up,
                linear_velocity * LIN_VEL_COEF,
                angular_velocity * ANG_VEL_COEF,
                np.asarray(
                    [
                        float(car.boost) * BOOST_COEF,
                        self._demo_timer(car_index, now),
                        on_ground,
                        is_boosting,
                        float(bool(car.is_super_sonic)),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32, copy=False)

    def _build_observation(self, packet: GameTickPacket) -> np.ndarray:
        num_cars = int(packet.num_cars)
        if num_cars < 2:
            raise RuntimeError(
                f"Need at least 2 cars for this 1v1 policy; packet has {num_cars}."
            )
        if len(self.pad_map) != 34:
            raise RuntimeError("Boost-pad map is not ready.")

        # The policy was trained in 1v1 and expects exactly one opponent in its
        # observation. In this RLBot setup, the configured participants are slots
        # 0 (this AI) and 1 (the example bot). Rocket League may also report the
        # local player as an extra car at index 2. Exclude any extra cars from the
        # neural-network observation so its input remains the trained 92 values.
        training_indices = list(range(min(2, num_cars)))
        if self.index not in training_indices:
            raise RuntimeError(
                f"Expected the trained bot in slot 0 or 1; got slot {self.index}."
            )
        me = packet.game_cars[self.index]
        opponents = [i for i in training_indices if i != self.index]
        if len(opponents) != 1:
            raise RuntimeError(
                f"Could not identify the trained 1v1 opponent in slots {training_indices}."
            )
        enemy_index = opponents[0]
        enemy = packet.game_cars[enemy_index]
        inverted = int(self.team) == 1
        now = float(packet.game_info.seconds_elapsed)

        ball = packet.game_ball.physics
        ball_position = vec3(ball.location)
        ball_velocity = vec3(ball.velocity)
        ball_angular_velocity = vec3(ball.angular_velocity)
        if inverted:
            ball_position *= INVERT_VEC
            ball_velocity *= INVERT_VEC
            ball_angular_velocity *= INVERT_VEC

        # Order boost pads by RLGym's BOOST_LOCATIONS, not by RLBot packet order.
        pad_timers = []
        for live_index in self.pad_map:
            pad = packet.game_boosts[live_index]
            timer = 0.0 if bool(pad.is_active) else float(pad.timer)
            pad_timers.append(float(np.clip(timer, 0.0, 10.0)))
        if inverted:
            pad_timers.reverse()  # RLGym GameState.inverted_boost_pad_timers uses [::-1].

        on_ground = bool(me.has_wheel_contact)
        has_jumped = bool(me.jumped)
        has_double_jumped = self.has_double_jumped
        can_flip = (
            not on_ground
            and not bool(self.last_action[5] > 0.5)
            and not self.has_flipped
            and not has_double_jumped
            and self.air_time_since_jump < 1.25
        )
        partial = np.asarray(
            [
                float(bool(self.last_action[5] > 0.5)),
                float(self.last_action[7]),
                float(has_jumped),
                float(self.is_jumping),
                float(self.has_flipped),
                float(self.has_flipped and self.flip_time < 0.65),
                float(has_double_jumped),
                float(can_flip),
                float(self.air_time_since_jump),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                ball_position * POS_COEF,
                ball_velocity * LIN_VEL_COEF,
                ball_angular_velocity * ANG_VEL_COEF,
                np.asarray(pad_timers, dtype=np.float32) * PAD_TIMER_COEF,
                partial,
                self._car_obs(me, self.index, inverted, now, own_car=True),
                self._car_obs(enemy, enemy_index, inverted, now, own_car=False),
            ]
        ).astype(np.float32, copy=False)

        if obs.shape != (OBS_SIZE,):
            raise RuntimeError(f"Built observation has shape {obs.shape}; expected ({OBS_SIZE},).")
        return obs

    @staticmethod
    def _as_controller(action: np.ndarray) -> SimpleControllerState:
        controls = SimpleControllerState()
        controls.throttle = float(action[0])
        controls.steer = float(action[1])
        controls.pitch = float(action[2])
        controls.yaw = float(action[3])
        controls.roll = float(action[4])
        controls.jump = bool(action[5] > 0.5)
        controls.boost = bool(action[6] > 0.5)
        controls.handbrake = bool(action[7] > 0.5)
        return controls

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        frame = int(packet.game_info.frame_num)
        if self.last_frame is None or frame < self.last_frame:
            dt = 1.0 / 120.0
        else:
            dt = float(np.clip((frame - self.last_frame) / 120.0, 0.0, 0.1))
        self.last_frame = frame

        self._track_packet(packet, dt)
        self._update_jump_features(packet, dt)

        should_infer = (
            self.last_decision_frame is None
            or frame - self.last_decision_frame >= ACTION_REPEAT
        )
        if not should_infer:
            return self.last_controls

        try:
            observation = self._build_observation(packet)
            standardized = np.clip(
                (observation - self.obs_mean) / self.obs_std, -5.0, 5.0
            ).astype(np.float32, copy=False)
            with torch.no_grad():
                tensor = torch.as_tensor(standardized, dtype=torch.float32, device="cpu")
                probabilities = self.policy(tensor)

                # The 90-action lookup table mixes 24 ground actions with aerial
                # actions. Some aerial-table entries have jump=0; when selected on
                # the ground, they can send pitch/roll inputs that have no useful
                # effect and sometimes map to no throttle. Keep normal ground
                # actions plus genuine jump actions available while wheel contact
                # is true. This is an inference-time safety heuristic; it does not
                # alter or retrain the checkpoint.
                on_ground = bool(packet.game_cars[self.index].has_wheel_contact)
                if on_ground:
                    action_ids = np.arange(N_ACTIONS)
                    valid_grounded_actions = (action_ids < 24) | (self.actions[:, 5] > 0.5)
                    valid_mask = torch.as_tensor(valid_grounded_actions, dtype=torch.bool)
                    probabilities = probabilities.clone()
                    probabilities[~valid_mask] = 0.0
                    probability_sum = probabilities.sum().clamp_min(1e-8)
                    probabilities = probabilities / probability_sum

                action_index = int(torch.argmax(probabilities).item())
                top_probs, top_indices = torch.topk(probabilities, k=3)
            self.last_action = self.actions[action_index].copy()
            self.last_controls = self._as_controller(self.last_action)
            self.last_decision_frame = frame

            # Low-rate diagnostics: show the controls being chosen and the policy's
            # top candidates, without flooding the console at the full game tick rate.
            if self.last_debug_frame is None or frame - self.last_debug_frame >= 240:
                top_summary = ", ".join(
                    f"{int(idx)}:{float(prob):.2f}"
                    for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
                )
                print(
                    "AI action debug | "
                    f"frame={frame} action={action_index} "
                    f"ground={int(bool(packet.game_cars[self.index].has_wheel_contact))} "
                    f"throttle={self.last_controls.throttle:.0f} "
                    f"steer={self.last_controls.steer:.0f} "
                    f"pitch={self.last_controls.pitch:.0f} "
                    f"yaw={self.last_controls.yaw:.0f} "
                    f"roll={self.last_controls.roll:.0f} "
                    f"jump={int(self.last_controls.jump)} "
                    f"boost={int(self.last_controls.boost)} "
                    f"handbrake={int(self.last_controls.handbrake)} "
                    f"top3=[{top_summary}]"
                )
                self.last_debug_frame = frame

            # Reset opponent boost-change markers for the next 8-frame decision window.
            for car_index in range(int(packet.num_cars)):
                self.boost_drop_seen[car_index] = False
            return self.last_controls
        except Exception as exc:
            # Fail closed instead of sending accidental/unbounded controls if any
            # packet does not match the trained 1v1 setup.
            if not self.printed_error:
                print(f"TrainedPolicyBot observation/inference error: {exc}")
                self.printed_error = True
            self.last_action = np.zeros(8, dtype=np.float32)
            self.last_controls = SimpleControllerState()
            self.last_decision_frame = frame
            return self.last_controls
