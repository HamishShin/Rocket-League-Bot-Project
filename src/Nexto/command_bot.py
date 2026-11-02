"""Command-driven Nexto wrapper.

By default the bot waits with neutral controls. Send commands using command_input.py.
Commands: frontflip, backflip, halfflip, wavedash, jump, doublejump, hit, score, wait/stop, cancel, resume, help.
The original Nexto bot implementation in bot.py is loaded unchanged.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

from rlbot.agents.base_agent import SimpleControllerState

# ======================= MECHANIC TUNING =======================
# Half-flip: these are phase lengths in seconds. Change these first if needed.
HALF_FLIP_REVERSE_TIME = 0.10
HALF_FLIP_FIRST_JUMP_TIME = 0.01
HALF_FLIP_JUMP_RELEASE_TIME = 0.08
HALF_FLIP_BACKFLIP_INPUT_TIME = 0.09
HALF_FLIP_INVERTED_TIMEOUT = 0.10
HALF_FLIP_CANCEL_FORWARD_TIME = 0.50  # keep stick forward longer to cancel the backflip
HALF_FLIP_ROLL_DIRECTION = 1.0   # +1 = roll right; -1 = roll left
HALF_FLIP_TOTAL_TIMEOUT = 3.8  # safety fallback; normally air-roll lasts until wheel contact

# Wave-dash: pitch > 0 raises the nose; pitch < 0 pitches forward.
WAVEDASH_JUMP_TIME = 0.08
WAVEDASH_JUMP_RELEASE_TIME = 0.09
WAVEDASH_NOSE_UP_PITCH = 0.16
WAVEDASH_TRIGGER_HEIGHT_ABOVE_START = 20.0  # trigger earlier so the dodge press registers before landing
WAVEDASH_MIN_DESCENDING_SPEED = 45.0        # require velocity.z < -this value; small jumps descend slowly
WAVEDASH_FORWARD_DODGE_TIME = 0.18          # hold forward + jump long enough to register the dodge
WAVEDASH_LANDING_TIMEOUT = 1.2
WAVEDASH_TOTAL_TIMEOUT = 2.5
# ===============================================================

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Load the original Nexto bot under another module name to avoid importing
# this wrapper recursively (this file is command_bot.py, not bot.py).
_CORE_NAME = "_nexto_core_for_commands"
_core_path = HERE / "bot.py"
_spec = importlib.util.spec_from_file_location(_CORE_NAME, _core_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load original Nexto bot at {_core_path}")
_core_module = importlib.util.module_from_spec(_spec)
sys.modules[_CORE_NAME] = _core_module
_spec.loader.exec_module(_core_module)
NextoBase = _core_module.Nexto


class NextoCommandBot(NextoBase):
    """Nexto that idles until commanded; scripted actions end in idle mode."""

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self._command_file = HERE / "command.txt"
        self._last_command_token = None
        self._last_poll_time = 0.0
        self._active_command = None
        self._command_start_time = 0.0
        self._mode = "idle"  # idle by default; 'autonomous' only after `resume`
        self._idle_brake_until = 0.0
        self._last_idle_brake_note = False
        self._score_baseline = None
        self._score_timeout = 30.0
        self._hit_baseline_touch_time = None
        self._hit_timeout = 20.0
        self._move_phase = None
        self._move_phase_start_time = 0.0
        self._move_base_z = 0.0
        self._move_airborne_seen = False

    @staticmethod
    def _read_command_and_token(contents: str):
        if "|" in contents:
            command, token = contents.split("|", 1)
        else:
            command, token = contents, contents
        return command.strip().lower().lstrip("!"), token.strip()

    def initialize_agent(self):
        super().initialize_agent()
        try:
            self._command_file.touch(exist_ok=True)
            # Ignore a command left in the inbox from a previous run/reload.
            contents = self._command_file.read_text(encoding="utf-8").strip()
            if contents:
                _, self._last_command_token = self._read_command_and_token(contents)
        except (OSError, UnicodeError) as exc:
            print(f"[Nexto commands] Could not initialize command inbox: {exc}")
        print("NextoCommandBot ready. Mode: IDLE (waiting for commands).")
        print("Commands: frontflip | backflip | halfflip | wavedash | jump | doublejump | hit | score | stop/wait | cancel | resume | help")
        print("After a scripted move, Nexto returns to IDLE. `resume` enables normal Nexto play.")

    def _enter_idle(self, packet, message: str):
        self._mode = "idle"
        self._active_command = None
        self._move_phase = None
        self._move_airborne_seen = False
        self._score_baseline = None
        self._hit_baseline_touch_time = None
        try:
            now = float(packet.game_info.seconds_elapsed)
        except (AttributeError, TypeError, ValueError):
            now = 0.0
        # Brief ground-only reverse-throttle braking helps the car settle after
        # a command; controls return to neutral after this short window.
        self._idle_brake_until = now + 2.5
        self._last_idle_brake_note = False
        print(f"[Nexto commands] {message} Mode: IDLE; waiting for the next command.")

    def _poll_command(self, packet):
        now_mono = time.monotonic()
        if now_mono - self._last_poll_time < 0.10:
            return
        self._last_poll_time = now_mono
        try:
            contents = self._command_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return
        if not contents:
            return

        command, token = self._read_command_and_token(contents)
        if not command or token == self._last_command_token:
            return
        self._last_command_token = token

        if command == "help":
            print("[Nexto commands] frontflip, backflip, halfflip (13), wavedash (14), jump, doublejump, hit, score, stop/wait, cancel, resume, help")
            print("[Nexto commands] jump/doublejump/flips run a short sequence then wait; hit lets Nexto play until it touches the ball once; score plays until a goal or timeout.")
            print("[Nexto commands] Default is idle. `resume` gives control back to Nexto's AI.")
            return

        if command in {"stop", "wait"}:
            self._enter_idle(packet, "Stopped autonomous control.")
            return

        if command == "cancel":
            self._enter_idle(packet, "Active command cancelled.")
            return

        if command == "resume":
            self._active_command = None
            self._score_baseline = None
            self._hit_baseline_touch_time = None
            self._mode = "autonomous"
            self._idle_brake_until = 0.0
            print("[Nexto commands] Autonomous Nexto control resumed. Send `stop` or `wait` to pause it.")
            return

        if command == "hit":
            if self._active_command is not None:
                print("[Nexto commands] A command is already active. Use `cancel` first.")
                return
            try:
                self._hit_baseline_touch_time = float(packet.game_ball.latest_touch.time_seconds)
            except (AttributeError, TypeError, ValueError):
                self._hit_baseline_touch_time = None
            self._mode = "autonomous"
            self._active_command = "hit"
            self._command_start_time = float(packet.game_info.seconds_elapsed)
            self._idle_brake_until = 0.0
            print(
                f"[Nexto commands] Hit attempt started. Nexto will play until it touches the ball, "
                f"or for {self._hit_timeout:.0f} seconds, then wait."
            )
            return

        if command == "score":
            if self._active_command is not None:
                print("[Nexto commands] A command is already active. Use `cancel` first.")
                return
            try:
                self._score_baseline = (int(packet.teams[0].score), int(packet.teams[1].score))
            except (AttributeError, IndexError, TypeError, ValueError):
                self._score_baseline = None
            self._mode = "autonomous"
            self._active_command = "score"
            self._command_start_time = float(packet.game_info.seconds_elapsed)
            self._idle_brake_until = 0.0
            print(
                f"[Nexto commands] Score attempt started. Nexto will use its normal policy "
                f"until a goal is scored or {self._score_timeout:.0f} seconds pass, then wait."
            )
            return

        if command in {"frontflip", "backflip", "halfflip", "wavedash", "jump", "doublejump"}:
            if self._active_command is not None:
                print("[Nexto commands] A command is already active. Use `cancel` first.")
                return
            try:
                car = packet.game_cars[self.index]
                on_ground = bool(car.has_wheel_contact)
            except (AttributeError, IndexError, TypeError):
                on_ground = False
            if not on_ground:
                print(f"[Nexto commands] {command} ignored: car must be on the ground.")
                return
            if bool(packet.game_info.is_kickoff_pause):
                print(f"[Nexto commands] {command} ignored during kickoff pause.")
                return
            self._mode = "command"
            self._active_command = command
            now = float(packet.game_info.seconds_elapsed)
            self._command_start_time = now
            self._move_phase_start_time = now
            self._move_airborne_seen = False
            try:
                self._move_base_z = float(car.physics.location.z)
            except (AttributeError, TypeError, ValueError):
                self._move_base_z = 17.0
            if command == "halfflip":
                self._move_phase = "reverse"
            elif command == "wavedash":
                self._move_phase = "wave_jump"
            else:
                self._move_phase = None
            self._idle_brake_until = 0.0
            print(f"[Nexto commands] {command} started. Nexto will wait after the move.")
            return

        print(f"[Nexto commands] Unknown command {command!r}; type `help`.")

    def _set_move_phase(self, phase, now):
        self._move_phase = phase
        self._move_phase_start_time = float(now)

    @staticmethod
    def _car_up_z(car):
        """Approximate world-up component of the car's roof-up vector."""
        try:
            rotation = car.physics.rotation
            return math.cos(float(rotation.pitch)) * math.cos(float(rotation.roll))
        except (AttributeError, TypeError, ValueError):
            return 1.0

    def _half_flip_controls(self, packet):
        now = float(packet.game_info.seconds_elapsed)
        phase_elapsed = now - self._move_phase_start_time
        total_elapsed = now - self._command_start_time
        car = packet.game_cars[self.index]
        invertedness = self._car_up_z(car)  # below 0 means the roof is pointing more toward the floor

        if total_elapsed > HALF_FLIP_TOTAL_TIMEOUT:
            self._enter_idle(packet, "Half-flip timed out; waiting for the next command.")
            return None

        if self._move_phase == "reverse":
            if phase_elapsed >= HALF_FLIP_REVERSE_TIME:
                self._set_move_phase("first_jump", now)
            return SimpleControllerState(throttle=-1.0)

        if self._move_phase == "first_jump":
            if phase_elapsed >= HALF_FLIP_FIRST_JUMP_TIME:
                self._set_move_phase("jump_release", now)
            return SimpleControllerState(throttle=-1.0, jump=True)

        if self._move_phase == "jump_release":
            if phase_elapsed >= HALF_FLIP_JUMP_RELEASE_TIME:
                self._set_move_phase("backflip", now)
            return SimpleControllerState(throttle=-1.0, jump=False)

        if self._move_phase == "backflip":
            # A second jump press with pitch back starts the backflip.
            if phase_elapsed >= HALF_FLIP_BACKFLIP_INPUT_TIME:
                self._set_move_phase("wait_inverted", now)
            return SimpleControllerState(throttle=-1.0, pitch=1.0, jump=True)

        if self._move_phase == "wait_inverted":
            # Let the backflip rotate. Cancel as soon as the car is upside down,
            # or use a timeout fallback if the rotation signal never crosses over.
            if (phase_elapsed >= 0.06 and invertedness < -0.05) or phase_elapsed >= HALF_FLIP_INVERTED_TIMEOUT:
                self._set_move_phase("cancel_forward", now)
                print(f"[Nexto commands] Half-flip: forward cancel started (up_z={invertedness:.2f}).")
            return SimpleControllerState()

        if self._move_phase == "cancel_forward":
            if phase_elapsed >= HALF_FLIP_CANCEL_FORWARD_TIME:
                self._set_move_phase("air_roll", now)
            return SimpleControllerState(pitch=-1.0,roll=1.0)

        if self._move_phase == "air_roll":
            # Keep air-roll held until the wheels actually touch the ground.
            # Do not stop merely because the car looks upright; that can leave
            # it sideways and cause a poor landing. The overall move timeout is
            # the safety fallback if the car never lands.
            if bool(car.has_wheel_contact):
                self._enter_idle(packet, "Half-flip landed; waiting for the next command.")
                return None
            return SimpleControllerState(roll=HALF_FLIP_ROLL_DIRECTION)

        return SimpleControllerState()

    def _wavedash_controls(self, packet):
        now = float(packet.game_info.seconds_elapsed)
        phase_elapsed = now - self._move_phase_start_time
        total_elapsed = now - self._command_start_time
        car = packet.game_cars[self.index]
        on_ground = bool(car.has_wheel_contact)
        try:
            z = float(car.physics.location.z)
            vertical_speed = float(car.physics.velocity.z)
        except (AttributeError, TypeError, ValueError):
            z, vertical_speed = self._move_base_z + 999.0, 0.0

        if total_elapsed > WAVEDASH_TOTAL_TIMEOUT:
            self._enter_idle(packet, "Wave-dash timed out; waiting for the next command.")
            return None

        if self._move_phase == "wave_jump":
            if phase_elapsed >= WAVEDASH_JUMP_TIME:
                self._set_move_phase("wave_release", now)
            return SimpleControllerState(throttle=1.0, jump=True)

        if self._move_phase == "wave_release":
            if phase_elapsed >= WAVEDASH_JUMP_RELEASE_TIME:
                self._set_move_phase("wave_tilt", now)
            return SimpleControllerState(throttle=1.0, jump=False)

        if self._move_phase == "wave_tilt":
            if not on_ground:
                self._move_airborne_seen = True

            # Trigger before wheel contact, not at contact. The player's car
            # location is near its center, so a larger height margin is needed
            # than the apparent wheel-to-ground gap. This is an approximation:
            # RLBot does not expose separate front/rear wheel contact here.
            height_above_start = z - self._move_base_z
            near_landing = height_above_start <= WAVEDASH_TRIGGER_HEIGHT_ABOVE_START
            descending = vertical_speed < -WAVEDASH_MIN_DESCENDING_SPEED
            if self._move_airborne_seen and descending and near_landing:
                self._set_move_phase("wave_forward_dodge", now)
                print(
                    f"[Nexto commands] Wave-dash: FORWARD DODGE input started "
                    f"(height_above_start={height_above_start:.1f}, vz={vertical_speed:.1f}). "
                    f"If no dodge occurs, adjust WAVEDASH_TRIGGER_HEIGHT_ABOVE_START."
                )
                return SimpleControllerState(throttle=1.0, pitch=-1.0, jump=True)

            if self._move_airborne_seen and on_ground:
                self._enter_idle(
                    packet,
                    f"Wave-dash missed the dodge window (height_above_start={height_above_start:.1f}, "
                    f"vz={vertical_speed:.1f}); raise WAVEDASH_TRIGGER_HEIGHT_ABOVE_START."
                )
                return None
            if phase_elapsed > 1.15:
                self._enter_idle(
                    packet,
                    f"Wave-dash did not reach its dodge trigger (height_above_start={height_above_start:.1f}, "
                    f"vz={vertical_speed:.1f}); check jump height and trigger settings."
                )
                return None
            return SimpleControllerState(throttle=1.0, pitch=WAVEDASH_NOSE_UP_PITCH)

        if self._move_phase == "wave_forward_dodge":
            if phase_elapsed >= WAVEDASH_FORWARD_DODGE_TIME:
                self._set_move_phase("wave_landing", now)
            return SimpleControllerState(throttle=1.0, pitch=-1.0, jump=True)

        if self._move_phase == "wave_landing":
            if on_ground:
                self._enter_idle(packet, "Wave-dash landed; waiting for the next command.")
                return None
            if phase_elapsed >= WAVEDASH_LANDING_TIMEOUT:
                self._enter_idle(packet, "Wave-dash sequence finished; waiting for the next command.")
                return None
            return SimpleControllerState(throttle=1.0)

        return SimpleControllerState()

    def _command_controls(self, packet):
        command = self._active_command
        if command == "halfflip":
            return self._half_flip_controls(packet)
        if command == "wavedash":
            return self._wavedash_controls(packet)
        if command not in {"frontflip", "backflip", "halfflip", "wavedash", "jump", "doublejump"}:
            return None

        elapsed = float(packet.game_info.seconds_elapsed) - self._command_start_time

        if command == "jump":
            if elapsed < 0.12:
                return SimpleControllerState(throttle=1.0, jump=True)
            self._enter_idle(packet, "Jump finished.")
            return None

        if command == "doublejump":
            if elapsed < 0.10:
                return SimpleControllerState(throttle=1.0, jump=True)
            if elapsed < 0.17:
                return SimpleControllerState(throttle=1.0, jump=False)
            if elapsed < 0.29:
                return SimpleControllerState(throttle=1.0, jump=True)
            self._enter_idle(packet, "Double-jump sequence ended.")
            return None

        if command in {"frontflip", "backflip"}:
            # Basic dodge: jump, release jump, then jump + pitch.
            pitch = -1.0 if command == "frontflip" else 1.0
            if elapsed < 0.10:
                return SimpleControllerState(throttle=1.0, jump=True)
            if elapsed < 0.17:
                return SimpleControllerState(throttle=1.0, jump=False)
            if elapsed < 0.31:
                return SimpleControllerState(throttle=1.0, pitch=pitch, jump=True)
            self._enter_idle(packet, f"{command.capitalize()} sequence ended.")
            return None


    def _idle_controls(self, packet):
        """No AI steering while waiting; briefly brake in the direction needed."""
        try:
            car = packet.game_cars[self.index]
            now = float(packet.game_info.seconds_elapsed)
            on_ground = bool(car.has_wheel_contact)
            if on_ground and now < self._idle_brake_until:
                velocity = car.physics.velocity
                yaw = float(car.physics.rotation.yaw)
                forward_speed = float(velocity.x) * math.cos(yaw) + float(velocity.y) * math.sin(yaw)
                if forward_speed > 120.0:
                    return SimpleControllerState(throttle=-1.0)
                if forward_speed < -120.0:
                    return SimpleControllerState(throttle=1.0)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return SimpleControllerState()

    def _check_score_attempt(self, packet):
        if self._active_command != "score":
            return
        try:
            now = float(packet.game_info.seconds_elapsed)
            scores = (int(packet.teams[0].score), int(packet.teams[1].score))
        except (AttributeError, IndexError, TypeError, ValueError):
            try:
                now = float(packet.game_info.seconds_elapsed)
            except (AttributeError, TypeError, ValueError):
                return
            scores = None

        if self._score_baseline is not None and scores is not None and scores != self._score_baseline:
            self._enter_idle(packet, "A goal was scored; score command finished.")
        elif now - self._command_start_time >= self._score_timeout:
            self._enter_idle(packet, "Score attempt timed out.")

    def _check_hit_attempt(self, packet):
        if self._active_command != "hit":
            return
        try:
            now = float(packet.game_info.seconds_elapsed)
            touch = packet.game_ball.latest_touch
            touch_time = float(touch.time_seconds)
            touch_player = int(touch.player_index)
        except (AttributeError, TypeError, ValueError):
            try:
                now = float(packet.game_info.seconds_elapsed)
            except (AttributeError, TypeError, ValueError):
                return
            touch_time = None
            touch_player = -1

        if (
            self._hit_baseline_touch_time is not None
            and touch_time is not None
            and touch_time > self._hit_baseline_touch_time + 0.001
            and touch_player == self.index
        ):
            self._enter_idle(packet, "Nexto touched the ball; hit command finished.")
        elif now - self._command_start_time >= self._hit_timeout:
            self._enter_idle(packet, "Hit command timed out before Nexto touched the ball.")

    def get_output(self, packet):
        # Poll first so commands can take effect immediately. Do NOT call the
        # Nexto policy unless autonomous mode is explicitly enabled.
        self._poll_command(packet)
        self._check_score_attempt(packet)
        self._check_hit_attempt(packet)
        override = self._command_controls(packet)
        if override is not None:
            return override
        if self._mode == "autonomous":
            return super().get_output(packet)
        return self._idle_controls(packet)
