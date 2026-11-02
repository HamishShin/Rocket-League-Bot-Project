"""Nexto command console with number-sequence input for Rocket League.

When RocketLeague.exe has foreground focus, type a command number and then 0
as the execute key. Example: 1 then 0 sends jump; 12 then 0 sends resume.
The low-level Windows keyboard hook consumes digit keys only while Rocket
League is focused, so the digits do not leak into the game. The normal textual
console commands remain available in this terminal.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

COMMAND_FILE = Path(__file__).resolve().parent / "command.txt"
ALLOWED = {
    "frontflip", "backflip", "halfflip", "wavedash", "jump", "doublejump", "dribble", "hit", "score",
    "stop", "wait", "cancel", "resume", "help", "quit"
}

# Numeric IDs are sent by typing the digits and then pressing 0 to execute.
# Avoid IDs ending in 0 because 0 is the execute key.
NUMBER_COMMANDS: dict[str, str] = {
    "1": "jump",
    "2": "doublejump",
    "3": "frontflip",
    "4": "backflip",
    "5": "hit",
    "6": "score",
    "7": "wait",
    "8": "cancel",
    "9": "help",
    "12": "resume", 
    "13": "halfflip",
    "14": "wavedash",
    "15": "dribble",
}

HELP_LINES = (
    "Nexto numeric commands (type digits while Rocket League is focused, then press 0 to execute):",
    "  0 alone    - immediately stop Nexto and wait",
    "  1 then 0   - jump",
    "  2 then 0   - double jump",
    "  3 then 0   - frontflip",
    "  4 then 0   - backflip",
    "  5 then 0   - hit the ball once (Nexto plays until touch or timeout)",
    "  6 then 0   - score attempt (Nexto plays until a goal or timeout)",
    "  7 then 0   - nothing",
    "  8 then 0   - nothing",
    "  9 then 0   - print this help in the command console",
    "  12 then 0  - resume autonomous Nexto control",
    "  13 then 0  - half-flip (experimental timed sequence)",
    "  14 then 0  - wave-dash (experimental timed sequence)",
    "  15 then 0  - dribble balance mode (does NOT work)",
    "",
    "Example: press 1, then 0; or press 1, 2, then 0 for command 12.",
    "Press 0 by itself at any time to send STOP; after digits, 0 executes the selected command.",
    "Optional: press '-' between the command number and 0 (for example 12-0).",
    "Backspace edits the selected number; Escape clears it. Selection expires after 5 seconds.",
    "Digits are intercepted only while RocketLeague.exe has focus. You can still type words in this console.",
    "Text commands are also accepted here: jump, doublejump, frontflip, backflip, halfflip, wavedash, dribble, hit, score, wait, cancel, resume, help, quit.",
)

_write_lock = threading.Lock()
_keyboard_ready = threading.Event()
_keyboard_thread_id: int | None = None


def send_command(command: str) -> None:
    """Atomically replace the command inbox with a unique command token."""
    command = command.strip().lower().lstrip("!")
    if command == "double jump":
        command = "doublejump"
    if command not in ALLOWED or command == "quit":
        raise ValueError(f"Unsupported command: {command!r}")

    COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = f"{command}|{time.time_ns()}\n"
    temporary = COMMAND_FILE.with_name("command.txt.tmp")
    with _write_lock:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(COMMAND_FILE)


def _print_help() -> None:
    print("\n".join(HELP_LINES), flush=True)


def _send_numbered_command(code: str) -> None:
    command = NUMBER_COMMANDS.get(code)
    if command is None:
        print(f"\n[Nexto keys] Unknown command number {code!r}. Press 9 then 0 for help.", flush=True)
        return
    try:
        if command == "help":
            _print_help()
        send_command(command)
        print(f"\n[Nexto keys] {code} -> {command} (sent).", flush=True)
    except (OSError, ValueError) as exc:
        print(f"\n[Nexto keys] Could not send {command}: {exc}", flush=True)


def _keyboard_hook_thread() -> None:
    """Install a low-level Windows hook; swallow digit input only in Rocket League."""
    global _keyboard_thread_id

    if os.name != "nt":
        print("Global number commands are Windows-only; textual console commands still work.", flush=True)
        _keyboard_ready.set()
        return

    import ctypes
    from ctypes import wintypes

    try:
        import psutil
    except ImportError:
        print("[keyboard] psutil is missing; global number commands are disabled. Install with: python -m pip install psutil", flush=True)
        _keyboard_ready.set()
        return

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    WH_KEYBOARD_LL = 13
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105
    WM_QUIT = 0x0012
    VK_BACK = 0x08
    VK_ESCAPE = 0x1B
    VK_OEM_MINUS = 0xBD
    VK_SUBTRACT = 0x6D

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [
            ("vkCode", wintypes.DWORD),
            ("scanCode", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, ctypes.c_void_p, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = ctypes.c_void_p
    user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = ctypes.c_ssize_t
    user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = ctypes.c_int
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p

    _keyboard_thread_id = int(kernel32.GetCurrentThreadId())
    sequence = ""
    sequence_started = 0.0
    pressed: set[int] = set()
    intercepted: set[int] = set()
    sequence_lock = threading.Lock()
    focus_miss_logged: set[tuple[str, str]] = set()

    def foreground_info() -> tuple[str, str]:
        """Return the current foreground process name and window title."""
        try:
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "<no foreground window>", ""
            title_len = user32.GetWindowTextLengthW(hwnd)
            title_buf = ctypes.create_unicode_buffer(max(1, title_len + 1))
            user32.GetWindowTextW(hwnd, title_buf, len(title_buf))
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process_name = psutil.Process(pid.value).name() if pid.value else "<unknown>"
            return process_name, title_buf.value
        except (psutil.Error, OSError, ValueError):
            return "<could not read foreground process>", ""

    def rocket_league_has_focus() -> bool:
        # Match by substring as well as known executable names, since different
        # launch paths/builds may expose a slightly different process name.
        process_name, title = foreground_info()
        return "rocketleague" in process_name.lower().replace(" ", "") or "rocket league" in title.lower()

    def handle_digit(digit: str) -> None:
        """Handle one intercepted digit. A bare 0 always sends stop; 0 after digits executes them."""
        nonlocal sequence, sequence_started
        command_code = None
        standalone_stop = False
        now = time.monotonic()

        # Keep selection edits atomic, and always clear the selection before
        # sending an action. This prevents a stale sequence from capturing a
        # later standalone 0.
        with sequence_lock:
            if sequence and now - sequence_started > 5.0:
                print(f"\n[Nexto keys] Number selection {sequence!r} expired; clearing it.", flush=True)
                sequence = ""
                sequence_started = 0.0

            if digit == "0":
                if sequence:
                    command_code = sequence
                    sequence = ""
                    sequence_started = 0.0
                else:
                    standalone_stop = True
            else:
                if not sequence:
                    sequence_started = now
                sequence += digit
                if len(sequence) > 3:
                    print("\n[Nexto keys] Command numbers can be at most 3 digits. Selection cleared.", flush=True)
                    sequence = ""
                    sequence_started = 0.0
                else:
                    print(f"\n[Nexto keys] Selected {sequence!r}; press 0 to execute.", flush=True)

        if command_code is not None:
            print(f"\n[Nexto keys] Execute key received; executing command {command_code!r}.", flush=True)
            _send_numbered_command(command_code)
        elif standalone_stop:
            print("\n[Nexto keys] Execute key received with no pending number; sending STOP.", flush=True)
            try:
                send_command("stop")
                print("[Nexto keys] 0 -> stop/wait sent.", flush=True)
            except (OSError, ValueError) as exc:
                print(f"[Nexto keys] Could not send stop: {exc}", flush=True)

    def callback(n_code, w_param, l_param):
        nonlocal sequence, sequence_started
        if n_code >= 0 and w_param in (WM_KEYDOWN, WM_SYSKEYDOWN, WM_KEYUP, WM_SYSKEYUP):
            info = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk = int(info.vkCode)
            is_down = w_param in (WM_KEYDOWN, WM_SYSKEYDOWN)
            is_up = w_param in (WM_KEYUP, WM_SYSKEYUP)

            # If this hook consumed the keydown while the game was focused,
            # consume its corresponding keyup too even if focus changed meanwhile.
            if is_up and vk in intercepted:
                intercepted.discard(vk)
                pressed.discard(vk)
                return 1

            game_focused = rocket_league_has_focus()
            digit = None
            if 0x30 <= vk <= 0x39:
                digit = chr(vk)
            elif 0x60 <= vk <= 0x69:
                digit = str(vk - 0x60)

            if digit is not None and is_down and not game_focused:
                # Diagnostic: the previous version silently ignored keys if the
                # actual foreground process name did not match its strict check.
                info = foreground_info()
                if info not in focus_miss_logged:
                    focus_miss_logged.add(info)
                    print(
                        f"\n[keyboard debug] Number key detected, but Rocket League focus was not recognized. "
                        f"Foreground process={info[0]!r}, window title={info[1]!r}. "
                        "Click the actual game window and try again.",
                        flush=True,
                    )

            if game_focused and digit is not None:
                if is_down:
                    if vk not in pressed:  # Ignore key-repeat while held.
                        pressed.add(vk)
                        handle_digit(digit)
                    intercepted.add(vk)
                    return 1
                if is_up:
                    pressed.discard(vk)
                    intercepted.add(vk)
                    return 1

            if game_focused and is_down:
                with sequence_lock:
                    has_sequence = bool(sequence)
                if has_sequence and vk in (VK_OEM_MINUS, VK_SUBTRACT):
                    intercepted.add(vk)
                    return 1  # Optional visual separator: 12-0.
                if has_sequence and vk == VK_BACK:
                    with sequence_lock:
                        sequence = sequence[:-1]
                        if not sequence:
                            sequence_started = 0.0
                        print(f"\n[Nexto keys] Selected {sequence!r}.", flush=True)
                    intercepted.add(vk)
                    return 1
                if has_sequence and vk == VK_ESCAPE:
                    with sequence_lock:
                        sequence = ""
                        sequence_started = 0.0
                    print("\n[Nexto keys] Number selection cleared.", flush=True)
                    intercepted.add(vk)
                    return 1
            if is_up and vk in intercepted:
                intercepted.discard(vk)
                pressed.discard(vk)
                return 1

        return user32.CallNextHookEx(hook_handle, n_code, w_param, l_param)

    hook_proc = HOOKPROC(callback)  # Keep callback alive while hook is installed.
    hook_handle = user32.SetWindowsHookExW(WH_KEYBOARD_LL, hook_proc, kernel32.GetModuleHandleW(None), 0)
    _keyboard_ready.set()
    if not hook_handle:
        error_code = ctypes.get_last_error()
        print(f"[keyboard] Could not install keyboard hook (Windows error {error_code}). Text commands still work.", flush=True)
        return

    print("[keyboard] Number-entry hook active for Rocket League.", flush=True)
    print("[keyboard] Type a command number, then press 0. Example: 1, 0 = jump; 1, 2, 0 = resume.", flush=True)
    try:
        msg = wintypes.MSG()
        while True:
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    finally:
        user32.UnhookWindowsHookEx(hook_handle)


def _stop_keyboard_thread() -> None:
    if os.name != "nt" or _keyboard_thread_id is None:
        return
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostThreadMessageW.restype = wintypes.BOOL
    user32.PostThreadMessageW(_keyboard_thread_id, 0x0012, 0, 0)  # WM_QUIT


def main() -> None:
    COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMMAND_FILE.touch(exist_ok=True)

    listener = threading.Thread(target=_keyboard_hook_thread, name="NextoNumberKeys", daemon=True)
    listener.start()
    _keyboard_ready.wait(timeout=3.0)

    print("Nexto command console is running.")
    print("Focus Rocket League and type a command number, then press 0 to execute; 0 alone stops Nexto.")
    print("Type `help` here for the command map, or enter a text command directly.")
    _print_help()

    try:
        while True:
            try:
                entered = input("Nexto command> ").strip().lower().lstrip("!")
            except (EOFError, KeyboardInterrupt):
                print("\nCommand console closing.")
                break

            command = {"double jump": "doublejump"}.get(entered, entered)
            if command == "quit":
                print("Command console closing. The match will keep running.")
                break
            if command not in ALLOWED:
                print("Unknown command. Type `help` for the available commands.")
                continue
            if command == "help":
                _print_help()
            try:
                send_command(command)
                if command != "help":
                    print(f"Sent: {command}")
            except (OSError, ValueError) as exc:
                print(f"Could not send command: {exc}")
    finally:
        _stop_keyboard_thread()


if __name__ == "__main__":
    main()
