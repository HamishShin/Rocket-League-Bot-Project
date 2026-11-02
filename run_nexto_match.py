from pathlib import Path
from rlbot.setup_manager import SetupManager

def main():
    root = Path(__file__).resolve().parent
    config_path = root / "rlbot_nexto.cfg"

    if not config_path.is_file():
        raise FileNotFoundError(f"Match config not found: {config_path}")

    print(f"Loading Nexto test match: {config_path}")
    manager = SetupManager()
    manager.load_config(config_location=str(config_path))
    manager.connect_to_game()
    manager.launch_early_start_bot_processes()
    manager.start_match()
    manager.launch_bot_processes()
    manager.infinite_loop()

if __name__ == "__main__":
    main()
