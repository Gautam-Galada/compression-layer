#!/usr/bin/env python3
"""
Dataset Manager
Manages swapping between sanitized and original training datasets.

Usage:
    python dataset_manager.py --update    # Switch to sanitized dataset
    python dataset_manager.py --revert    # Switch back to original dataset
    python dataset_manager.py --status    # Check current state
    python dataset_manager.py --log       # View change history
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Main training file (the one used by training scripts)
    "active_train": Path("data/training/train.jsonl"),
    # Backup of original data
    "original_backup": Path("data/training/train.original.jsonl"),
    # Sanitized data
    "sanitized_data": Path("data/training/sanitized_train.jsonl"),
    # State and log files
    "state_file": Path("data/training/.dataset_state.json"),
    "log_file": Path("data/training/.dataset_changes.log"),
}


# ============================================================================
# STATE MANAGEMENT
# ============================================================================


def load_state() -> dict:
    """Load current dataset state."""
    if CONFIG["state_file"].exists():
        with open(CONFIG["state_file"]) as f:
            return json.load(f)

    return {
        "current": "original",  # 'original' or 'sanitized'
        "last_change": None,
        "change_count": 0,
    }


def save_state(state: dict):
    """Save dataset state."""
    CONFIG["state_file"].parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["state_file"], "w") as f:
        json.dump(state, f, indent=2)


def log_change(action: str, from_state: str, to_state: str, details: str = ""):
    """Log a dataset change."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"[{timestamp}] {action}: {from_state} → {to_state}"
    if details:
        log_entry += f" | {details}"
    log_entry += "\n"

    CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["log_file"], "a") as f:
        f.write(log_entry)

    print(f"✓ Logged: {log_entry.strip()}")


# ============================================================================
# FILE OPERATIONS
# ============================================================================


def count_samples(path):
    """
    Count number of JSON objects in a JSONL file.
    """
    path = Path(path)

    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue  # skip malformed lines safely

    return count


def verify_files_exist() -> bool:
    """Verify required files exist."""
    errors = []

    if not CONFIG["active_train"].exists():
        errors.append(f"❌ Active training file not found: {CONFIG['active_train']}")

    if not CONFIG["sanitized_data"].exists():
        errors.append(f"❌ Sanitized data not found: {CONFIG['sanitized_data']}")

    if errors:
        print("\n".join(errors))
        print("\nPlease run sanitize_training_data_v2.py first to generate sanitized data.")
        return False

    return True


def backup_original():
    """Create backup of original data if it doesn't exist."""
    if not CONFIG["original_backup"].exists() and CONFIG["active_train"].exists():
        print("Creating backup of original data...")
        shutil.copy2(CONFIG["active_train"], CONFIG["original_backup"])
        print(f"✓ Backup created: {CONFIG['original_backup']}")
        return True
    return False


# ============================================================================
# MAIN OPERATIONS
# ============================================================================


def update_to_sanitized() -> bool:
    """Switch to sanitized dataset."""
    state = load_state()

    if state["current"] == "sanitized":
        print("⚠ Already using sanitized dataset. No changes made.")
        return False

    if not verify_files_exist():
        return False

    # Create backup of original if needed
    backup_original()

    # Get sample counts
    original_count = count_samples(CONFIG["active_train"])
    sanitized_count = count_samples(CONFIG["sanitized_data"])

    print("\nSwitching to sanitized dataset...")
    print(f"  Original samples:  {original_count}")
    print(f"  Sanitized samples: {sanitized_count}")
    print(f"  Removed samples:   {original_count - sanitized_count}")

    # Perform swap
    try:
        shutil.copy2(CONFIG["sanitized_data"], CONFIG["active_train"])

        # Update state
        state["current"] = "sanitized"
        state["last_change"] = datetime.now().isoformat()
        state["change_count"] += 1
        save_state(state)

        # Log change
        log_change(
            action="UPDATE",
            from_state="original",
            to_state="sanitized",
            details=f"{original_count} → {sanitized_count} samples",
        )

        print("\n✓ Successfully switched to sanitized dataset")
        print(f"✓ {CONFIG['active_train']} now contains {sanitized_count} samples")
        return True

    except Exception as e:
        print(f"\n❌ Error during update: {e}")
        return False


def revert_to_original() -> bool:
    """Revert to original dataset."""
    state = load_state()

    if state["current"] == "original":
        print("⚠ Already using original dataset. No changes made.")
        return False

    if not CONFIG["original_backup"].exists():
        print(f"❌ Original backup not found: {CONFIG['original_backup']}")
        print("Cannot revert without backup.")
        return False

    # Get sample counts
    sanitized_count = count_samples(CONFIG["active_train"])
    original_count = count_samples(CONFIG["original_backup"])

    print("\nReverting to original dataset...")
    print(f"  Sanitized samples: {sanitized_count}")
    print(f"  Original samples:  {original_count}")

    # Perform swap
    try:
        shutil.copy2(CONFIG["original_backup"], CONFIG["active_train"])

        # Update state
        state["current"] = "original"
        state["last_change"] = datetime.now().isoformat()
        state["change_count"] += 1
        save_state(state)

        # Log change
        log_change(
            action="REVERT",
            from_state="sanitized",
            to_state="original",
            details=f"{sanitized_count} → {original_count} samples",
        )

        print("\n✓ Successfully reverted to original dataset")
        print(f"✓ {CONFIG['active_train']} now contains {original_count} samples")
        return True

    except Exception as e:
        print(f"\n❌ Error during revert: {e}")
        return False


def show_status():
    """Show current dataset status."""
    state = load_state()

    print("\n" + "=" * 80)
    print("DATASET STATUS")
    print("=" * 80)
    print()

    # Current state
    current = state["current"].upper()
    print(f"Current dataset:    {current}")
    print(f"Total changes:      {state['change_count']}")

    if state["last_change"]:
        last_change = datetime.fromisoformat(state["last_change"])
        print(f"Last change:        {last_change.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Last change:        Never")

    print()

    # File information
    print("Files:")

    if CONFIG["active_train"].exists():
        active_count = count_samples(CONFIG["active_train"])
        print(f"  ✓ train.jsonl:              {active_count:4d} samples (ACTIVE)")
    else:
        print("  ❌ train.jsonl:              Not found")

    if CONFIG["original_backup"].exists():
        original_count = count_samples(CONFIG["original_backup"])
        print(f"  ✓ train.original.jsonl:     {original_count:4d} samples (backup)")
    else:
        print("  ⚠ train.original.jsonl:     Not found (will be created on first update)")

    if CONFIG["sanitized_data"].exists():
        sanitized_count = count_samples(CONFIG["sanitized_data"])
        print(f"  ✓ sanitized_train.jsonl:    {sanitized_count:4d} samples")
    else:
        print("  ❌ sanitized_train.jsonl:    Not found")

    print()

    # Recommendations
    print("=" * 80)
    print("ACTIONS")
    print("=" * 80)
    print()

    if state["current"] == "original":
        print("To switch to sanitized dataset:")
        print("  python dataset_manager.py --update")
    else:
        print("To revert to original dataset:")
        print("  python dataset_manager.py --revert")

    print()
    print("To view change history:")
    print("  python dataset_manager.py --log")
    print()


def show_log(lines: int | None = None):
    """Show change log."""
    if not CONFIG["log_file"].exists():
        print("No changes logged yet.")
        return

    print("\n" + "=" * 80)
    print("DATASET CHANGE LOG")
    print("=" * 80)
    print()

    with open(CONFIG["log_file"]) as f:
        log_lines = f.readlines()

    # Show last N lines if specified
    if lines:
        log_lines = log_lines[-lines:]

    if not log_lines:
        print("No changes logged yet.")
        return

    for line in log_lines:
        print(line.rstrip())

    print()
    print(f"Total entries: {len(log_lines)}")
    print()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Manage dataset switching between sanitized and original training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Switch to sanitized dataset
  python dataset_manager.py --update
  
  # Revert to original dataset
  python dataset_manager.py --revert
  
  # Check current status
  python dataset_manager.py --status
  
  # View full change log
  python dataset_manager.py --log
  
  # View last 10 changes
  python dataset_manager.py --log --lines 10
        """,
    )

    parser.add_argument("--update", action="store_true", help="Switch to sanitized dataset")

    parser.add_argument("--revert", action="store_true", help="Revert to original dataset")

    parser.add_argument("--status", action="store_true", help="Show current dataset status")

    parser.add_argument("--log", action="store_true", help="Show change log")

    parser.add_argument(
        "--lines", type=int, metavar="N", help="Show last N log entries (use with --log)"
    )

    args = parser.parse_args()

    # Execute requested action
    if args.update:
        update_to_sanitized()
    elif args.revert:
        revert_to_original()
    elif args.log:
        show_log(args.lines)
    elif args.status:
        show_status()
    else:
        # Default: show status
        show_status()


if __name__ == "__main__":
    main()
