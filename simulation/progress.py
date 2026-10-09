"""Configurable console progress behavior."""

import config


def get_progress_mode() -> str:
    """Return the normalized configured progress mode."""
    return config.PROGRESS_MODE.strip().lower()


def should_print_progress(
    completed_items: int,
    total_items: int,
    coarse_interval: int,
) -> bool:
    """Return True when the current progress mode should print this update."""
    mode = get_progress_mode()
    if mode == "silent":
        return False
    if mode == "verbose":
        return True

    return (
        completed_items == 1
        or completed_items == total_items
        or completed_items % coarse_interval == 0
    )


def should_print_summary() -> bool:
    """Return True unless console output is fully silent."""
    return get_progress_mode() != "silent"
