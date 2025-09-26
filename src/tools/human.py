from __future__ import annotations

from typing import Optional


def human_confirm(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        resp = input(f"{prompt} {suffix} ").strip().lower()
        if not resp:
            return default
        if resp in {"y", "yes"}:
            return True
        if resp in {"n", "no"}:
            return False
        print("Please answer y or n.")


def human_input(prompt: str) -> str:
    return input(f"{prompt} ")

