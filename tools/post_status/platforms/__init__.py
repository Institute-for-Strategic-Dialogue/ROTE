"""
Platform registry for post / account status checks.

Adding a new platform:
  1. Create ``platforms/<name>.py`` that subclasses ``PlatformChecker``.
  2. Import it here and add an instance to ``PLATFORMS``.
  3. The UI will list it automatically.
"""
from __future__ import annotations

from typing import Optional

from .base import (
    NotSupported,
    PlatformChecker,
    StatusResult,
    STATUS_DEACTIVATED,
    STATUS_ERROR,
    STATUS_GONE,
    STATUS_INVALID,
    STATUS_LIVE,
    STATUS_NOT_FOUND,
    STATUS_PROTECTED,
    STATUS_RATE_LIMITED,
    STATUS_SUSPENDED,
    STATUS_SUSPENDED_OR_DEACTIVATED,
    STATUS_UNKNOWN,
)
from .x import XChecker

# Stable platform identifiers used in CSV inputs and the UI dropdown.
PLATFORMS: dict[str, PlatformChecker] = {
    "x": XChecker(),
}


def get_platform(name: str) -> Optional[PlatformChecker]:
    if not name:
        return None
    return PLATFORMS.get(name.strip().lower())


def detect_platform_from_url(url: str) -> Optional[str]:
    """Return platform name when the URL clearly belongs to one of our platforms."""
    if not url:
        return None
    for name, checker in PLATFORMS.items():
        if checker.matches_url(url):
            return name
    return None


def list_platforms() -> list[dict]:
    return [
        {
            "name": p.NAME,
            "display_name": p.DISPLAY_NAME,
            "supports_posts": p.SUPPORTS_POSTS,
            "supports_accounts": p.SUPPORTS_ACCOUNTS,
        }
        for p in PLATFORMS.values()
    ]


__all__ = [
    "PLATFORMS",
    "PlatformChecker",
    "StatusResult",
    "NotSupported",
    "get_platform",
    "detect_platform_from_url",
    "list_platforms",
    "STATUS_LIVE",
    "STATUS_GONE",
    "STATUS_PROTECTED",
    "STATUS_SUSPENDED_OR_DEACTIVATED",
    "STATUS_SUSPENDED",
    "STATUS_DEACTIVATED",
    "STATUS_NOT_FOUND",
    "STATUS_UNKNOWN",
    "STATUS_ERROR",
    "STATUS_INVALID",
    "STATUS_RATE_LIMITED",
]
