"""
Platform-agnostic abstractions for post / account status checking.

Each platform plugin subclasses ``PlatformChecker`` and implements
``check_post`` / ``check_account``. Status codes are unified across platforms
so output rows are comparable regardless of source. A platform that doesn't
support one of the modes can simply leave the corresponding ``check_*`` to
raise :class:`NotSupported`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# -- Unified status codes ------------------------------------------------

STATUS_LIVE = "live"
STATUS_GONE = "gone"                                # post-only, collapsed verdict
STATUS_PROTECTED = "protected"                      # account-only
STATUS_SUSPENDED_OR_DEACTIVATED = "suspended_or_deactivated"
STATUS_SUSPENDED = "suspended"                      # only when we can confirm
STATUS_DEACTIVATED = "deactivated"                  # only when we can confirm
STATUS_NOT_FOUND = "not_found"
STATUS_UNKNOWN = "unknown"                          # probes ran but verdict ambiguous
STATUS_ERROR = "error"
STATUS_INVALID = "invalid_url"
STATUS_RATE_LIMITED = "rate_limited"                # transient throttling, may resolve on retry
STATUS_SKIPPED = "skipped"                          # never attempted (e.g. aborted batch)


@dataclass
class StatusResult:
    """One row of output. All optional metadata fields default to None so the
    DataFrame stays consistent across platforms."""

    input: str
    mode: str                                       # "post" or "account"
    platform: str
    status: str
    status_label: str
    detail: Optional[str] = None
    normalized_url: Optional[str] = None
    identifier: Optional[str] = None                # post id or username
    author_handle: Optional[str] = None
    author_name: Optional[str] = None
    author_url: Optional[str] = None
    posted_at: Optional[str] = None
    oembed_title: Optional[str] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    statuses: Optional[int] = None
    verified: Optional[bool] = None
    profile_created_at: Optional[str] = None
    bio: Optional[str] = None
    withheld_in_countries: Optional[str] = None
    error: Optional[str] = None
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class NotSupported(Exception):
    """Raised when a platform does not support the requested mode."""


class PlatformChecker:
    """Subclass and override the class attributes + the two ``check_*`` methods."""

    NAME: str = ""
    DISPLAY_NAME: str = ""
    SUPPORTS_POSTS: bool = False
    SUPPORTS_ACCOUNTS: bool = False
    HOST_PATTERNS: list[re.Pattern] = []

    def matches_url(self, url: str) -> bool:
        if not url:
            return False
        for pat in self.HOST_PATTERNS:
            if pat.search(url):
                return True
        return False

    def parse_post(self, raw: str) -> Optional[str]:
        """Return canonical post identifier (id), or None if not parseable."""
        return None

    def parse_account(self, raw: str) -> Optional[str]:
        """Return canonical username, or None if not parseable."""
        return None

    def check_post(self, raw: str) -> StatusResult:
        raise NotSupported(f"{self.DISPLAY_NAME} does not support post status checks")

    def check_account(self, raw: str) -> StatusResult:
        raise NotSupported(f"{self.DISPLAY_NAME} does not support account status checks")
