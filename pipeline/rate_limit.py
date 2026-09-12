from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class GroqRateLimitSettings:
    """Client-side guardrails for a shared Groq organization quota."""

    tokens_per_minute: int = 8_000
    utilization: float = 0.90
    window_seconds: float = 60.0
    max_429_retries: int = 4
    retry_buffer_seconds: float = 1.0
    max_retry_wait_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.tokens_per_minute < 0:
            raise ValueError("tokens_per_minute must be non-negative")
        if not 0 < self.utilization <= 1:
            raise ValueError("utilization must be in the interval (0, 1]")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.max_429_retries < 0:
            raise ValueError("max_429_retries must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.tokens_per_minute > 0

    @property
    def working_token_budget(self) -> int:
        if not self.enabled:
            return 0
        return max(1, math.floor(self.tokens_per_minute * self.utilization))

    @property
    def profile(self) -> str:
        if not self.enabled:
            return "groq-rate-limit-disabled"
        utilization = round(self.utilization * 100)
        window = f"{self.window_seconds:g}"
        return (
            f"groq-{self.tokens_per_minute}tpm-{utilization}pct-"
            f"{window}s-r{self.max_429_retries}-v1"
        )

    def as_metadata(self) -> dict[str, int | float | bool | str]:
        return {
            **asdict(self),
            "enabled": self.enabled,
            "working_token_budget": self.working_token_budget,
            "profile": self.profile,
        }


@dataclass
class _Reservation:
    started_at: float
    tokens: int
    active: bool = True


class RollingTokenRateLimiter:
    """Reserve tokens inside a conservative rolling time window.

    A reservation is made before a request and replaced with provider-reported
    usage after the response. This allows short calls to proceed sooner while
    preventing the combined estimated usage of sequential pipeline calls from
    exceeding the configured working budget.
    """

    def __init__(
        self,
        settings: GroqRateLimitSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._sleeper = sleeper
        self._reservations: list[_Reservation] = []
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.settings.window_seconds
        self._reservations = [
            reservation
            for reservation in self._reservations
            if reservation.active and reservation.started_at > cutoff
        ]

    def acquire(self, estimated_tokens: int) -> tuple[_Reservation | None, int]:
        if not self.settings.enabled:
            return None, 0

        # An estimate can exceed the conservative working budget while still
        # fitting Groq's hard TPM limit. In that case, serialize it by reserving
        # the entire working budget. A genuine oversized request will be rejected
        # by Groq and surfaced with a clear error by the normal retry path.
        reserved_tokens = min(
            max(1, int(estimated_tokens)), self.settings.working_token_budget
        )
        waited_seconds = 0.0

        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                used_tokens = sum(item.tokens for item in self._reservations)
                if used_tokens + reserved_tokens <= self.settings.working_token_budget:
                    reservation = _Reservation(now, reserved_tokens)
                    self._reservations.append(reservation)
                    return reservation, round(waited_seconds * 1_000)

                oldest = min(item.started_at for item in self._reservations)
                wait_seconds = max(
                    0.05,
                    oldest + self.settings.window_seconds - now + 0.05,
                )

            print(
                "Groq TPM guard: "
                f"waiting {wait_seconds:.1f}s "
                f"({used_tokens}/{self.settings.working_token_budget} rolling tokens used)",
                flush=True,
            )
            self._sleeper(wait_seconds)
            waited_seconds += wait_seconds

    def complete(self, reservation: _Reservation | None, actual_tokens: int | None) -> None:
        if reservation is None or actual_tokens is None:
            return
        with self._lock:
            if reservation.active:
                reservation.tokens = max(1, int(actual_tokens))

    def cancel(self, reservation: _Reservation | None) -> None:
        if reservation is None:
            return
        with self._lock:
            reservation.active = False


def _environment_settings() -> GroqRateLimitSettings:
    return GroqRateLimitSettings(
        tokens_per_minute=int(os.getenv("GROQ_TPM_LIMIT", "8000")),
        utilization=float(os.getenv("GROQ_RATE_LIMIT_UTILIZATION", "0.90")),
        window_seconds=float(os.getenv("GROQ_RATE_LIMIT_WINDOW_SECONDS", "60")),
        max_429_retries=int(os.getenv("GROQ_MAX_429_RETRIES", "4")),
    )


_SETTINGS = _environment_settings()
_LIMITER = RollingTokenRateLimiter(_SETTINGS)


def configure_groq_rate_limit(
    *,
    tokens_per_minute: int,
    utilization: float,
    max_429_retries: int,
) -> GroqRateLimitSettings:
    global _SETTINGS, _LIMITER
    _SETTINGS = GroqRateLimitSettings(
        tokens_per_minute=tokens_per_minute,
        utilization=utilization,
        max_429_retries=max_429_retries,
    )
    _LIMITER = RollingTokenRateLimiter(_SETTINGS)
    return _SETTINGS


def get_groq_rate_limit_settings() -> GroqRateLimitSettings:
    return _SETTINGS


def reserve_groq_tokens(estimated_tokens: int) -> tuple[_Reservation | None, int]:
    return _LIMITER.acquire(estimated_tokens)


def complete_groq_reservation(
    reservation: _Reservation | None, actual_tokens: int | None
) -> None:
    _LIMITER.complete(reservation, actual_tokens)


def cancel_groq_reservation(reservation: _Reservation | None) -> None:
    _LIMITER.cancel(reservation)


def is_rate_limit_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    message = str(error).lower()
    return (
        status_code == 429
        or response_status == 429
        or "429" in message
        or "rate_limit_exceeded" in message
        or ("rate limit" in message and "exceed" in message)
    )


def is_daily_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return bool(
        re.search(
            r"\b(?:tokens?|requests?)\s+per\s+day\b"
            r"|\b(?:tokens?|requests?)_per_day\b"
            r"|\b(?:tpd|rpd)\b",
            message,
        )
    )


def _parse_duration(value: str) -> float | None:
    text = value.strip().lower()
    try:
        return float(text)
    except ValueError:
        pass

    match = re.fullmatch(
        r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
        text,
    )
    if not match or not any(match.groupdict().values()):
        return None
    return (
        float(match.group("hours") or 0) * 3_600
        + float(match.group("minutes") or 0) * 60
        + float(match.group("seconds") or 0)
    )


def retry_after_seconds(error: Exception, attempt: int) -> float:
    settings = get_groq_rate_limit_settings()
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    for header in ("retry-after", "x-ratelimit-reset-tokens"):
        value = headers.get(header) if hasattr(headers, "get") else None
        if value is not None:
            parsed = _parse_duration(str(value))
            if parsed is not None:
                return min(
                    parsed + settings.retry_buffer_seconds,
                    settings.max_retry_wait_seconds,
                )

    message = str(error).lower()
    match = re.search(
        r"(?:try again in|retry after)\s+(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds?|m|min|minutes?)",
        message,
    )
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == "ms":
            value /= 1_000
        elif unit.startswith("m"):
            value *= 60
        return min(
            value + settings.retry_buffer_seconds,
            settings.max_retry_wait_seconds,
        )

    fallback = settings.window_seconds * (2 ** max(0, attempt - 1))
    return min(fallback, settings.max_retry_wait_seconds)
