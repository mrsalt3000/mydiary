"""Core configuration helpers for the Local Diary application (Task 03).

Public surface: application settings and the cached settings accessor.
Model-profile configuration and diagnostics live in sibling modules
(``diary.core.models_config``, ``diary.core.profile_status``).
"""

from diary.core.settings import (
    PROJECT_ROOT,
    AppSettings,
    get_settings,
    reload_settings,
)

__all__ = ["PROJECT_ROOT", "AppSettings", "get_settings", "reload_settings"]
