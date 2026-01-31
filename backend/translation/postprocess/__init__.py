"""
Translation Post-processing Module

Handles post-translation processing:
1. Subtitle length control (auto-split long lines)
2. Number/unit localization
3. Consistency checks
"""
from .length_control import SubtitleLengthController, LengthControlConfig
from .localizer import NumberUnitLocalizer, LocalizationConfig

__all__ = [
    "SubtitleLengthController",
    "LengthControlConfig",
    "NumberUnitLocalizer",
    "LocalizationConfig",
]
