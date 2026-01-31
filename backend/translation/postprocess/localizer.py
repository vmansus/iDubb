"""
Number and Unit Localizer

Localizes numbers and units for the target language/region.

Examples (English → Chinese):
- $100 → 100美元
- 10 miles → 16公里  
- 6 feet → 1.83米
- 150 lbs → 68公斤
- 70°F → 21°C
"""
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class LocalizationConfig:
    """Configuration for number/unit localization"""
    # Target region settings
    target_region: str = "zh-CN"  # zh-CN, zh-TW, en-US, etc.
    
    # What to localize
    localize_currency: bool = True
    localize_distance: bool = True
    localize_weight: bool = True
    localize_temperature: bool = True
    localize_volume: bool = True
    localize_time: bool = True
    
    # Format preferences
    use_chinese_numbers: bool = False  # 100 vs 一百
    decimal_places: int = 2
    
    # Keep original in parentheses
    keep_original: bool = False  # "16公里 (10 miles)"


@dataclass
class LocalizationResult:
    """Result of localization processing"""
    original: str
    processed: str
    was_modified: bool
    conversions: List[Dict[str, Any]]  # List of conversions made


class NumberUnitLocalizer:
    """
    Localizes numbers and units in translated text.
    
    Supports:
    - Currency ($, €, £, ¥)
    - Distance (miles, feet, inches, yards)
    - Weight (pounds, ounces)
    - Temperature (°F)
    - Volume (gallons, fluid ounces)
    - Time formats
    """

    # Currency patterns and conversions
    CURRENCY_PATTERNS = {
        'usd': (r'\$\s*([\d,]+(?:\.\d{2})?)', '{amount}美元'),
        'usd_cents': (r'([\d,]+)\s*cents?', '{amount}美分'),
        'eur': (r'€\s*([\d,]+(?:\.\d{2})?)', '{amount}欧元'),
        'gbp': (r'£\s*([\d,]+(?:\.\d{2})?)', '{amount}英镑'),
        'usd_word': (r'([\d,]+(?:\.\d{2})?)\s*dollars?', '{amount}美元'),
        'usd_million': (r'\$([\d.]+)\s*million', '{amount}百万美元'),
        'usd_billion': (r'\$([\d.]+)\s*billion', '{amount}十亿美元'),
    }

    # Distance conversions (to metric)
    DISTANCE_CONVERSIONS = {
        'miles': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*miles?',
            'factor': 1.60934,
            'unit': '公里',
        },
        'feet': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*(?:feet|foot|ft)',
            'factor': 0.3048,
            'unit': '米',
        },
        'inches': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*(?:inches|inch|in)',
            'factor': 2.54,
            'unit': '厘米',
        },
        'yards': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*yards?',
            'factor': 0.9144,
            'unit': '米',
        },
    }

    # Weight conversions (to metric)
    WEIGHT_CONVERSIONS = {
        'pounds': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*(?:pounds?|lbs?)',
            'factor': 0.453592,
            'unit': '公斤',
        },
        'ounces': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*(?:ounces?|oz)',
            'factor': 28.3495,
            'unit': '克',
        },
    }

    # Temperature conversion (F to C)
    TEMPERATURE_PATTERN = r'([\d,]+(?:\.\d+)?)\s*°?\s*F(?:ahrenheit)?'
    
    # Volume conversions
    VOLUME_CONVERSIONS = {
        'gallons': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*gallons?',
            'factor': 3.78541,
            'unit': '升',
        },
        'fl_oz': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*(?:fl\.?\s*oz|fluid\s*ounces?)',
            'factor': 29.5735,
            'unit': '毫升',
        },
        'pints': {
            'pattern': r'([\d,]+(?:\.\d+)?)\s*pints?',
            'factor': 0.473176,
            'unit': '升',
        },
    }

    # Time format patterns
    TIME_PATTERNS = {
        'am_pm': (r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)', None),  # 3:30 PM → 15:30
        '12h': (r'(\d{1,2})\s*(AM|PM|am|pm)', None),  # 3 PM → 15:00
    }

    def __init__(self, config: Optional[LocalizationConfig] = None):
        self.config = config or LocalizationConfig()

    def process(self, text: str) -> LocalizationResult:
        """
        Process text to localize numbers and units.
        
        Args:
            text: Input text
            
        Returns:
            LocalizationResult with processed text
        """
        original = text
        processed = text
        conversions = []

        # Currency
        if self.config.localize_currency:
            processed, currency_convs = self._localize_currency(processed)
            conversions.extend(currency_convs)

        # Distance
        if self.config.localize_distance:
            processed, distance_convs = self._localize_units(
                processed, self.DISTANCE_CONVERSIONS
            )
            conversions.extend(distance_convs)

        # Weight
        if self.config.localize_weight:
            processed, weight_convs = self._localize_units(
                processed, self.WEIGHT_CONVERSIONS
            )
            conversions.extend(weight_convs)

        # Temperature
        if self.config.localize_temperature:
            processed, temp_convs = self._localize_temperature(processed)
            conversions.extend(temp_convs)

        # Volume
        if self.config.localize_volume:
            processed, volume_convs = self._localize_units(
                processed, self.VOLUME_CONVERSIONS
            )
            conversions.extend(volume_convs)

        # Time
        if self.config.localize_time:
            processed, time_convs = self._localize_time(processed)
            conversions.extend(time_convs)

        return LocalizationResult(
            original=original,
            processed=processed,
            was_modified=processed != original,
            conversions=conversions
        )

    def _parse_number(self, num_str: str) -> float:
        """Parse a number string with commas"""
        return float(num_str.replace(',', ''))

    def _format_number(self, num: float) -> str:
        """Format a number for display"""
        if num == int(num):
            return str(int(num))
        return f"{num:.{self.config.decimal_places}f}".rstrip('0').rstrip('.')

    def _localize_currency(self, text: str) -> Tuple[str, List[Dict]]:
        """Localize currency mentions"""
        conversions = []
        
        for currency_type, (pattern, template) in self.CURRENCY_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                original_match = match.group(0)
                amount = self._parse_number(match.group(1))
                
                # Format the localized version
                formatted_amount = self._format_number(amount)
                localized = template.format(amount=formatted_amount)
                
                if self.config.keep_original:
                    localized = f"{localized} ({original_match})"
                
                text = text.replace(original_match, localized, 1)
                conversions.append({
                    'type': 'currency',
                    'original': original_match,
                    'localized': localized,
                })
        
        return text, conversions

    def _localize_units(
        self, 
        text: str, 
        conversions_map: Dict
    ) -> Tuple[str, List[Dict]]:
        """Localize unit measurements"""
        conversions = []
        
        for unit_type, config in conversions_map.items():
            pattern = config['pattern']
            factor = config['factor']
            target_unit = config['unit']
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                original_match = match.group(0)
                value = self._parse_number(match.group(1))
                
                # Convert
                converted = value * factor
                formatted = self._format_number(converted)
                localized = f"{formatted}{target_unit}"
                
                if self.config.keep_original:
                    localized = f"{localized} ({original_match})"
                
                text = text.replace(original_match, localized, 1)
                conversions.append({
                    'type': unit_type,
                    'original': original_match,
                    'value': value,
                    'converted': converted,
                    'localized': localized,
                })
        
        return text, conversions

    def _localize_temperature(self, text: str) -> Tuple[str, List[Dict]]:
        """Convert Fahrenheit to Celsius"""
        conversions = []
        
        for match in re.finditer(self.TEMPERATURE_PATTERN, text, re.IGNORECASE):
            original_match = match.group(0)
            fahrenheit = self._parse_number(match.group(1))
            
            # F to C conversion
            celsius = (fahrenheit - 32) * 5 / 9
            formatted = self._format_number(celsius)
            localized = f"{formatted}°C"
            
            if self.config.keep_original:
                localized = f"{localized} ({original_match})"
            
            text = text.replace(original_match, localized, 1)
            conversions.append({
                'type': 'temperature',
                'original': original_match,
                'fahrenheit': fahrenheit,
                'celsius': celsius,
                'localized': localized,
            })
        
        return text, conversions

    def _localize_time(self, text: str) -> Tuple[str, List[Dict]]:
        """Convert 12-hour time to 24-hour format"""
        conversions = []
        
        # Pattern for HH:MM AM/PM
        pattern_full = r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)'
        for match in re.finditer(pattern_full, text):
            original_match = match.group(0)
            hour = int(match.group(1))
            minute = int(match.group(2))
            period = match.group(3).upper()
            
            # Convert to 24-hour
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
            
            localized = f"{hour:02d}:{minute:02d}"
            
            text = text.replace(original_match, localized, 1)
            conversions.append({
                'type': 'time',
                'original': original_match,
                'localized': localized,
            })
        
        # Pattern for H AM/PM (no minutes)
        pattern_simple = r'(\d{1,2})\s*(AM|PM|am|pm)(?!\s*[:.\d])'
        for match in re.finditer(pattern_simple, text):
            original_match = match.group(0)
            hour = int(match.group(1))
            period = match.group(2).upper()
            
            if period == 'PM' and hour != 12:
                hour += 12
            elif period == 'AM' and hour == 12:
                hour = 0
            
            localized = f"{hour}点"
            
            text = text.replace(original_match, localized, 1)
            conversions.append({
                'type': 'time',
                'original': original_match,
                'localized': localized,
            })
        
        return text, conversions

    def process_batch(self, texts: List[str]) -> List[LocalizationResult]:
        """Process multiple texts"""
        return [self.process(text) for text in texts]


# Convenience function
def localize_text(
    text: str,
    target_region: str = "zh-CN",
    keep_original: bool = False
) -> str:
    """
    Quick function to localize numbers and units in text.
    
    Args:
        text: Input text
        target_region: Target region code
        keep_original: Whether to keep original in parentheses
        
    Returns:
        Localized text
    """
    config = LocalizationConfig(
        target_region=target_region,
        keep_original=keep_original
    )
    localizer = NumberUnitLocalizer(config)
    result = localizer.process(text)
    return result.processed
