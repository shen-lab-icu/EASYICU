"""R expression parsing utilities for concept dictionary.

This module handles parsing of R-style expressions found in concept dictionary
callback definitions, such as binary_op, set_val, apply_map, etc.
"""

from __future__ import annotations

import operator
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def apply_binary_op(symbol: str, series: pd.Series, value: object) -> pd.Series:
    """Apply binary operation or conversion function."""
    # Import conversion functions
    from .callback_utils import fahr_to_cels
    from .unit_conversion import celsius_to_fahrenheit, fahrenheit_to_celsius
    
    # Special case: set_val_na - set all values to NA
    if symbol == "set_val_na":
        return pd.Series([np.nan] * len(series), index=series.index)
    
    # Function map for unit conversions
    func_map = {
        "fahr_to_cels": fahr_to_cels,
        "fahrenheit_to_celsius": fahrenheit_to_celsius,
        "celsius_to_fahrenheit": celsius_to_fahrenheit,
    }
    
    # If it's a known function name, apply it
    if symbol in func_map:
        return func_map[symbol](series)
    
    # Otherwise treat as binary operator
    op_map = {
        "*": operator.mul,
        "/": operator.truediv,
        "+": operator.add,
        "-": operator.sub,
        "^": operator.pow,
    }

    if symbol not in op_map:
        raise NotImplementedError(f"Unsupported binary operator '{symbol}'")

    # Safe handling for division operations
    if symbol == "/":
        from .callback_utils import binary_op
        # Convert series to apply safe binary operation element-wise
        safe_op = binary_op(op_map[symbol], value)
        return series.apply(safe_op)
    else:
        try:
            return op_map[symbol](series, value)
        except (TypeError, ZeroDivisionError):
            return series  # Return original series on error


def parse_binary_op(expr: str) -> tuple[str, object]:
    """Parse binary_op expression.
    
    Handles:
    - binary_op(`+`, 10)
    - fahr_to_cels (function name only)
    - set_val(NA) (special: set all values to NA)
    """
    # Check for set_val(NA) - special case for convert_unit
    if re.fullmatch(r'set_val\(NA\)', expr.strip(), re.IGNORECASE):
        return 'set_val_na', None
    
    # Check if it's just a function name (like fahr_to_cels)
    if re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', expr.strip()):
        # It's a function name - return it as a special operator
        return expr.strip(), None
    
    # Otherwise parse as binary_op(symbol, value)
    match = re.fullmatch(r"binary_op\(`(.+?)`,\s*(.+)\)", expr.strip(), flags=re.DOTALL)
    if not match:
        raise NotImplementedError(f"Unsupported binary_op expression '{expr}'")
    symbol = match.group(1)
    value = parse_literal(match.group(2))
    return symbol, value


def parse_mapping(body: str) -> Dict[object, object]:
    """Parse R-style key=value mapping."""
    mapping: Dict[object, object] = {}
    for pair in split_arguments(body):
        if "=" not in pair:
            continue
        key_text, value_text = pair.split("=", 1)
        key = parse_literal(key_text.strip())
        value = parse_literal(value_text.strip())
        mapping[key] = value
    return mapping


def parse_r_arguments(expr: str) -> list:
    """Parse R function arguments."""
    return [parse_r_value(arg) for arg in split_arguments(expr)]


def parse_r_value(token: str):
    """Parse an R value (handles list(), c(), literals)."""
    text = token.strip()
    if text.startswith("list(") and text.endswith(")"):
        inner = text[5:-1]
        return [parse_r_value(arg) for arg in split_arguments(inner)]
    if text.startswith("c(") and text.endswith(")"):
        inner = text[2:-1]
        return [parse_r_value(arg) for arg in split_arguments(inner)]
    return parse_literal(text)


def split_arguments(argument_str: str) -> List[str]:
    """Split comma-separated arguments respecting parentheses nesting."""
    args: List[str] = []
    level = 0
    current: List[str] = []

    for char in argument_str:
        if char == "(":
            level += 1
        elif char == ")":
            level = max(level - 1, 0)
        elif char == "," and level == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        args.append(tail)

    return args


def strip_quotes(token: str | None) -> Optional[str]:
    """Strip quotes from a token, handling R-style escapes."""
    if token is None:
        return None
    text = token.strip()
    if text in {"NA", "NULL", ""}:
        return None
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        text = text[1:-1]
    # 🔧 FIX: 只对包含 R 风格转义序列（如 \n, \t）的字符串进行 unicode_escape 解码
    # 直接的 UTF-8 字符（如荷兰语 ï）不应该被转换
    # unicode_escape 会错误地将 UTF-8 字节解释为转义序列
    if '\\' in text:
        try:
            return text.encode("utf8").decode("unicode_escape")
        except UnicodeDecodeError:
            return text
    return text


def maybe_float(value: object) -> Optional[float]:
    """Try to convert value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def default_aggregator_for_dtype(series: pd.Series) -> str:
    """Determine default aggregator based on dtype."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "sum"
    if pd.api.types.is_numeric_dtype(dtype):
        return "median"
    return "first"


def maybe_int(value: object) -> Optional[int]:
    """Try to convert value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def maybe_timedelta(value: object) -> Optional[pd.Timedelta]:
    """Try to convert value to timedelta."""
    if value in (None, False, ""):
        return None
    if isinstance(value, pd.Timedelta):
        return value
    try:
        return pd.to_timedelta(value)
    except (TypeError, ValueError):
        return None


def parse_literal(token: str):
    """Parse an R literal value (TRUE, FALSE, NA, numbers, strings)."""
    raw = token.strip()
    if raw in {"TRUE", "True"}:
        return True
    if raw in {"FALSE", "False"}:
        return False
    if raw in {"NA", "NA_real_", "NA_integer_", "NA_character_"}:
        return pd.NA
    if raw in {"NULL", "null"}:
        return None
    # 支持反引号（R语言中用于标识符）
    if raw.startswith("`") and raw.endswith("`"):
        # 去掉反引号，然后尝试解析为数字或返回字符串
        raw = raw[1:-1]
        try:
            # 优先尝试整数，如果失败再尝试浮点数
            if "." not in raw:
                return int(raw)
            return float(raw)
        except ValueError:
            return raw
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return strip_quotes(raw)
    if raw.endswith("L"):
        raw = raw[:-1]
    try:
        # 优先尝试整数，如果失败再尝试浮点数
        if "." not in raw:
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


# Backward compatibility aliases (with underscore prefix)
_apply_binary_op = apply_binary_op
_parse_binary_op = parse_binary_op
_parse_mapping = parse_mapping
_parse_r_arguments = parse_r_arguments
_parse_r_value = parse_r_value
_split_arguments = split_arguments
_strip_quotes = strip_quotes
_maybe_float = maybe_float
_default_aggregator_for_dtype = default_aggregator_for_dtype
_maybe_int = maybe_int
_maybe_timedelta = maybe_timedelta
_parse_literal = parse_literal
