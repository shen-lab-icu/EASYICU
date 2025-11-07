"""Callback system for concept loading (R ricu callback-cncpt.R, callback-itm.R).

Provides functions for executing callbacks during concept loading.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union
import pandas as pd

from .table import IdTbl, TsTbl, WinTbl, ICUTable
from .assertions import assert_that, is_string


def do_callback(
    callback: Optional[Callable],
    data: Union[IdTbl, TsTbl, WinTbl, pd.DataFrame],
    **kwargs
) -> Union[IdTbl, TsTbl, WinTbl, pd.DataFrame]:
    """Execute callback function (R ricu do_callback).
    
    Args:
        callback: Callback function to execute
        data: Data to process
        **kwargs: Additional arguments for callback
        
    Returns:
        Processed data
        
    Examples:
        >>> result = do_callback(transform_fun(lambda x: x * 2), data)
    """
    if callback is None:
        return data
    
    if not callable(callback):
        raise TypeError(f"Callback must be callable, got {type(callback)}")
    
    # Execute callback
    try:
        result = callback(data, **kwargs)
        return result
    except Exception as e:
        raise RuntimeError(f"Callback execution failed: {e}") from e


def do_itm_load(
    item: Any,
    data: Union[IdTbl, TsTbl, WinTbl, pd.DataFrame],
    **kwargs
) -> Union[IdTbl, TsTbl, WinTbl, pd.DataFrame]:
    """Execute item load (R ricu do_itm_load).
    
    Args:
        item: Item object to load
        data: Data to process
        **kwargs: Additional arguments
        
    Returns:
        Loaded data
        
    Examples:
        >>> result = do_itm_load(item_obj, data)
    """
    # Check if item has load method
    if hasattr(item, 'load'):
        return item.load(data, **kwargs)
    
    # Check if item has callback
    if hasattr(item, 'callback'):
        return do_callback(item.callback, data, **kwargs)
    
    # Default: return data as-is
    return data


def set_callback(
    item: Any,
    callback: Callable,
) -> Any:
    """Set callback for item (R ricu set_callback).
    
    Args:
        item: Item object
        callback: Callback function to set
        
    Returns:
        Item with callback set
        
    Examples:
        >>> item = set_callback(item, transform_fun(lambda x: x * 2))
    """
    if not callable(callback):
        raise TypeError(f"Callback must be callable, got {type(callback)}")
    
    # Set callback attribute
    if hasattr(item, 'callback'):
        item.callback = callback
    elif hasattr(item, '_callback'):
        item._callback = callback
    else:
        # Try to add callback attribute
        try:
            item.callback = callback
        except Exception:
            raise AttributeError(f"Cannot set callback on {type(item)}")
    
    return item


def prepare_query(
    item: Any,
    query: Optional[str] = None,
) -> Optional[Callable]:
    """Prepare query for item loading (R ricu prepare_query).
    
    Args:
        item: Item object
        query: Optional query string or expression
        
    Returns:
        Prepared query function or None
        
    Examples:
        >>> query_fun = prepare_query(item, "itemid == 50809")
    """
    if query is None:
        # Try to get query from item
        if hasattr(item, 'query'):
            query = item.query
        elif hasattr(item, 'filter'):
            query = item.filter
        else:
            return None
    
    # Convert query string to function
    if isinstance(query, str):
        # Simple string-based query
        # This is a simplified implementation
        # Full implementation would need to parse and compile the query
        def query_func(df: pd.DataFrame) -> pd.DataFrame:
            # Use pandas eval for simple queries
            try:
                return df.query(query)
            except Exception:
                # Fallback: return as-is
                return df
        
        return query_func
    
    if callable(query):
        return query
    
    return None


def add_weight(
    concept: Any,
    weight: float,
    weight_var: str = "weight",
) -> Any:
    """Add weight to concept (R ricu add_weight).
    
    Args:
        concept: Concept object
        weight: Weight value
        weight_var: Name of weight variable
        
    Returns:
        Concept with weight added
        
    Examples:
        >>> concept = add_weight(concept, 0.5)
    """
    if hasattr(concept, 'weight'):
        concept.weight = weight
    elif hasattr(concept, 'weights'):
        if not hasattr(concept.weights, 'update'):
            concept.weights = {}
        concept.weights[weight_var] = weight
    else:
        # Try to add weight attribute
        try:
            concept.weight = weight
        except Exception:
            raise AttributeError(f"Cannot add weight to {type(concept)}")
    
    return concept


def get_target(
    concept: Any,
) -> Optional[str]:
    """Get target variable from concept (R ricu get_target).
    
    Args:
        concept: Concept object
        
    Returns:
        Target variable name or None
        
    Examples:
        >>> target = get_target(concept)
    """
    if hasattr(concept, 'target'):
        return concept.target
    if hasattr(concept, 'target_var'):
        return concept.target_var
    if hasattr(concept, 'val_var'):
        return concept.val_var
    
    return None


def set_target(
    concept: Any,
    target: str,
) -> Any:
    """Set target variable for concept (R ricu set_target).
    
    Args:
        concept: Concept object
        target: Target variable name
        
    Returns:
        Concept with target set
        
    Examples:
        >>> concept = set_target(concept, 'heart_rate')
    """
    assert_that(is_string(target))
    
    if hasattr(concept, 'target'):
        concept.target = target
    elif hasattr(concept, 'target_var'):
        concept.target_var = target
    elif hasattr(concept, 'val_var'):
        concept.val_var = target
    else:
        # Try to add target attribute
        try:
            concept.target = target
        except Exception:
            raise AttributeError(f"Cannot set target on {type(concept)}")
    
    return concept


def get_itm_var(
    item: Any,
    var_type: str = "val_var",
) -> Optional[str]:
    """Get item variable (R ricu get_itm_var).
    
    Args:
        item: Item object
        var_type: Type of variable ('val_var', 'unit_var', etc.)
        
    Returns:
        Variable name or None
        
    Examples:
        >>> val_var = get_itm_var(item, 'val_var')
    """
    assert_that(is_string(var_type))
    
    if hasattr(item, var_type):
        return getattr(item, var_type)
    
    # Try common variations
    variations = {
        'val_var': ['value_var', 'value', 'val'],
        'unit_var': ['unit', 'uom'],
        'dur_var': ['duration_var', 'duration', 'dur'],
    }
    
    if var_type in variations:
        for var_name in variations[var_type]:
            if hasattr(item, var_name):
                return getattr(item, var_name)
    
    return None

