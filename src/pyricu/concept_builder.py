"""Concept and item builders (R ricu concept-utils.R).

Provides functions for creating new concepts and items.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from .concept import ConceptDefinition, ConceptSource
from .assertions import assert_that, is_string

def new_concept(
    name: str,
    sources: Optional[List[ConceptSource]] = None,
    **kwargs
) -> ConceptDefinition:
    """Create new concept (R ricu new_concept).
    
    Args:
        name: Concept name
        sources: List of concept sources
        **kwargs: Additional concept attributes
        
    Returns:
        New ConceptDefinition
        
    Examples:
        >>> concept = new_concept('my_concept', sources=[...])
    """
    assert_that(is_string(name))
    
    if sources is None:
        sources = []
    
    # Create concept definition
    payload = {
        'name': name,
        'sources': [s.to_dict() if hasattr(s, 'to_dict') else s for s in sources],
        **kwargs
    }
    
    return ConceptDefinition.from_name_and_payload(name, payload)

def new_item(
    name: str,
    table: Optional[str] = None,
    column: Optional[str] = None,
    callback: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create new item (R ricu new_item).
    
    Args:
        name: Item name
        table: Table name
        column: Column name
        callback: Optional callback function
        **kwargs: Additional item attributes
        
    Returns:
        Item dictionary
        
    Examples:
        >>> item = new_item('heart_rate', table='vitals', column='hr')
    """
    assert_that(is_string(name))
    
    item_dict = {
        'name': name,
        **kwargs
    }
    
    if table:
        item_dict['table'] = table
    if column:
        item_dict['column'] = column
    if callback:
        item_dict['callback'] = callback
    
    return item_dict

def new_itm(
    item_type: str,
    **kwargs
) -> Dict[str, Any]:
    """Create new item object (R ricu new_itm).
    
    Args:
        item_type: Item type ('col_itm', 'fun_itm', 'hrd_itm', etc.)
        **kwargs: Item attributes
        
    Returns:
        Item dictionary with type
        
    Examples:
        >>> itm = new_itm('col_itm', table='vitals', column='hr')
    """
    assert_that(is_string(item_type))
    
    item_dict = {
        'type': item_type,
        **kwargs
    }
    
    return item_dict

def new_cncpt(
    concept_type: str,
    name: str,
    **kwargs
) -> Dict[str, Any]:
    """Create new concept object (R ricu new_cncpt).
    
    Args:
        concept_type: Concept type ('num_cncpt', 'fct_cncpt', 'rec_cncpt', etc.)
        name: Concept name
        **kwargs: Concept attributes
        
    Returns:
        Concept dictionary with type
        
    Examples:
        >>> cncpt = new_cncpt('num_cncpt', 'heart_rate', ...)
    """
    assert_that(is_string(concept_type), is_string(name))
    
    concept_dict = {
        'type': concept_type,
        'name': name,
        **kwargs
    }
    
    return concept_dict

def init_cncpt(
    concept: Union[str, Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Initialize concept (R ricu init_cncpt).
    
    Args:
        concept: Concept name or dictionary
        **kwargs: Initialization parameters
        
    Returns:
        Initialized concept dictionary
        
    Examples:
        >>> cncpt = init_cncpt('heart_rate', ...)
    """
    if isinstance(concept, str):
        concept_dict = {'name': concept}
    elif isinstance(concept, dict):
        concept_dict = concept.copy()
    else:
        raise TypeError(f"Concept must be string or dict, got {type(concept)}")
    
    # Apply initialization
    concept_dict.update(kwargs)
    
    # Set default type if not specified
    if 'type' not in concept_dict:
        concept_dict['type'] = 'num_cncpt'  # Default: numeric concept
    
    return concept_dict

def init_itm(
    item: Union[str, Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Initialize item (R ricu init_itm).
    
    Args:
        item: Item name or dictionary
        **kwargs: Initialization parameters
        
    Returns:
        Initialized item dictionary
        
    Examples:
        >>> itm = init_itm('heart_rate_item', table='vitals', column='hr')
    """
    if isinstance(item, str):
        item_dict = {'name': item}
    elif isinstance(item, dict):
        item_dict = item.copy()
    else:
        raise TypeError(f"Item must be string or dict, got {type(item)}")
    
    # Apply initialization
    item_dict.update(kwargs)
    
    # Set default type if not specified
    if 'type' not in item_dict:
        item_dict['type'] = 'col_itm'  # Default: column item
    
    return item_dict

def is_cncpt(x: Any) -> bool:
    """Check if object is a concept (R ricu is_cncpt).
    
    Args:
        x: Object to check
        
    Returns:
        True if object is a concept, False otherwise
    """
    if isinstance(x, ConceptDefinition):
        return True
    if isinstance(x, dict):
        return 'type' in x and 'cncpt' in x.get('type', '')
    return False

def is_concept(x: Any) -> bool:
    """Check if object is a concept object (R ricu is_concept).
    
    Args:
        x: Object to check
        
    Returns:
        True if object is a concept, False otherwise
    """
    return isinstance(x, ConceptDefinition) or is_cncpt(x)

def is_item(x: Any) -> bool:
    """Check if object is an item (R ricu is_item).
    
    Args:
        x: Object to check
        
    Returns:
        True if object is an item, False otherwise
    """
    if isinstance(x, dict):
        return 'type' in x and 'itm' in x.get('type', '')
    if hasattr(x, 'item_type'):
        return True
    return False

def is_itm(x: Any) -> bool:
    """Check if object is an item object (R ricu is_itm).
    
    Args:
        x: Object to check
        
    Returns:
        True if object is an item, False otherwise
    """
    return is_item(x)

def new_src_tbl(
    table_name: str,
    src: Union[str, Any],
) -> Any:
    """Create new source table object (R ricu new_src_tbl).
    
    Args:
        table_name: Table name
        src: Data source identifier
        
    Returns:
        Source table representation
        
    Examples:
        >>> tbl = new_src_tbl('patients', 'mimic_demo')
    """
    assert_that(is_string(table_name))
    
    from .table_convert import as_src_tbl
    return as_src_tbl(table_name, src)

