"""Concept management utilities (R ricu concept-utils.R).

Provides functions for managing concepts, checking availability, and explaining dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .concept import ConceptDictionary, ConceptDefinition
from .resources import load_dictionary


def add_concept(
    dict_name: str,
    concept_name: str,
    definition: Dict[str, Any],
    dict_dirs: Optional[List[Union[str, Path]]] = None,
) -> None:
    """Add a concept to dictionary (R ricu add_concept).
    
    Args:
        dict_name: Dictionary name
        concept_name: Name of concept to add
        definition: Concept definition dictionary
        dict_dirs: Optional list of dictionary directories
        
    Note:
        This is a simplified implementation. In practice, you would
        modify the JSON file or use a writable dictionary.
    """
    # Load dictionary
    dictionary = load_dictionary(dict_name, dict_dirs)
    
    # Create concept definition
    concept_def = ConceptDefinition.from_name_and_payload(concept_name, definition)
    
    # Add to dictionary
    dictionary._concepts[concept_name] = concept_def
    
    # Note: In a real implementation, you would save the dictionary back to file
    # For now, this is an in-memory operation


def concept_availability(
    concepts: Union[str, List[str]],
    src: Optional[str] = None,
    dict_name: Optional[str] = None,
) -> Dict[str, bool]:
    """Check concept availability (R ricu concept_availability).
    
    Args:
        concepts: Concept name(s)
        src: Optional data source name
        dict_name: Optional dictionary name
        
    Returns:
        Dictionary mapping concept names to availability
        
    Examples:
        >>> concept_availability(['hr', 'sbp'], src='mimic_demo')
        {'hr': True, 'sbp': True}
    """
    if isinstance(concepts, str):
        concepts = [concepts]
    
    # Load dictionary
    dictionary = load_dictionary(dict_name)
    
    result = {}
    for concept_name in concepts:
        if concept_name in dictionary:
            if src:
                # Check if concept is available for this source
                concept_def = dictionary[concept_name]
                available = any(
                    src_name in source.name for source in concept_def.sources
                    if hasattr(source, 'name')
                )
                result[concept_name] = available
            else:
                result[concept_name] = True
        else:
            result[concept_name] = False
    
    return result


def explain_dictionary(
    dict_name: Optional[str] = None,
    dict_dirs: Optional[List[Union[str, Path]]] = None,
) -> str:
    """Explain dictionary structure (R ricu explain_dictionary).
    
    Args:
        dict_name: Dictionary name
        dict_dirs: Optional list of dictionary directories
        
    Returns:
        String explanation of dictionary
        
    Examples:
        >>> print(explain_dictionary())
    """
    dictionary = load_dictionary(dict_name, dict_dirs)
    
    lines = []
    lines.append(f"Concept Dictionary: {dict_name or 'default'}")
    lines.append(f"Total concepts: {len(dictionary)}")
    lines.append("")
    
    # Group by concept type
    concept_types = {}
    for name, concept_def in dictionary.items():
        concept_type = type(concept_def).__name__
        if concept_type not in concept_types:
            concept_types[concept_type] = []
        concept_types[concept_type].append(name)
    
    for concept_type, names in concept_types.items():
        lines.append(f"{concept_type}: {len(names)} concepts")
        for name in sorted(names)[:10]:  # Show first 10
            lines.append(f"  - {name}")
        if len(names) > 10:
            lines.append(f"  ... and {len(names) - 10} more")
        lines.append("")
    
    return "\n".join(lines)


def subset_src(
    concept: Union[str, ConceptDefinition],
    src: str,
) -> Optional[ConceptDefinition]:
    """Subset concept for specific source (R ricu subset_src).
    
    Args:
        concept: Concept name or definition
        src: Data source name
        
    Returns:
        Concept definition subset for source, or None if not available
        
    Examples:
        >>> subset_src('hr', 'mimic_demo')
    """
    if isinstance(concept, str):
        dictionary = load_dictionary()
        if concept not in dictionary:
            return None
        concept = dictionary[concept]
    
    # Find source-specific definition
    for source in concept.sources:
        if hasattr(source, 'name') and source.name == src:
            # Create a new concept definition with only this source
            return ConceptDefinition(
                name=concept.name,
                sources=[source]
            )
    
    return None

