"""
Schemas for cleaning agent
"""
from typing import TypedDict, List, Dict, Any


class CleaningState(TypedDict):
    """State for cleaning graph"""
    input_data: List[Dict[str, Any]]  # Data from unified data lake
    deduplicated_data: List[Dict[str, Any]]  # After duplicate detection
    filtered_data: List[Dict[str, Any]]  # After noise filtering
    segmented_data: List[Dict[str, Any]]  # After segmentation
    entities_extracted: List[Dict[str, Any]]  # After entity extraction
    relationships_mapped: List[Dict[str, Any]]  # After relationship mapping
    clean_data: List[Dict[str, Any]]  # Final clean output
    processing_stats: Dict[str, Any]  # Statistics about processing

