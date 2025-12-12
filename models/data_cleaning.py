
import json
import logging
import hashlib
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model_name='gemini-2.0-flash', temperature=0.1)

# Batch size for processing
BATCH_SIZE = 50


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


def load_from_data_lake(state: CleaningState) -> CleaningState:
    """Load data from unified data lake (normalized_data.jsonl)"""
    try:
        input_file = 'data/normalized_data.jsonl'
        
        data = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line: {str(e)}")
        except FileNotFoundError:
            logger.warning(f"Input file not found: {input_file}, using empty dataset")
        
        state['input_data'] = data
        state['processing_stats'] = {
            'total_documents': len(data),
            'duplicates_removed': 0,
            'noise_filtered': 0,
            'segments_created': 0
        }
        logger.info(f"Loaded {len(data)} documents from data lake")
        return state
    except Exception as e:
        logger.error(f"Error loading from data lake: {str(e)}")
        state['input_data'] = []
        state['processing_stats'] = {}
        return state


def duplicate_detection(state: CleaningState) -> CleaningState:
    """Detect and remove duplicate documents"""
    try:
        input_data = state.get('input_data', [])
        if not input_data:
            state['deduplicated_data'] = []
            return state
        
        # Create content hashes for duplicate detection
        seen_hashes = set()
        deduplicated = []
        duplicates_removed = 0
        
        for doc in input_data:
            # Create hash from content + source + timestamp
            content = doc.get('content', '')
            source = doc.get('source', '')
            timestamp = doc.get('timestamp', '')
            content_hash = hashlib.md5(f"{content}{source}{timestamp}".encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(doc)
            else:
                duplicates_removed += 1
        
        # Also use LLM for semantic duplicate detection on small batches
        if len(deduplicated) <= 100:
            try:
                prompt = PromptTemplate(
                    template="""You are a duplicate detection agent. Analyze these documents and identify semantic duplicates.
                    
                    Return a JSON array with indices of documents that are duplicates (keep the first occurrence).
                    Example: [2, 5, 8] means documents at indices 2, 5, 8 are duplicates.
                    
                    Documents:
                    {documents}
                    
                    Return only JSON array of duplicate indices, or empty array [] if no duplicates."""
                )
                
                docs_summary = [f"{i}: {doc.get('content', '')[:200]}" for i, doc in enumerate(deduplicated[:50])]
                messages = [
                    SystemMessage(content=prompt.format(documents='\n'.join(docs_summary))),
                    HumanMessage(content="Identify semantic duplicates.")
                ]
                
                response = llm.invoke(messages)
                duplicate_indices = json.loads(response.content.strip())
                
                # Remove duplicates (in reverse order to maintain indices)
                for idx in sorted(duplicate_indices, reverse=True):
                    if 0 <= idx < len(deduplicated):
                        deduplicated.pop(idx)
                        duplicates_removed += 1
            except Exception as e:
                logger.warning(f"LLM duplicate detection failed: {str(e)}")
        
        state['deduplicated_data'] = deduplicated
        state['processing_stats']['duplicates_removed'] = duplicates_removed
        logger.info(f"Duplicate detection: {len(deduplicated)} unique documents, {duplicates_removed} removed")
        return state
    except Exception as e:
        logger.error(f"Duplicate detection error: {str(e)}")
        state['deduplicated_data'] = state.get('input_data', [])
        return state


def noise_filtering(state: CleaningState) -> CleaningState:
    """Filter low-quality text and noise"""
    try:
        deduplicated_data = state.get('deduplicated_data', [])
        if not deduplicated_data:
            state['filtered_data'] = []
            return state
        
        filtered = []
        noise_removed = 0
        
        for doc in deduplicated_data:
            content = doc.get('content', '')
            
            # Basic filtering rules
            if not content or len(content.strip()) < 10:
                noise_removed += 1
                continue
            
            # Check for excessive whitespace or special characters
            if len(content) > 0 and (content.count(' ') / len(content)) > 0.5:
                noise_removed += 1
                continue
            
            # Use LLM for quality assessment on batches
            if len(filtered) < BATCH_SIZE:
                try:
                    prompt = PromptTemplate(
                        template="""You are a data quality filter. Assess if this text is high-quality and meaningful.
                        
                        Text: {content}
                        
                        Respond with "KEEP" if the text is meaningful and useful, or "REMOVE" if it's noise/low-quality.
                        Only respond with KEEP or REMOVE."""
                    )
                    
                    messages = [
                        SystemMessage(content=prompt.format(content=content[:1000])),
                        HumanMessage(content="Should this text be kept?")
                    ]
                    
                    response = llm.invoke(messages)
                    decision = response.content.strip().upper()
                    
                    if decision == "REMOVE":
                        noise_removed += 1
                        continue
                except Exception as e:
                    logger.warning(f"LLM noise filtering failed for doc: {str(e)}")
            
            filtered.append(doc)
        
        state['filtered_data'] = filtered
        state['processing_stats']['noise_filtered'] = noise_removed
        logger.info(f"Noise filtering: {len(filtered)} documents kept, {noise_removed} removed")
        return state
    except Exception as e:
        logger.error(f"Noise filtering error: {str(e)}")
        state['filtered_data'] = state.get('deduplicated_data', [])
        return state


def segmentation(state: CleaningState) -> CleaningState:
    """Segment content logically into smaller chunks"""
    try:
        filtered_data = state.get('filtered_data', [])
        if not filtered_data:
            state['segmented_data'] = []
            return state
        
        segmented = []
        total_segments = 0
        
        for doc in filtered_data:
            content = doc.get('content', '')
            if not content:
                continue
            
            # Simple segmentation: split by paragraphs or sentences for long content
            if len(content) > 2000:
                # Use LLM to segment intelligently
                try:
                    prompt = PromptTemplate(
                        template="""You are a content segmentation agent. Split this text into logical segments.
                        
                        Each segment should be:
                        - 200-500 words
                        - Semantically complete
                        - Preserve context
                        
                        Text: {content}
                        
                        Return JSON array of segments: ["segment1", "segment2", ...]"""
                    )
                    
                    messages = [
                        SystemMessage(content=prompt.format(content=content[:5000])),
                        HumanMessage(content="Segment this text into logical chunks.")
                    ]
                    
                    response = llm.invoke(messages)
                    segments = json.loads(response.content.strip())
                    
                    for i, segment in enumerate(segments):
                        segment_doc = doc.copy()
                        segment_doc['content'] = segment
                        segment_doc['segment_id'] = f"{doc.get('document_id', 'doc')}_seg_{i}"
                        segment_doc['metadata']['segment_index'] = i
                        segment_doc['metadata']['total_segments'] = len(segments)
                        segmented.append(segment_doc)
                        total_segments += 1
                except Exception as e:
                    logger.warning(f"LLM segmentation failed: {str(e)}")
                    # Fallback: keep original
                    segmented.append(doc)
                    total_segments += 1
            else:
                # Short content: keep as-is
                segmented.append(doc)
                total_segments += 1
        
        state['segmented_data'] = segmented
        state['processing_stats']['segments_created'] = total_segments
        logger.info(f"Segmentation: {total_segments} segments created from {len(filtered_data)} documents")
        return state
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        state['segmented_data'] = state.get('filtered_data', [])
        return state


def entity_extraction(state: CleaningState) -> CleaningState:
    """Extract entities from content"""
    try:
        segmented_data = state.get('segmented_data', [])
        if not segmented_data:
            state['entities_extracted'] = []
            return state
        
        entities_extracted = []
        
        # Process in batches
        for i in range(0, len(segmented_data), BATCH_SIZE):
            batch = segmented_data[i:i + BATCH_SIZE]
            
            try:
                prompt = PromptTemplate(
                    template="""You are an entity extraction agent. Extract key entities from these documents.
                    
                    For each document, extract:
                    - People (names)
                    - Organizations
                    - Locations
                    - Dates
                    - Key topics/domains
                    
                    Return JSON array where each item has:
                    {{
                        "document_id": "id",
                        "entities": {{
                            "people": [],
                            "organizations": [],
                            "locations": [],
                            "dates": [],
                            "topics": []
                        }}
                    }}
                    
                    Documents: {documents}"""
                )
                
                docs_text = [f"ID: {doc.get('document_id')}, Content: {doc.get('content', '')[:500]}" 
                            for doc in batch]
                
                messages = [
                    SystemMessage(content=prompt.format(documents='\n\n'.join(docs_text))),
                    HumanMessage(content="Extract entities from these documents.")
                ]
                
                response = llm.invoke(messages)
                entities_data = json.loads(response.content.strip())
                
                # Merge entities back into documents
                for doc in batch:
                    doc_id = doc.get('document_id')
                    doc_entities = next((e for e in entities_data if e.get('document_id') == doc_id), {})
                    doc['metadata']['entities'] = doc_entities.get('entities', {})
                    entities_extracted.append(doc)
            except Exception as e:
                logger.warning(f"Entity extraction failed for batch {i}: {str(e)}")
                # Fallback: add empty entities
                for doc in batch:
                    doc['metadata']['entities'] = {}
                    entities_extracted.append(doc)
        
        state['entities_extracted'] = entities_extracted
        logger.info(f"Entity extraction: processed {len(entities_extracted)} documents")
        return state
    except Exception as e:
        logger.error(f"Entity extraction error: {str(e)}")
        state['entities_extracted'] = state.get('segmented_data', [])
        return state


def relationship_mapping(state: CleaningState) -> CleaningState:
    """Map relationships between entities and documents"""
    try:
        entities_extracted = state.get('entities_extracted', [])
        if not entities_extracted:
            state['relationships_mapped'] = []
            return state
        
        relationships_mapped = []
        
        # For MVP: Simple relationship mapping
        # In production: More sophisticated graph-based relationships
        for doc in entities_extracted:
            entities = doc.get('metadata', {}).get('entities', {})
            
            # Create simple relationships
            relationships = {
                'document_entities': entities,
                'related_documents': []  # Would be populated with document IDs in production
            }
            
            doc['metadata']['relationships'] = relationships
            relationships_mapped.append(doc)
        
        state['relationships_mapped'] = relationships_mapped
        logger.info(f"Relationship mapping: processed {len(relationships_mapped)} documents")
        return state
    except Exception as e:
        logger.error(f"Relationship mapping error: {str(e)}")
        state['relationships_mapped'] = state.get('entities_extracted', [])
        return state


def finalize_clean_data(state: CleaningState) -> CleaningState:
    """Finalize and save clean data"""
    try:
        relationships_mapped = state.get('relationships_mapped', [])
        
        # Add processing metadata
        for doc in relationships_mapped:
            doc['metadata']['processing'] = {
                'stage': 'cleaned',
                'cleaning_stats': state.get('processing_stats', {})
            }
        
        state['clean_data'] = relationships_mapped
        
        # Save to file
        output_file = 'data/cleaned_data.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in relationships_mapped:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"Clean data finalized: {len(relationships_mapped)} documents saved to {output_file}")
        return state
    except Exception as e:
        logger.error(f"Finalize clean data error: {str(e)}")
        return state


# Build LangGraph
memory = InMemorySaver()
graph = StateGraph(CleaningState)

# Add nodes
graph.add_node("load_from_data_lake", load_from_data_lake)
graph.add_node("duplicate_detection", duplicate_detection)
graph.add_node("noise_filtering", noise_filtering)
graph.add_node("segmentation", segmentation)
graph.add_node("entity_extraction", entity_extraction)
graph.add_node("relationship_mapping", relationship_mapping)
graph.add_node("finalize_clean_data", finalize_clean_data)

# Add edges (sequential flow)
graph.add_edge(START, "load_from_data_lake")
graph.add_edge("load_from_data_lake", "duplicate_detection")
graph.add_edge("duplicate_detection", "noise_filtering")
graph.add_edge("noise_filtering", "segmentation")
graph.add_edge("segmentation", "entity_extraction")
graph.add_edge("entity_extraction", "relationship_mapping")
graph.add_edge("relationship_mapping", "finalize_clean_data")
graph.add_edge("finalize_clean_data", END)

# Compile graph
cleaning_node = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
