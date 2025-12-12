
import json
import logging
import hashlib
from datetime import datetime
from typing import TypedDict, Optional, Any, Dict, List
from pydantic import BaseModel, Field, ValidationError
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

# Unified Schema Definition
class UnifiedSchema(BaseModel):
    """Unified schema for all normalized data"""
    document_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Data source (Slack, Gmail, Drive, etc.)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    content: str = Field(..., description="Main content/text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    lineage: Dict[str, Any] = Field(default_factory=dict, description="Data lineage information")


class NormalizationInput(BaseModel):
    """Input state for normalization"""
    raw_data: str = Field(..., description="Raw data from ingestion")
    file_type: Optional[str] = Field(None, description="File type if applicable")
    source_type: Optional[str] = Field(None, description="Pre-detected source type")
    raw_input: Optional[Dict[str, Any]] = Field(None, description="Parsed raw input")
    dynamic_prompt: Optional[str] = Field(None, description="Generated prompt for normalization")
    normalized_json: Optional[Dict[str, Any]] = Field(None, description="Normalized JSON output")
    validation_errors: List[str] = Field(default_factory=list, description="Schema validation errors")


class NormalizationOutput(BaseModel):
    """Output state for normalization"""
    normalized_data: Optional[Dict[str, Any]] = None
    validation_passed: bool = False


class NormalizationState(TypedDict):
    """State for normalization graph"""
    input: NormalizationInput
    output: NormalizationOutput


def load_raw_data(state: NormalizationState) -> NormalizationState:
    """Load and parse raw data from ingestion"""
    try:
        raw_data = state['input'].raw_data
        # Try to parse as JSON, otherwise treat as plain text
        try:
            parsed_data = json.loads(raw_data)
        except json.JSONDecodeError:
            parsed_data = {"text": raw_data}
        
        state['input'].raw_input = parsed_data
        logger.info(f"Loaded raw data, type: {type(parsed_data)}")
        return state
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        state['input'].raw_input = {"text": str(raw_data), "error": str(e)}
        return state


def router(state: NormalizationState) -> str:
    """Route to OCR if image/PDF, otherwise to source detection"""
    file_type = state['input'].file_type
    if file_type and file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif', 'pdf']:
        return 'use_vision_ocr'
    return 'source_detector'


def use_vision_ocr(state: NormalizationState) -> NormalizationState:
    """Extract text from images/PDFs using Vision LLM (GPT-4o or Gemini Vision)"""
    try:
        file_type = state['input'].file_type
        raw_data = state['input'].raw_data
        
        # For MVP: Using Gemini 2.0 Flash which supports vision
        prompt = PromptTemplate(
            template="""You are a vision OCR agent. Extract all text content from this {file_type} file.
            Preserve the structure, layout, and any tables if present.
            Return only the extracted text content, no explanations."""
        )
        
        messages = [
            SystemMessage(content=prompt.format(file_type=file_type)),
            HumanMessage(content=raw_data)  # In production, this would be image bytes/base64
        ]
        
        response = llm.invoke(messages)
        ocr_text = response.content.strip()
        
        state['input'].raw_input = {
            "text": ocr_text,
            "extraction_method": "vision_ocr",
            "file_type": file_type
        }
        logger.info(f"OCR completed for {file_type}, extracted {len(ocr_text)} characters")
        return state
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        state['input'].raw_input = {"text": "", "error": f"OCR failed: {str(e)}"}
        return state


def source_detector(state: NormalizationState) -> NormalizationState:
    """LLM #1: Detect data source from raw input"""
    # Skip if already detected
    if state['input'].source_type:
        logger.info(f"Source already detected: {state['input'].source_type}")
        return state
    
    try:
        raw_input = state['input'].raw_input or {}
        raw_text = raw_input.get('text', '') or str(raw_input)
        
        prompt = PromptTemplate(
            template="""You are an expert data source detector. Analyze the following data and identify its source.
            
            Possible sources: Slack, Gmail, Google Drive, GitHub, Jira, Zendesk, Other
            
            Respond with ONLY the source name (e.g., "Slack", "Gmail", etc.). If uncertain, respond with "Other".
            
            Data to analyze:
            {raw_data}"""
        )
        
        messages = [
            SystemMessage(content=prompt.format(raw_data=raw_text[:2000])),  # Limit input size
            HumanMessage(content="Identify the source of this data.")
        ]
        
        response = llm.invoke(messages)
        detected_source = response.content.strip()
        
        # Validate and normalize source name
        valid_sources = ['Slack', 'Gmail', 'Google Drive', 'GitHub', 'Jira', 'Zendesk', 'Other']
        if detected_source not in valid_sources:
            detected_source = 'Other'
        
        state['input'].source_type = detected_source
        logger.info(f"Source detected: {detected_source}")
        return state
    except Exception as e:
        logger.error(f"Source detection error: {str(e)}")
        state['input'].source_type = 'Other'
        return state


def prompt_builder(state: NormalizationState) -> NormalizationState:
    """Build dynamic prompt based on detected source type"""
    try:
        source_type = state['input'].source_type or 'Other'
        raw_input = state['input'].raw_input or {}
        
        # Source-specific prompt templates
        source_prompts = {
            'Slack': """Extract from this Slack message:
- document_id: unique message ID
- source: "Slack"
- timestamp: ISO 8601 format
- content: message text
- metadata: {{"channel": channel_name, "user": username, "thread_ts": thread_timestamp if exists}}
- lineage: {{"original_path": channel_name, "extraction_method": "slack_api"}}""",
            
            'Gmail': """Extract from this email:
- document_id: unique email ID
- source: "Gmail"
- timestamp: ISO 8601 format
- content: email body text
- metadata: {{"subject": subject, "from": sender, "to": recipients, "cc": cc_recipients}}
- lineage: {{"original_path": email_id, "extraction_method": "imap"}}""",
            
            'Google Drive': """Extract from this Google Drive document:
- document_id: unique file ID
- source: "Google Drive"
- timestamp: ISO 8601 format
- content: document text content
- metadata: {{"file_name": name, "mime_type": mime_type, "author": owner}}
- lineage: {{"original_path": file_path, "extraction_method": "drive_api"}}""",
            
            'Other': """Extract from this document:
- document_id: generate unique ID
- source: detected source name
- timestamp: ISO 8601 format (extract or use current time)
- content: main text content
- metadata: {{"word_count": approximate, "file_type": file_type if available}}
- lineage: {{"original_path": "unknown", "extraction_method": "unknown"}}"""
        }
        
        prompt_template = source_prompts.get(source_type, source_prompts['Other'])
        
        dynamic_prompt = f"""You are a normalization agent. {prompt_template}

Return ONLY valid JSON matching this exact structure:
{{
    "document_id": "string",
    "source": "string",
    "timestamp": "ISO 8601 string",
    "content": "string",
    "metadata": {{}},
    "lineage": {{}}
}}

Input data:
{json.dumps(raw_input, indent=2)[:3000]}"""
        
        state['input'].dynamic_prompt = dynamic_prompt
        logger.info(f"Built dynamic prompt for source: {source_type}")
        return state
    except Exception as e:
        logger.error(f"Prompt builder error: {str(e)}")
        return state


def schema_normalization(state: NormalizationState) -> NormalizationState:
    """LLM #2: Normalize data to unified schema"""
    try:
        dynamic_prompt = state['input'].dynamic_prompt
        if not dynamic_prompt:
            logger.error("No dynamic prompt available")
            return state
        
        messages = [
            SystemMessage(content=dynamic_prompt),
            HumanMessage(content="Normalize this data to the unified schema. Return only valid JSON.")
        ]
        
        response = llm.invoke(messages)
        normalized_text = response.content.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if normalized_text.startswith('```'):
            normalized_text = normalized_text.split('```')[1]
            if normalized_text.startswith('json'):
                normalized_text = normalized_text[4:]
        normalized_text = normalized_text.strip()
        
        try:
            normalized_json = json.loads(normalized_text)
            state['input'].normalized_json = normalized_json
            logger.info("Schema normalization completed")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            # Fallback: create minimal structure
            state['input'].normalized_json = {
                "document_id": hashlib.md5(normalized_text.encode()).hexdigest(),
                "source": state['input'].source_type or "Unknown",
                "timestamp": datetime.utcnow().isoformat(),
                "content": normalized_text[:1000],
                "metadata": {},
                "lineage": {}
            }
        return state
    except Exception as e:
        logger.error(f"Schema normalization error: {str(e)}")
        return state


def schema_validator(state: NormalizationState) -> NormalizationState:
    """Validate normalized JSON against unified schema"""
    try:
        normalized_json = state['input'].normalized_json
        if not normalized_json:
            state['input'].validation_errors.append("No normalized JSON to validate")
            state['output'].validation_passed = False
            return state
        
        # Validate using Pydantic
        try:
            validated_data = UnifiedSchema(**normalized_json)
            state['input'].normalized_json = validated_data.model_dump()
            state['output'].normalized_data = validated_data.model_dump()
            state['output'].validation_passed = True
            logger.info("Schema validation passed")
        except ValidationError as e:
            errors = [str(err) for err in e.errors()]
            state['input'].validation_errors = errors
            state['output'].validation_passed = False
            logger.warning(f"Schema validation failed: {errors}")
            
            # Try to fix common issues
            fixed_data = normalized_json.copy()
            if not fixed_data.get('document_id'):
                fixed_data['document_id'] = hashlib.md5(str(normalized_json).encode()).hexdigest()
            if not fixed_data.get('timestamp'):
                fixed_data['timestamp'] = datetime.utcnow().isoformat()
            if not fixed_data.get('metadata'):
                fixed_data['metadata'] = {}
            if not fixed_data.get('lineage'):
                fixed_data['lineage'] = {}
            
            try:
                validated_data = UnifiedSchema(**fixed_data)
                state['input'].normalized_json = validated_data.model_dump()
                state['output'].normalized_data = validated_data.model_dump()
                state['output'].validation_passed = True
                logger.info("Schema validation passed after fixes")
            except:
                pass
        
        return state
    except Exception as e:
        logger.error(f"Schema validation error: {str(e)}")
        state['output'].validation_passed = False
        return state


def unified_data_lake_writer(state: NormalizationState) -> NormalizationState:
    """Write validated data to unified data lake"""
    try:
        if not state['output'].validation_passed:
            logger.warning("Skipping data lake write - validation failed")
            return state
        
        normalized_data = state['output'].normalized_data
        if not normalized_data:
            logger.warning("No validated data to write")
            return state
        
        # For MVP: Write to local JSONL file
        # In production: Write to GCS/S3 with proper partitioning
        output_file = 'data/normalized_data.jsonl'
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(normalized_data, ensure_ascii=False) + '\n')
        
        logger.info(f"Written to unified data lake: {normalized_data.get('document_id')}")
        return state
    except Exception as e:
        logger.error(f"Data lake write error: {str(e)}")
        return state


# Build LangGraph
memory = InMemorySaver()
graph = StateGraph(NormalizationState)

# Add nodes
graph.add_node("load_raw_data", load_raw_data)
graph.add_node("use_vision_ocr", use_vision_ocr)
graph.add_node("source_detector", source_detector)
graph.add_node("prompt_builder", prompt_builder)
graph.add_node("schema_normalization", schema_normalization)
graph.add_node("schema_validator", schema_validator)
graph.add_node("unified_data_lake_writer", unified_data_lake_writer)

# Add edges
graph.add_edge(START, "load_raw_data")
graph.add_conditional_edges(
    "load_raw_data",
    router,
    {
        "use_vision_ocr": "use_vision_ocr",
        "source_detector": "source_detector"
    }
)
graph.add_edge("use_vision_ocr", "source_detector")
graph.add_edge("source_detector", "prompt_builder")
graph.add_edge("prompt_builder", "schema_normalization")
graph.add_edge("schema_normalization", "schema_validator")
graph.add_edge("schema_validator", "unified_data_lake_writer")
graph.add_edge("unified_data_lake_writer", END)

# Compile graph
normalization_node = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
