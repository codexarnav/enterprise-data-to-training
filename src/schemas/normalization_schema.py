"""
Schemas for normalization agent
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class UnifiedSchema(BaseModel):
    """Unified schema for all normalized data"""
    document_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Data source (Slack, Gmail, Drive, etc.)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    content: str = Field(..., description="Main content/text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    lineage: Dict[str, Any] = Field(default_factory=dict, description="Data lineage information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "source": "Slack",
                "timestamp": "2024-01-01T00:00:00Z",
                "content": "Sample content",
                "metadata": {"channel": "general"},
                "lineage": {"original_path": "channel/general", "extraction_method": "slack_api"}
            }
        }


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

