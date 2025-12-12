"""
Custom exceptions for pipeline operations
"""


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass


class NormalizationError(PipelineError):
    """Exception raised during normalization"""
    pass


class CleaningError(PipelineError):
    """Exception raised during data cleaning"""
    pass


class ValidationError(PipelineError):
    """Exception raised during schema validation"""
    pass


class DataLakeError(PipelineError):
    """Exception raised during data lake operations"""
    pass


class LLMError(PipelineError):
    """Exception raised during LLM operations"""
    pass

