"""
Constants and configuration for the ETL pipeline
"""

# Data paths
DATA_DIR = 'data'
NORMALIZED_DATA_FILE = 'data/normalized_data.jsonl'
CLEANED_DATA_FILE = 'data/cleaned_data.jsonl'

# Processing constants
BATCH_SIZE = 50
MAX_CONTENT_LENGTH = 5000
SEGMENTATION_THRESHOLD = 2000
MIN_CONTENT_LENGTH = 10

# LLM Configuration
LLM_MODEL = 'gemini-2.0-flash'
LLM_TEMPERATURE = 0.1

# Valid source types
VALID_SOURCES = ['Slack', 'Gmail', 'Google Drive', 'GitHub', 'Jira', 'Zendesk', 'Other']

# Image/PDF file types
IMAGE_FILE_TYPES = ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif', 'pdf']

