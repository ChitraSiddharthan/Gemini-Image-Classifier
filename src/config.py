"""
Configuration settings for the Gemini Image Classifier application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"  # Model name for Gemini 2.5 Flash

# Classification Configuration
DEFAULT_CATEGORIES = [
    "landscape",
    "portrait",
    "food",
    "animal",
    "architecture",
    "art",
    "vehicle",
    "sports",
    "technology",
    "other"
]

# Temperature affects the randomness of the model's output
# Lower values make output more deterministic
TEMPERATURE = 0.2

# Max output tokens for the model
MAX_OUTPUT_TOKENS = 1024

# UI Configuration
UI_TITLE = "Gemini Image Classifier"
UI_DESCRIPTION = """
# Gemini Image Classifier
Upload images to classify them using Google's Gemini 2.5 Flash model.
You can use predefined categories or create your own custom categories.
"""
UI_THEME = "default"  # Possible values: "default", "huggingface", "grass", "peach"
UI_PORT = 7860

# File Handling
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB

# Logging Configuration
LOG_LEVEL = "INFO"


