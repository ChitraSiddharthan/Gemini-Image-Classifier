#!/usr/bin/env python
"""
Main entry point for the Gemini Image Classifier application.
This script initializes and launches the Gradio interface.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Check if API key is available
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("No Google API key found. Please set the GOOGLE_API_KEY environment variable.")
    print("Error: No Google API key found. Please set the GOOGLE_API_KEY environment variable.")
    exit(1)

# Import the Gradio interface
from src.ui.gradio_interface import create_interface

def main():
    """Initialize and launch the application."""
    logger.info("Starting Gemini Image Classifier")
    
    # Create and launch the Gradio interface
    demo = create_interface()
    
    # Start the interface
    demo.launch(share=False, inbrowser=True)
    
    logger.info("Gemini Image Classifier has been shut down")

if __name__ == "__main__":
    main()


