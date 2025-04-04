"""
Client for interacting with the Google Gemini API.
"""

import os
import logging
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from PIL import Image

from src.config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS

# Configure logging
logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for interacting with Google's Gemini model for image classification.
    """
    
    def __init__(self):
        """
        Initialize the Gemini client with the API key from config.
        """
        self.api_key = GOOGLE_API_KEY
        self.model_name = GEMINI_MODEL
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get the specified model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )
            logger.info(f"Successfully initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def _prepare_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Prepare an image for submission to the Gemini API.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with image data in the format expected by Gemini
        """
        try:
            # Open the image using PIL
            img = Image.open(BytesIO(image_data))
            
            # Create a BytesIO object to store the image data
            buffered = BytesIO()
            
            # Save the image as JPEG to the BytesIO object
            img.save(buffered, format="JPEG")
            
            # Get the byte data and encode it as base64
            img_bytes = buffered.getvalue()
            
            # Create the image part for the Gemini API
            image_part = {
                "mime_type": "image/jpeg",
                "data": img_bytes
            }
            
            return image_part
        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            raise
    
    async def classify_image(self, image_data: bytes, categories: List[str]) -> Dict[str, float]:
        """
        Classify an image against a list of categories using Gemini.
        
        Args:
            image_data: Raw image bytes
            categories: List of category labels to classify against
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        try:
            # Prepare the image for the API
            image_part = self._prepare_image(image_data)
            
            # Create the prompt for classification
            categories_text = ", ".join(categories)
            prompt = f"""
            Please analyze this image and classify it into one of the following categories:
            {categories_text}
            
            For each category, provide a confidence score between 0 and 1, where 1 is the highest confidence.
            Format your response as a JSON object with categories as keys and confidence scores as values.
            Only include these categories in your response: {categories_text}
            """
            
            # Generate content using the model
            response = await self.model.generate_content_async(
                [image_part, prompt]
            )
            
            # Extract the text response
            text_response = response.text
            
            # Parse the response to extract categories and scores
            # This is a simplified version; in a real app, you would use proper JSON parsing
            import json
            try:
                # Attempt to extract JSON from the response
                # Sometimes the model might wrap JSON in markdown code blocks or other text
                if "```json" in text_response:
                    json_str = text_response.split("```json")[1].split("```")[0].strip()
                elif "```" in text_response:
                    json_str = text_response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = text_response.strip()
                
                # Parse the JSON string
                classification_results = json.loads(json_str)
                
                # Ensure all categories are present with default score of 0
                for category in categories:
                    if category not in classification_results:
                        classification_results[category] = 0.0
                
                return classification_results
            except json.JSONDecodeError:
                logger.error(f"Failed to parse model response as JSON: {text_response}")
                # Return a default result with all categories set to 0
                return {category: 0.0 for category in categories}
                
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            # Return a default result with all categories set to 0
            return {category: 0.0 for category in categories}


