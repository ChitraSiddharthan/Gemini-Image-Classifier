"""
Image classification logic using the Gemini model.
"""

import asyncio
import logging
from typing import List, Dict, Any, Union, Tuple
from PIL import Image
import io

from src.classifier.gemini_client import GeminiClient
from src.config import DEFAULT_CATEGORIES

# Configure logging
logger = logging.getLogger(__name__)

class ImageClassifier:
    """
    Handles image classification using the Gemini API.
    """
    
    def __init__(self):
        """
        Initialize the image classifier with a Gemini client.
        """
        self.client = GeminiClient()
        self.default_categories = DEFAULT_CATEGORIES
        logger.info("ImageClassifier initialized with default categories")
    
    def _validate_categories(self, categories: List[str]) -> List[str]:
        """
        Validate and clean up category names.
        
        Args:
            categories: List of category names
            
        Returns:
            Cleaned list of category names
        """
        # Filter out empty categories
        valid_categories = [cat.strip() for cat in categories if cat.strip()]
        
        # If no valid categories provided, use defaults
        if not valid_categories:
            return self.default_categories
        
        return valid_categories
    
    def _prepare_image(self, image: Union[str, bytes, Image.Image]) -> bytes:
        """
        Prepare an image for classification.
        
        Args:
            image: Image to classify, can be a file path, bytes, or PIL Image
            
        Returns:
            Image bytes ready for processing
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # File path
                with open(image, 'rb') as img_file:
                    return img_file.read()
            elif isinstance(image, bytes):
                # Raw bytes
                return image
            elif isinstance(image, Image.Image):
                # PIL Image
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                return img_byte_arr.getvalue()
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            raise
    
    async def classify(
        self, 
        image: Union[str, bytes, Image.Image], 
        categories: List[str] = None
    ) -> Dict[str, float]:
        """
        Classify an image against a list of categories.
        
        Args:
            image: Image to classify, can be a file path, bytes, or PIL Image
            categories: List of category labels to classify against (uses defaults if None)
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        # Validate categories
        if categories is None:
            categories = self.default_categories
        else:
            categories = self._validate_categories(categories)
            
        logger.info(f"Classifying image with categories: {categories}")
        
        try:
            # Prepare the image
            image_bytes = self._prepare_image(image)
            
            # Classify the image
            results = await self.client.classify_image(image_bytes, categories)
            
            # Sort results by confidence score (descending)
            sorted_results = dict(
                sorted(results.items(), key=lambda x: x[1], reverse=True)
            )
            
            logger.info(f"Classification complete: {sorted_results}")
            return sorted_results
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Return empty results on error
            return {category: 0.0 for category in categories}
    
    async def batch_classify(
        self, 
        images: List[Union[str, bytes, Image.Image]], 
        categories: List[str] = None
    ) -> List[Dict[str, float]]:
        """
        Classify multiple images in a batch.
        
        Args:
            images: List of images to classify
            categories: List of category labels to classify against (uses defaults if None)
            
        Returns:
            List of dictionaries mapping categories to confidence scores for each image
        """
        # Validate categories
        if categories is None:
            categories = self.default_categories
        else:
            categories = self._validate_categories(categories)
            
        logger.info(f"Batch classifying {len(images)} images with categories: {categories}")
        
        # Create a list of classification tasks
        tasks = [self.classify(image, categories) for image in images]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, handling any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error classifying image {i}: {result}")
                # Add empty results for failed classifications
                processed_results.append({category: 0.0 for category in categories})
            else:
                processed_results.append(result)
        
        logger.info(f"Batch classification complete for {len(images)} images")
        return processed_results


