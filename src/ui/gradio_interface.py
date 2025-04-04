"""
Gradio interface for the Gemini Image Classifier.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Union, Tuple
import numpy as np

import gradio as gr
from PIL import Image

from src.classifier.image_classifier import ImageClassifier
from src.config import (
    DEFAULT_CATEGORIES, 
    UI_TITLE, 
    UI_DESCRIPTION, 
    UI_THEME,
    UI_PORT
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the classifier
classifier = ImageClassifier()

async def process_image(
    image: np.ndarray, 
    use_default_categories: bool,
    custom_categories: str = ""
) -> Dict[str, float]:
    """
    Process a single image for classification.
    
    Args:
        image: Image data from Gradio
        use_default_categories: Whether to use default categories
        custom_categories: Comma-separated custom categories if not using defaults
        
    Returns:
        Dictionary with classification results
    """
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Determine which categories to use
        if use_default_categories:
            categories = DEFAULT_CATEGORIES
        else:
            # Parse custom categories
            categories = [cat.strip() for cat in custom_categories.split(',') if cat.strip()]
            if not categories:
                # Fall back to defaults if no valid custom categories
                categories = DEFAULT_CATEGORIES
        
        logger.info(f"Processing image with categories: {categories}")
        
        # Classify the image
        results = await classifier.classify(pil_image, categories)
        
        # Sort results for display
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_results
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {"Error": 1.0}

async def process_batch(
    images: List[np.ndarray],
    use_default_categories: bool,
    custom_categories: str = ""
) -> List[Dict[str, float]]:
    """
    Process multiple images for classification.
    
    Args:
        images: List of image data from Gradio
        use_default_categories: Whether to use default categories
        custom_categories: Comma-separated custom categories if not using defaults
        
    Returns:
        List of dictionaries with classification results for each image
    """
    try:
        # Determine which categories to use
        if use_default_categories:
            categories = DEFAULT_CATEGORIES
        else:
            # Parse custom categories
            categories = [cat.strip() for cat in custom_categories.split(',') if cat.strip()]
            if not categories:
                # Fall back to defaults if no valid custom categories
                categories = DEFAULT_CATEGORIES
        
        logger.info(f"Processing batch of {len(images)} images with categories: {categories}")
        
        # Convert numpy arrays to PIL Images
        pil_images = [Image.fromarray(img) for img in images]
        
        # Classify all images
        results = await classifier.batch_classify(pil_images, categories)
        
        return results
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return [{"Error": 1.0} for _ in images]

def sync_process_image(
    image: np.ndarray, 
    use_default_categories: bool,
    custom_categories: str = ""
) -> Dict[str, float]:
    """
    Synchronous wrapper for the async image processing function.
    """
    return asyncio.run(process_image(image, use_default_categories, custom_categories))

def sync_process_batch(
    images: List[np.ndarray],
    use_default_categories: bool,
    custom_categories: str = ""
) -> List[Dict[str, float]]:
    """
    Synchronous wrapper for the async batch processing function.
    """
    return asyncio.run(process_batch(images, use_default_categories, custom_categories))

def format_results(results: Dict[str, float]) -> str:
    """
    Format classification results for display.
    
    Args:
        results: Dictionary mapping categories to confidence scores
        
    Returns:
        Formatted string for display
    """
    if "Error" in results:
        return "❌ An error occurred during classification."
    
    formatted = "## Classification Results\n\n"
    for category, score in results.items():
        # Convert score to percentage
        percentage = score * 100
        # Add bars for visual representation
        bars = "█" * int(percentage / 10)
        formatted += f"**{category}**: {percentage:.1f}% {bars}\n"
    
    return formatted

def create_interface() -> gr.Blocks:
    """
    Create the Gradio interface for the image classifier.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title=UI_TITLE, theme=UI_THEME) as demo:
        gr.Markdown(UI_DESCRIPTION)
        
        with gr.Tab("Single Image Classification"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(
                        type="numpy", 
                        label="Upload an image to classify"
                    )
                    
                    with gr.Row():
                        use_default = gr.Checkbox(
                            value=True, 
                            label=f"Use default categories ({', '.join(DEFAULT_CATEGORIES)})"
                        )
                        
                    with gr.Row():
                        custom_cats = gr.Textbox(
                            value="", 
                            label="Custom categories (comma-separated)",
                            visible=False
                        )
                        
                    # Show/hide custom categories based on checkbox
                    use_default.change(
                        fn=lambda x: {"visible": not x},
                        inputs=[use_default],
                        outputs=[custom_cats]
                    )
                    
                    classify_btn = gr.Button("Classify", variant="primary")
                
                with gr.Column(scale=1):
                    output_text = gr.Markdown(label="Classification Results")
            
            # Connect the button to the processing function
            classify_btn.click(
                fn=sync_process_image,
                inputs=[input_image, use_default, custom_cats],
                outputs=output_text,
                api_name="classify_single"
            ).then(
                fn=format_results,
                inputs=output_text,
                outputs=output_text
            )
        
        with gr.Tab("Batch Classification"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_images = gr.Gallery(
                        label="Upload images to classify",
                        object_fit="contain"
                    )
                    
                    # Upload button for multiple images
                    upload_button = gr.UploadButton(
                        "Click to Upload Images", 
                        file_types=["image"], 
                        file_count="multiple"
                    )
                    
                    with gr.Row():
                        batch_use_default = gr.Checkbox(
                            value=True, 
                            label=f"Use default categories ({', '.join(DEFAULT_CATEGORIES)})"
                        )
                        
                    with gr.Row():
                        batch_custom_cats = gr.Textbox(
                            value="", 
                            label="Custom categories (comma-separated)",
                            visible=False
                        )
                        
                    # Show/hide custom categories based on checkbox
                    batch_use_default.change(
                        fn=lambda x: {"visible": not x},
                        inputs=[batch_use_default],
                        outputs=[batch_custom_cats]
                    )
                    
                    batch_classify_btn = gr.Button("Classify All", variant="primary")
                
                with gr.Column(scale=1):
                    batch_output = gr.Dataframe(
                        headers=["Image", "Top Category", "Confidence"],
                        label="Batch Classification Results"
                    )
            
            # Handle file uploads to gallery
            upload_button.upload(
                fn=lambda files: [file.name for file in files],
                inputs=upload_button,
                outputs=input_images
            )
            
            # Connect the batch button to the processing function
            batch_classify_btn.click(
                fn=sync_process_batch,
                inputs=[input_images, batch_use_default, batch_custom_cats],
                outputs=batch_output,
                api_name="classify_batch"
            )
        
        # Add examples
        gr.Examples(
            examples=[
                ["examples/cat.jpg", True, ""],
                ["examples/car.jpg", True, ""],
                ["examples/food.jpg", False, "pasta, dessert, steak, salad, soup"],
            ],
            inputs=[input_image, use_default, custom_cats],
            outputs=output_text,
            fn=sync_process_image,
            cache_examples=True,
        )
        
        # Add documentation tab
        with gr.Tab("About"):
            gr.Markdown("""
            # About Gemini Image Classifier
            
            This application uses Google's Gemini 2.5 Flash model to classify images into various categories.
            The model is a state-of-the-art multimodal AI capable of understanding images and text.
            
            ## How it works
            
            1. Upload an image or multiple images
            2. Choose whether to use default categories or specify your own
            3. Click the "Classify" button to process the image(s)
            4. View the results, with confidence scores for each category
            
            ## Tips for best results
            
            - Use clear, well-lit images for the best classification accuracy
            - When using custom categories, be specific and use common terms
            - For batch processing, limit to around 10 images at a time for best performance
            
            ## Privacy
            
            Your images are processed by the Gemini API and are subject to Google's privacy policy.
            Images are not stored permanently beyond the duration needed for processing.
            """)
    
    return demo

if __name__ == "__main__":
    # If run directly, create and launch the interface
    demo = create_interface()
    demo.launch(server_port=UI_PORT)


