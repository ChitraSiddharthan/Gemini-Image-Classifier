# Gemini Image Classifier

A powerful image classification application using Google's Gemini 2.5 Flash model with a user-friendly Gradio interface.

## Overview 

This project utilizes Google's Gemini 2.5 Flash, a multimodal AI model capable of understanding and classifying images. The application provides a simple web interface built with Gradio that allows users to:

- Upload images for classification
- Select from predefined classification categories or create custom ones
- View classification results with confidence scores
- Compare results across multiple images
<img width="779" alt="Screenshot_2025-04-03_at_9 01 26_PM" src="https://github.com/user-attachments/assets/09c6c0af-cc11-48b0-9b82-9100fa45ada5" />


## Features

- 🖼️ Classify images using Google's state-of-the-art Gemini 2.5 Flash model
- 🚀 Fast and efficient image processing
- 🧠 Leverage multimodal understanding capabilities of Gemini
- 🌐 Simple web interface with Gradio
- 🔄 Batch processing of multiple images
- 📊 Confidence scores for classification results

## Installation

### Prerequisites

- Python 3.9+
- Google AI API key (for Gemini access)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gemini-image-classifier.git
cd gemini-image-classifier
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on the example:
```bash
cp .env.example .env
```

5. Add your Google AI API key to the `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

Run the application:

```bash
python app.py
```

The Gradio interface will start and be accessible at `http://localhost:7860`.

### Using the Interface

1. Upload an image or multiple images
2. Select from predefined categories or enter custom categories
3. Click "Classify" to process the images
4. View the classification results and confidence scores

## Development

### Project Structure

The project follows a modular structure:
- `app.py`: Main entry point
- `src/classifier`: Contains the Gemini integration and classification logic
- `src/ui`: Contains the Gradio UI components
- `config.py`: Configuration settings

### Extending the Project

- Add new classification categories in `src/config.py`
- Modify the prompting strategy in `src/classifier/image_classifier.py`
- Enhance the UI in `src/ui/gradio_interface.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Google Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/flash/) for the powerful multimodal model
- [Gradio](https://gradio.app/) for the simple and powerful UI framework


