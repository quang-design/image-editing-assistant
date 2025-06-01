import os
import mimetypes
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize client with proper error handling
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    raise ValueError(f"Failed to initialize Gemini client: {e}")

MODEL_NAME = "gemini-2.5-flash-preview-05-20"

def generate(prompt="", image=None, system_instruction="", response_schema=None, response_mime_type=None):
    """
    Generate content using Gemini API with support for structured output
    
    Args:
        prompt: Text prompt
        image: Path to image file or None
        system_instruction: System instruction for the model
        response_schema: Pydantic model for structured output
        response_mime_type: MIME type for response (e.g., 'application/json')
    """
    logger.info(f"Starting Gemini API call - Model: {MODEL_NAME}, Has image: {image is not None}, Prompt length: {len(prompt) if prompt else 0}")
    
    try:
        # Create contents list based on whether image is provided
        contents = []
        
        # Load image if provided
        if image is not None:
            logger.info(f"Processing image: {image}")
            # Check if the image file exists
            if not os.path.isfile(image):
                raise FileNotFoundError(f"Image file not found: {image}")
                
            # Get the MIME type of the image
            mime_type, _ = mimetypes.guess_type(image)
            if not mime_type or not mime_type.startswith('image/'):
                raise ValueError(f"File is not a recognized image format: {image}")
                
            # Read the image file as bytes
            with open(image, 'rb') as f:
                image_data = f.read()
                
            logger.info(f"Image loaded successfully - MIME type: {mime_type}, Size: {len(image_data)} bytes")
            
            # Create image part for Gemini API
            image_part = types.Part.from_bytes(mime_type=mime_type, data=image_data)
            contents.append(image_part)
            
        # Add text prompt if provided
        if prompt:
            contents.append(prompt)
        
        # Ensure contents list is not empty
        if not contents:
            raise ValueError("Either prompt or image must be provided")
        
        # Create config with optional structured output
        config_kwargs = {}
        if system_instruction:
            config_kwargs['system_instruction'] = system_instruction
        if response_schema:
            config_kwargs['response_schema'] = response_schema
            logger.info("Using structured output with response schema")
        if response_mime_type:
            config_kwargs['response_mime_type'] = response_mime_type
            
        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        
        logger.info("Making API call to Gemini")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config
        )
        
        logger.info(f"Gemini API call completed successfully - Response length: {len(response.text) if response.text else 0}")
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}", exc_info=True)
        return str(e)

def create_chat_session(model=MODEL_NAME):
    """Create a new chat session"""
    try:
        return client.chats.create(model=model)
    except Exception as e:
        raise ValueError(f"Failed to create chat session: {e}")

def generate_with_schema(prompt="", image=None, schema_class=None, system_instruction=""):
    """
    Generate structured output using a Pydantic schema
    
    Args:
        prompt: Text prompt
        image: Path to image file or None
        schema_class: Pydantic model class for structured output
        system_instruction: System instruction for the model
    """
    return generate(
        prompt=prompt,
        image=image,
        system_instruction=system_instruction,
        response_schema=schema_class,
        response_mime_type='application/json'
    )

def parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response with error handling"""
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        # Try to extract JSON from response if it's wrapped in other text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON response: {e}")