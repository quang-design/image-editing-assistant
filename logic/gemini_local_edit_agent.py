import json
import os
from typing import List, Dict, Any
import numpy as np
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from model.gemini import generate, generate_with_schema
from logic.models import BoundingBox
from google import genai
from google.genai import types

class DetectionResult(BaseModel):
    """Schema for object detection results from Gemini"""
    objects: List[Dict[str, Any]]

class GeminiLocalEditAgent:
    """Handles object detection and local edits using Gemini API"""
    
    def __init__(self, client=None):
        """Initialize the Gemini local edit agent"""
        try:
            # Initialize Gemini client
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            print("Gemini Local Edit Agent initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            raise ValueError(f"Failed to initialize Gemini client: {e}")
    
    def process_local_edit(self, image_path: str, prompt: str) -> dict:
        """Process local edit request with Gemini-based object detection and editing"""
        try:
            # Step 1: Detect objects and get bounding boxes
            detected_objects = self._detect_objects_with_gemini(image_path, prompt)
            
            if not detected_objects:
                return {
                    "edited_image_path": image_path,
                    "detected_objects": [],
                    "edited_regions": [],
                    "message": "No objects detected for local editing. Please be more specific about what you want to edit."
                }
            
            # Step 2: Edit the image using Gemini's image generation
            edited_path = self._edit_image_with_gemini(image_path, prompt, detected_objects)
            
            return {
                "edited_image_path": edited_path,
                "detected_objects": detected_objects,
                "edited_regions": detected_objects,  # For now, assume all detected objects are edited
                "message": f"Successfully processed local edit request: '{prompt}'"
            }
            
        except Exception as e:
            print(f"Error in process_local_edit: {e}")
            return {
                "edited_image_path": image_path,
                "detected_objects": [],
                "edited_regions": [],
                "message": f"Error processing local edit: {str(e)}"
            }
    
    def _detect_objects_with_gemini(self, image_path: str, prompt: str) -> List[BoundingBox]:
        """Use Gemini to detect objects in the image and return bounding boxes"""
        try:
            # Create a detection prompt
            detection_prompt = f"""
            Analyze this image and identify objects that are relevant to this editing request: "{prompt}"
            
            For each relevant object, provide:
            1. A descriptive label
            2. Bounding box coordinates (x1, y1, x2, y2) as percentages of image dimensions (0-100)
            3. Confidence score (0-1)
            
            Return the results as a JSON object with this structure:
            {{
                "objects": [
                    {{
                        "label": "object_name",
                        "x1": 10.5,
                        "y1": 20.3,
                        "x2": 45.7,
                        "y2": 60.8,
                        "confidence": 0.95
                    }}
                ]
            }}
            
            Only include objects that are clearly relevant to the editing request.
            """
            
            # Call Gemini for object detection
            response = generate(
                prompt=detection_prompt,
                image=image_path,
                response_mime_type='application/json'
            )
            
            # Parse the response
            detection_data = json.loads(response.strip())
            
            # Load image to get dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            # Convert to BoundingBox objects with absolute coordinates
            bounding_boxes = []
            for obj in detection_data.get("objects", []):
                # Convert percentage coordinates to absolute coordinates
                x1 = int((obj["x1"] / 100.0) * img_width)
                y1 = int((obj["y1"] / 100.0) * img_height)
                x2 = int((obj["x2"] / 100.0) * img_width)
                y2 = int((obj["y2"] / 100.0) * img_height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(x1, min(x2, img_width))
                y2 = max(y1, min(y2, img_height))
                
                bbox = BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=obj["label"],
                    confidence=obj["confidence"]
                )
                bounding_boxes.append(bbox)
                print(f"Detected {obj['label']} with confidence {obj['confidence']:.3f} at ({x1}, {y1}, {x2}, {y2})")
            
            return bounding_boxes
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []
    
    def _edit_image_with_gemini(self, image_path: str, prompt: str, detected_objects: List[BoundingBox]) -> str:
        """Use Gemini's image generation to edit the image"""
        try:
            # Load the original image
            with Image.open(image_path) as original_image:
                original_image = original_image.convert('RGB')
            
            # Create editing prompt
            object_descriptions = [obj.label for obj in detected_objects]
            editing_prompt = f"""
            {prompt}
            
            Focus on editing these detected objects: {', '.join(object_descriptions)}
            
            Please edit the image according to the request while maintaining the overall composition and quality.
            Make sure the edits look natural and blend well with the rest of the image.
            """
            
            # Save original image temporarily for Gemini API
            temp_path = image_path.replace('.', '_temp.')
            original_image.save(temp_path)
            
            try:
                # Call Gemini's image generation model
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=[editing_prompt, original_image],
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE']
                    )
                )
                
                # Extract the generated image
                edited_image = None
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        edited_image = Image.open(BytesIO(part.inline_data.data))
                        break
                
                if edited_image is None:
                    print("No image generated by Gemini")
                    return image_path
                
                # Save the edited image
                base_name, ext = os.path.splitext(image_path)
                output_path = f"{base_name}_gemini_edited{ext}"
                edited_image.save(output_path, quality=95 if ext.lower() in ['.jpg', '.jpeg'] else None)
                print(f"Saved Gemini-edited image to {output_path}")
                
                return output_path
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        except Exception as e:
            print(f"Error in image editing: {e}")
            return image_path
    
    def detect_and_edit(self, image_path: str, prompt: str) -> str:
        """Combined function for object detection and editing (for compatibility)"""
        result = self.process_local_edit(image_path, prompt)
        return result["edited_image_path"]
