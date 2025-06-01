import json
import os
from typing import List, Dict, Any
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
from model.gemini import generate

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    confidence: float = 0.0

class DetectionResult(BaseModel):
    objects: List[Dict[str, Any]]

class LocalEditAgent:
    """Handles object detection and local edits like inpainting"""
    
    def __init__(self, client=None):
        # Client is not needed since we use the generate functions
        pass
    
    def process_local_edit(self, image_path: str, prompt: str) -> dict:
        """Process local edit request with object detection and inpainting"""
        try:
            # First detect objects
            detected_objects = self.detect_objects(image_path, prompt)
            
            if not detected_objects:
                return {
                    "edited_image_path": image_path,
                    "detected_objects": [],
                    "edited_regions": [],
                    "message": "No objects detected for local editing. Please be more specific about what you want to edit."
                }
            
            # For now, edit the first detected object
            # In a more advanced version, you could ask the user to choose
            bbox = detected_objects[0]
            edited_path = self.inpaint_region(image_path, bbox, prompt)
            
            return {
                "edited_image_path": edited_path,
                "detected_objects": [bbox.model_dump() for bbox in detected_objects],
                "edited_regions": [bbox.model_dump()],
                "message": f"Successfully edited region containing '{bbox.label}'"
            }
            
        except Exception as e:
            return {
                "edited_image_path": image_path,
                "detected_objects": [],
                "edited_regions": [],
                "message": f"Local edit failed: {str(e)}"
            }
    
    def detect_objects(self, image_path: str, prompt: str) -> List[BoundingBox]:
        """Detect objects using Gemini vision"""
        
        detection_prompt = f"""
        Analyze this image and identify objects based on this request: "{prompt}"
        
        Look for objects that the user wants to edit, remove, or modify.
        For each relevant object, provide:
        - A descriptive name
        - Bounding box coordinates as percentages of image dimensions
        - Confidence level (0.0 to 1.0)
        
        Format bounding box as [x_percent, y_percent, width_percent, height_percent] 
        where (x,y) is the top-left corner.
        
        Examples:
        - "remove the person" -> look for people
        - "delete the car" -> look for vehicles
        - "edit the background" -> identify background regions
        
        Respond in JSON format:
        {{"objects": [{{"name": "person", "bbox": [25, 30, 20, 40], "confidence": 0.9}}]}}
        """
        
        try:
            response = generate(
                prompt=detection_prompt,
                image=image_path,
                system_instruction="You are an object detection assistant. Analyze images and identify objects with their locations. Always respond with valid JSON."
            )
            
            # Parse response manually since we're not using structured output here
            result = json.loads(response.strip())
            bboxes = []
            
            # Get image dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            for obj in result.get("objects", []):
                bbox_percent = obj.get("bbox", [])
                if len(bbox_percent) == 4:
                    bbox = BoundingBox(
                        x=int(bbox_percent[0] * img_width / 100),
                        y=int(bbox_percent[1] * img_height / 100),
                        width=int(bbox_percent[2] * img_width / 100),
                        height=int(bbox_percent[3] * img_height / 100),
                        label=obj.get("name", "object"),
                        confidence=obj.get("confidence", 0.5)
                    )
                    # Validate bounding box is within image bounds
                    if (bbox.x >= 0 and bbox.y >= 0 and 
                        bbox.x + bbox.width <= img_width and 
                        bbox.y + bbox.height <= img_height and
                        bbox.width > 0 and bbox.height > 0):
                        bboxes.append(bbox)
            
            return bboxes
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return []
    
    def inpaint_region(self, image_path: str, bounding_box: BoundingBox, prompt: str) -> str:
        """Simple inpainting by blurring/filling region"""
        
        try:
            # Load image with OpenCV for inpainting
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            h, w = img_cv.shape[:2]
            
            # Validate bounding box
            x = max(0, min(bounding_box.x, w-1))
            y = max(0, min(bounding_box.y, h-1))
            width = max(1, min(bounding_box.width, w - x))
            height = max(1, min(bounding_box.height, h - y))
            
            # Create mask for the region
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)
            
            # Apply slight dilation to mask for better inpainting
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Inpainting using OpenCV
            inpainted = cv2.inpaint(img_cv, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            
            # Create task name based on prompt
            task_name = self._create_task_name(prompt)
            
            # Save result with task name
            base_name, ext = os.path.splitext(image_path)
            output_path = f"{base_name}_{task_name}{ext}"
            
            # Save with high quality
            if ext.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(output_path, inpainted, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(output_path, inpainted)
                
            return output_path
            
        except Exception as e:
            print(f"Inpainting error: {e}")
            # Return original path if inpainting fails
            return image_path
    
    def _create_task_name(self, prompt: str) -> str:
        """Create descriptive task name based on edit prompt"""
        prompt_lower = prompt.lower()
        
        if "remove" in prompt_lower:
            if "person" in prompt_lower or "people" in prompt_lower:
                return "remove_person"
            elif "car" in prompt_lower or "vehicle" in prompt_lower:
                return "remove_vehicle"
            elif "object" in prompt_lower:
                return "remove_object"
            elif "background" in prompt_lower:
                return "remove_background"
            else:
                return "remove_element"
        elif "delete" in prompt_lower:
            return "delete_object"
        elif "inpaint" in prompt_lower:
            return "inpaint"
        elif "fill" in prompt_lower:
            return "fill_region"
        elif "edit" in prompt_lower:
            return "edit_region"
        else:
            return "local_edit"