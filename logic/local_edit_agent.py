import json
import os
from typing import List, Dict, Any
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
from model.gemini import generate,generate_with_schema
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: str 
    confidence: float 

class DetectionPromptResult(BaseModel):
    name: List[str]

class LocalEditAgent:
    """Handles object detection and local edits like inpainting"""
    def __init__(self, client=None):
        # src: https://huggingface.co/google/owlv2-base-patch16-ensemble
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    def process_local_edit(self, image_path: str, prompt: str) -> dict:
        """Process local edit request with object detection and inpainting"""
        try:
            # First detect objects
            detected_objects = self.detect_objects_v2(image_path, prompt)
            
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
    
    def detect_objects_v2(self, image_path: str, prompt: str) -> List[BoundingBox]:
          
      # Get image dimensions
      img = Image.open(image_path)
      img_width, img_height = img.size
      detection_prompt = f"""
      Analyze this prompt to identify an object class to be detected based on this request: "{prompt}"

      Look for objects that the user wants to edit, remove, or modify.
      For each relevant object, provide:
      - A descriptive name of class
      Examples:
      - "remove the person" -> object class is person
      - "delete the car" -> object class is vehicles  
      - "remove the person and delete the car" -> two object classes, object class is person and object class is vehicle

      Respond in JSON format:
      {{"name": ["person","vehicle"]}}
      """
      try:
        response = generate_with_schema(
            prompt=detection_prompt,
            schema_class=DetectionPromptResult,
            system_instruction="You are an object detection assistant. Analyze . Always respond with valid JSON."
        )
        print(response)
        image = Image.open(image_path)
        result = json.loads(response.strip())
        list_object_class = result["name"]
        texts = [["a photo of " + str(object_)] for object_ in list_object_class]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        
        with torch.no_grad():
          outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.5)
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        bboxes = []
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            bbox = BoundingBox(
                  x=int(box[0]),
                  y=int(box[1]),
                  width=int(abs(box[0] - box[2])),
                  height=int(abs(box[1] - box[3])),
                  label=str(label),
                  confidence=score
              )

            if (bbox.x >= 0 and bbox.y >= 0 and 
                bbox.x + bbox.width <= img_width and 
                bbox.y + bbox.height <= img_height and
                bbox.width > 0 and bbox.height > 0):
                bboxes.append(bbox)
            
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
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