%%writefile ./logic/local_edit_agent.py
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
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float
    action_prompt: str

class DetectionPromptResult(BaseModel):
    name: List[str]
    action_prompt: List[str]

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

class LocalEditAgent:
    """Handles object detection and local edits like inpainting"""
    def __init__(self, client=None):

        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
     
        self.pipe_pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
        self.pipe_pix2pix.to("cuda")
        self.pipe_pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config( self.pipe_pix2pix.scheduler.config)
  
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        self.pipe = self.pipe.to("cuda")

    """
    def __init__(self, client=None):
        # Client is not needed since we use the generate functions
        pass
    """
    def process_local_edit(self, image_path: str, prompt: str) -> dict:
        """Process local edit request with object detection and inpainting"""
        try:
            # Use the merged function
            edited_path = self.detect_and_inpaint(image_path, prompt)
            
            # Check if editing was successful
            if edited_path == image_path:
                return {
                    "edited_image_path": image_path,
                    "detected_objects": [],
                    "edited_regions": [],
                    "message": "No objects detected for local editing. Please be more specific about what you want to edit."
                }
            else:
                return {
                    "edited_image_path": edited_path,
                    "detected_objects": [],  # Could extract this info if needed
                    "edited_regions": [],    # Could extract this info if needed
                    "message": f"Successfully processed local edit request: '{prompt}'"
                }

        except Exception as e:
            return {
                "edited_image_path": image_path,
                "detected_objects": [],
                "edited_regions": [],
                "message": f"Local edit failed: {str(e)}"
            }

    def detect_and_inpaint(self, image_path: str, prompt: str) -> str:
        """Detect objects and inpaint them in one function"""
        print("Starting detection and inpainting process...")
        
        # DETECTION PHASE
        detection_prompt = f"""
        Analyze this prompt to identify an object class to be detected based on this request: "{prompt}"

        Look for objects that the user wants to edit, remove, or modify.
        For each relevant object, provide:
        - A descriptive name of class, and action to do with this class
        Examples:
        - "remove the person" -> object class is person, object action is remove
        - "delete the car" -> object class is vehicles, object action is delete
        - "remove the person and delete the car" -> two object classes, object class is person and object class is vehicle, object action is remove and delete

        Respond in JSON format:
        {{"name": ["person","vehicle"], "action_prompt":["remove the person", "delete the car"]}}
        """
        
        try:
            # Generate detection schema
            response = generate_with_schema(
                prompt=detection_prompt,
                schema_class=DetectionPromptResult,
                system_instruction="You are an object detection assistant. Analyze this prompt and always respond with valid JSON."
            )

            # Load image
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Parse detection results
            result = json.loads(response.strip())
            list_object_class = result["name"]
            list_action_prompt = result["action_prompt"]

            texts = [["a photo of " + str(object_)] for object_ in list_object_class]
            action_prompts = [[str(object_)] for object_ in list_action_prompt]

            # Run object detection
            inputs = self.processor(text=texts, images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Process detection results
            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs=outputs, 
                target_sizes=target_sizes, 
                threshold=0.5
            )
            
            # Extract bounding boxes
            i = 0  # First image
            bboxes = []
            text = texts[i]
            action_prompt = action_prompts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                bbox = BoundingBox(
                    x=int(box[0]),
                    y=int(box[1]),
                    width=int(abs(box[0] - box[2])),
                    height=int(abs(box[1] - box[3])),
                    label=str(label),
                    confidence=score,
                    action_prompt=action_prompt[label]
                )
                bboxes.append(bbox)
                print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box} with action prompt: {action_prompt[label]}")

            print(f"Detected {len(bboxes)} objects")
            
            # Check if any objects were detected
            if len(bboxes) < 1:
                print("No objects detected for inpainting")
                return image_path

            # INPAINTING PHASE
            print("Starting inpainting phase...")
            
            # Work with the same PIL image
            img_pil = image.copy()  # Create a copy to avoid modifying original
            w, h = img_pil.size
            
            for bounding_box in bboxes:
                # Validate bounding box
                x = max(0, min(bounding_box.x, w-1))
                y = max(0, min(bounding_box.y, h-1))
                width = max(1, min(bounding_box.width, w - x))
                height = max(1, min(bounding_box.height, h - y))
                
                print(f"Processing region: ({x}, {y}, {width}, {height}) with prompt: {bounding_box.action_prompt}")
                
                # Crop the region using PIL
                img_bbox = img_pil.crop((int(x), int(y), int(x+width), int(y+height)))

                # Apply the diffusion model
              
                images = self.pipe_pix2pix(
                    bounding_box.action_prompt, 
                    image=img_bbox, 
                    num_inference_steps=10, 
                    image_guidance_scale=1
                ).images
           


                # Paste the edited region back
                img_pil.paste(images[0], (int(x), int(y)))
                print(f"Applied edit for: {bounding_box.action_prompt}")


            img_pil = self.pipe("make this image more realistic", image=img_pil).images[0]
            # Save the result
            task_name = "detect_inpaint_"
            base_name, ext = os.path.splitext(image_path)
            output_path = f"{base_name}_{task_name}{ext}"
            img_pil.save(output_path, quality=95 if ext.lower() in ['.jpg', '.jpeg'] else None)
            print(f"Saved edited image to {output_path}")
            return output_path

        except Exception as e:
            print(f"Detection and inpainting error: {e}")
            return image_path