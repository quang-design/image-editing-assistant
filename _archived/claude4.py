import os
import json
import base64
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from google import genai
from google.genai import types


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

@dataclass
class EditRequest:
    action: str
    parameters: Dict
    bounding_box: Optional[BoundingBox] = None

class AgentRouter:
    """Routes requests to appropriate agents based on user intent"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    def route_request(self, image_path: str, prompt: str) -> str:
        """Determine which agent should handle the request"""
        
        routing_prompt = f"""
        Analyze this user request and determine the appropriate action:
        User prompt: "{prompt}"
        
        Respond with ONE of these actions:
        - "info": Get image information (resolution, histogram, metadata)
        - "global_edit": Apply global edits (brightness, contrast, color temperature)
        - "local_edit": Edit specific objects/regions (inpainting, object removal)
        - "clarify": Need more information from user
        
        Only respond with the action name.
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=routing_prompt
        )
        return response.text.strip().lower()

class ImageInfoAgent:
    """Provides detailed information about images"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    def analyze_image(self, image_path: str, prompt: str) -> Dict:
        """Extract comprehensive image information"""
        
        # Load image for technical analysis
        img = Image.open(image_path)
        
        # Basic metadata
        info = {
            "resolution": f"{img.width}x{img.height}",
            "format": img.format,
            "mode": img.mode,
            "file_size": os.path.getsize(image_path)
        }
        
        # Use Gemini for content analysis
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        analysis_prompt = f"""
        Analyze this image and provide:
        1. Main subjects/objects
        2. Scene description
        3. Lighting conditions
        4. Color palette
        5. Image quality assessment
        
        User specific question: {prompt}
        """
        
        image_part = types.Part.from_bytes(
            data=image_bytes, 
            mime_type='image/jpeg'
        )
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[analysis_prompt, image_part]
        )
        
        info["analysis"] = response.text
        return info

class GlobalEditAgent:
    """Handles global image adjustments"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    def edit_image(self, image_path: str, prompt: str) -> str:
        """Apply global edits based on prompt"""
        
        # Parse editing intent
        edit_prompt = f"""
        Based on this request: "{prompt}"
        
        Determine the editing parameters:
        - brightness: -100 to 100 (0 = no change)
        - contrast: -100 to 100 (0 = no change)  
        - saturation: -100 to 100 (0 = no change)
        - temperature: cold/neutral/warm
        
        Respond in JSON format only:
        {{"brightness": 0, "contrast": 0, "saturation": 0, "temperature": "neutral"}}
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=edit_prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        
        try:
            params = json.loads(response.text.strip())
        except:
            params = {"brightness": 0, "contrast": 0, "saturation": 0, "temperature": "neutral"}
        
        # Apply edits
        img = Image.open(image_path)
        
        # Brightness
        if params.get("brightness", 0) != 0:
            enhancer = ImageEnhance.Brightness(img)
            factor = 1 + (params["brightness"] / 100)
            img = enhancer.enhance(factor)
        
        # Contrast
        if params.get("contrast", 0) != 0:
            enhancer = ImageEnhance.Contrast(img)
            factor = 1 + (params["contrast"] / 100)
            img = enhancer.enhance(factor)
        
        # Saturation
        if params.get("saturation", 0) != 0:
            enhancer = ImageEnhance.Color(img)
            factor = 1 + (params["saturation"] / 100)
            img = enhancer.enhance(factor)
        
        # Temperature adjustment
        if params.get("temperature") == "warm":
            img = self._adjust_temperature(img, 1.1, 0.9)
        elif params.get("temperature") == "cold":
            img = self._adjust_temperature(img, 0.9, 1.1)
        
        # Save edited image
        output_path = image_path.replace('.', '_edited.')
        img.save(output_path)
        return output_path
    
    def _adjust_temperature(self, img: Image.Image, red_factor: float, blue_factor: float) -> Image.Image:
        """Adjust color temperature"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        data = np.array(img)
        data[:, :, 0] = np.clip(data[:, :, 0] * red_factor, 0, 255)  # Red
        data[:, :, 2] = np.clip(data[:, :, 2] * blue_factor, 0, 255)  # Blue
        
        return Image.fromarray(data.astype(np.uint8))

class LocalEditAgent:
    """Handles object detection and local edits like inpainting"""
    
    def __init__(self, client: genai.Client):
        self.client = client
    
    def detect_objects(self, image_path: str, prompt: str) -> List[BoundingBox]:
        """Detect objects using Gemini vision"""
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        detection_prompt = f"""
        Identify and locate objects in this image based on: "{prompt}"
        
        For each object, provide bounding box coordinates as percentage of image:
        Format: [x%, y%, width%, height%] where (x,y) is top-left corner
        
        Respond in JSON format:
        {{"objects": [{{"name": "object1", "bbox": [10, 20, 30, 40]}}, ...]}}
        """
        
        image_part = types.Part.from_bytes(
            data=image_bytes, 
            mime_type='image/jpeg'
        )
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[detection_prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        
        try:
            result = json.loads(response.text.strip())
            bboxes = []
            img = Image.open(image_path)
            
            for obj in result.get("objects", []):
                bbox_percent = obj["bbox"]
                bbox = BoundingBox(
                    x=int(bbox_percent[0] * img.width / 100),
                    y=int(bbox_percent[1] * img.height / 100),
                    width=int(bbox_percent[2] * img.width / 100),
                    height=int(bbox_percent[3] * img.height / 100)
                )
                bboxes.append(bbox)
            
            return bboxes
        except:
            return []
    
    def inpaint_region(self, image_path: str, bounding_box: BoundingBox, prompt: str) -> str:
        """Simple inpainting by blurring/filling region"""
        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Create mask for the region
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, 
                     (bounding_box.x, bounding_box.y),
                     (bounding_box.x + bounding_box.width, 
                      bounding_box.y + bounding_box.height),
                     255, -1)
        
        # Simple inpainting using OpenCV
        inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        # Save result
        output_path = image_path.replace('.', '_inpainted.')
        cv2.imwrite(output_path, inpainted)
        return output_path

class ImageEditingAssistant:
    """Main assistant coordinating all agents"""
    
    def __init__(self, api_key: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        # Initialize client with API key or environment variable
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            # Uses GOOGLE_API_KEY environment variable
            self.client = genai.Client()
        
        # Initialize agents
        self.router = AgentRouter(self.client)
        self.info_agent = ImageInfoAgent(self.client)
        self.global_agent = GlobalEditAgent(self.client)
        self.local_agent = LocalEditAgent(self.client)
    
    def process_request(self, image_path: str, prompt: str) -> Dict:
        """Process user request through appropriate agents"""
        
        try:
            # Route request
            action = self.router.route_request(image_path, prompt)
            
            result = {"action": action}
            
            if action == "info":
                result["data"] = self.info_agent.analyze_image(image_path, prompt)
                
            elif action == "global_edit":
                output_path = self.global_agent.edit_image(image_path, prompt)
                result["edited_image"] = output_path
                
            elif action == "local_edit":
                # Detect objects first
                bboxes = self.local_agent.detect_objects(image_path, prompt)
                if bboxes:
                    # Use first detected object for inpainting
                    output_path = self.local_agent.inpaint_region(image_path, bboxes[0], prompt)
                    result["edited_image"] = output_path
                    result["detected_objects"] = len(bboxes)
                else:
                    result["error"] = "No objects detected for local editing"
                    
            elif action == "clarify":
                result["message"] = "Please provide more specific details about what you'd like to do with the image."
            
            return result
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

# Advanced features using new API capabilities
class AdvancedImageEditingAssistant(ImageEditingAssistant):
    """Extended assistant with advanced Gemini 2.0 features"""
    
    def chat_based_editing(self, image_path: str, initial_prompt: str) -> Dict:
        """Multi-turn conversation for iterative editing"""
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes, 
            mime_type='image/jpeg'
        )
        
        # Create chat session
        chat = self.client.chats.create(model='gemini-2.0-flash-001')
        
        # Send initial message with image
        response = chat.send_message([
            "I'll help you edit this image. What would you like to do?",
            image_part
        ])
        
        return {
            "chat_id": id(chat),  # In real app, store chat sessions
            "initial_response": response.text,
            "chat_session": chat
        }
    
    def structured_edit_analysis(self, image_path: str, prompt: str) -> Dict:
        """Get structured analysis using response schema"""
        
        from pydantic import BaseModel
        from typing import List as PyList
        
        class ImageAnalysis(BaseModel):
            objects: PyList[str]
            dominant_colors: PyList[str]
            lighting: str
            scene_type: str
            quality_score: int
            suggested_edits: PyList[str]
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes, 
            mime_type='image/jpeg'
        )
        
        analysis_prompt = f"""
        Analyze this image comprehensively and provide:
        - List of main objects/subjects
        - Dominant colors (max 5)
        - Lighting conditions (natural/artificial/mixed/poor)
        - Scene type (portrait/landscape/macro/street/etc)
        - Quality score (1-10)
        - Suggested improvements/edits
        
        User request: {prompt}
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[analysis_prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=ImageAnalysis
            )
        )
        
        return json.loads(response.text)

# Usage example
def main():
    # Initialize assistant - uses GOOGLE_API_KEY environment variable
    assistant = ImageEditingAssistant()
    
    # For advanced features
    advanced_assistant = AdvancedImageEditingAssistant()
    
    # Example usage
    image_path = "test_images/test.png"
    
    # Test different types of requests
    requests = [
        "What's in this image?",
        "Make it brighter and more vibrant", 
        "Remove the person in the background",
        "Add more contrast"
    ]
    
    for prompt in requests:
        print(f"\nPrompt: {prompt}")
        result = assistant.process_request(image_path, prompt)
        print(f"Result: {result}")
    
    # Advanced structured analysis
    try:
        structured_result = advanced_assistant.structured_edit_analysis(
            image_path, 
            "Analyze this image for potential improvements"
        )
        print(f"\nStructured Analysis: {structured_result}")
    except Exception as e:
        print(f"Advanced features require valid image: {e}")

if __name__ == "__main__":
    main()