import os
import numpy as np
from PIL import Image, ImageEnhance
from pydantic import BaseModel
from model.gemini import generate_with_schema

class EditParameters(BaseModel):
    brightness: int = 0  # -100 to 100
    contrast: int = 0    # -100 to 100
    saturation: int = 0  # -100 to 100
    temperature: str = "neutral"  # cold/neutral/warm

class GlobalEditAgent:
    """Handles global image adjustments"""
    
    def __init__(self, client=None):
        # Client is not needed since we use the generate functions
        pass
    
    def edit_image(self, image_path: str, prompt: str) -> dict:
        """Apply global edits based on prompt"""
        
        # Parse editing intent using structured output
        edit_prompt = f"""
        Based on this request: "{prompt}"
        
        Determine the editing parameters:
        - brightness: -100 to 100 (0 = no change, positive = brighter, negative = darker)
        - contrast: -100 to 100 (0 = no change, positive = more contrast, negative = less contrast)  
        - saturation: -100 to 100 (0 = no change, positive = more vibrant, negative = less vibrant)
        - temperature: "cold", "neutral", or "warm" (cold = bluer, warm = warmer/redder)
        
        Examples:
        - "make it brighter" -> brightness: 30
        - "increase contrast" -> contrast: 40
        - "more vibrant colors" -> saturation: 50
        - "warmer tone" -> temperature: "warm"
        - "make it darker and cooler" -> brightness: -30, temperature: "cold"
        """
        
        try:
            response = generate_with_schema(
                prompt=edit_prompt,
                schema_class=EditParameters,
                system_instruction="You are an image editing parameter analyzer. Always respond with valid JSON containing the editing parameters."
            )
            
            params = EditParameters.model_validate_json(response)
            params_dict = params.model_dump()
            
        except Exception as e:
            print(f"Parameter parsing error: {e}")
            # Default safe parameters
            params_dict = {"brightness": 0, "contrast": 0, "saturation": 0, "temperature": "neutral"}
        
        try:
            # Apply edits
            img = Image.open(image_path)
            original_mode = img.mode
            
            # Convert to RGB for processing if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            edits_applied = []
            
            # Brightness
            if params_dict.get("brightness", 0) != 0:
                enhancer = ImageEnhance.Brightness(img)
                factor = 1 + (params_dict["brightness"] / 100)
                factor = max(0.1, min(3.0, factor))  # Clamp to reasonable range
                img = enhancer.enhance(factor)
                edits_applied.append(f"brightness {'+' if params_dict['brightness'] > 0 else ''}{params_dict['brightness']}")
            
            # Contrast
            if params_dict.get("contrast", 0) != 0:
                enhancer = ImageEnhance.Contrast(img)
                factor = 1 + (params_dict["contrast"] / 100)
                factor = max(0.1, min(3.0, factor))  # Clamp to reasonable range
                img = enhancer.enhance(factor)
                edits_applied.append(f"contrast {'+' if params_dict['contrast'] > 0 else ''}{params_dict['contrast']}")
            
            # Saturation
            if params_dict.get("saturation", 0) != 0:
                enhancer = ImageEnhance.Color(img)
                factor = 1 + (params_dict["saturation"] / 100)
                factor = max(0.0, min(3.0, factor))  # Clamp to reasonable range
                img = enhancer.enhance(factor)
                edits_applied.append(f"saturation {'+' if params_dict['saturation'] > 0 else ''}{params_dict['saturation']}")
            
            # Temperature adjustment
            if params_dict.get("temperature") == "warm":
                img = self._adjust_temperature(img, 1.1, 0.9)
                edits_applied.append("warmer temperature")
            elif params_dict.get("temperature") == "cold":
                img = self._adjust_temperature(img, 0.9, 1.1)
                edits_applied.append("cooler temperature")
            
            # Convert back to original mode if needed
            if original_mode != 'RGB' and original_mode in ['L', 'P', 'RGBA']:
                if original_mode == 'L':
                    img = img.convert('L')
                elif original_mode == 'P':
                    img = img.convert('P')
                elif original_mode == 'RGBA':
                    img = img.convert('RGBA')
            
            # Create task name for filename
            task_name = self._create_task_name(params_dict)
            
            # Save edited image with task name
            base_name, ext = os.path.splitext(image_path)
            output_path = f"{base_name}_{task_name}{ext}"
            img.save(output_path, quality=95 if ext.lower() in ['.jpg', '.jpeg'] else None)
            
            return {
                "edited_image_path": output_path,
                "edits_applied": edits_applied,
                "message": f"Applied global edits: {', '.join(edits_applied) if edits_applied else 'no changes needed'}"
            }
            
        except Exception as e:
            return {
                "error": f"Failed to apply edits: {str(e)}",
                "edited_image_path": image_path,
                "edits_applied": [],
                "message": "Global edit failed"
            }
    
    def _adjust_temperature(self, img: Image.Image, red_factor: float, blue_factor: float) -> Image.Image:
        """Adjust color temperature"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        data = np.array(img)
        data[:, :, 0] = np.clip(data[:, :, 0] * red_factor, 0, 255)  # Red
        data[:, :, 2] = np.clip(data[:, :, 2] * blue_factor, 0, 255)  # Blue
        
        return Image.fromarray(data.astype(np.uint8))
    
    def _create_task_name(self, params: dict) -> str:
        """Create descriptive task name based on edit parameters"""
        task_parts = []
        
        if params.get("brightness", 0) > 0:
            task_parts.append("brighter")
        elif params.get("brightness", 0) < 0:
            task_parts.append("darker")
            
        if params.get("contrast", 0) > 0:
            task_parts.append("highcontrast")
        elif params.get("contrast", 0) < 0:
            task_parts.append("lowcontrast")
            
        if params.get("saturation", 0) > 0:
            task_parts.append("vibrant")
        elif params.get("saturation", 0) < 0:
            task_parts.append("desaturated")
            
        if params.get("temperature") == "warm":
            task_parts.append("warm")
        elif params.get("temperature") == "cold":
            task_parts.append("cool")
        
        return "_".join(task_parts) if task_parts else "global_edit"