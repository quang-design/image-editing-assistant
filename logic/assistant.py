import logging
from logic.router_agent import AgentRouter, ActionType
from logic.info_agent import ImageInfoAgent
from logic.global_edit_agent import GlobalEditAgent
from logic.local_edit_agent import LocalEditAgent
from logic.gemini_local_edit_agent import GeminiLocalEditAgent
from logic.models import (
    BoundingBox, EditResponse, LocalEditResponse, 
    ClarifyResponse, ErrorResponse, AssistantResponse
)

class ImageEditingAssistant:
    """Main assistant coordinating all agents"""
    
    def __init__(self, api_key: str = None, use_gemini_local_edit: bool = False):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents (they don't need client anymore)
        self.router = AgentRouter()
        self.info_agent = ImageInfoAgent()
        self.global_agent = GlobalEditAgent()
        
        # Choose local edit agent based on configuration
        if use_gemini_local_edit:
            self.local_agent = GeminiLocalEditAgent()
            self.logger.info("Using Gemini Local Edit Agent")
        else:
            self.local_agent = LocalEditAgent()
            self.logger.info("Using Standard Local Edit Agent")
        
        self.logger.info("ImageEditingAssistant initialized with all agents")
    
    def process_request(self, image_path: str, prompt: str) -> AssistantResponse:
        """Process user request through appropriate agents"""
        
        try:
            # Route request using structured output
            action = self.router.route_request(image_path, prompt)
            self.logger.info(f"Action determined: {action}")
            
            # Initialize response with action
            response = AssistantResponse(action=action)
            
            if action == ActionType.ANSWER:
                self.logger.info("Processing ANSWER action")
                # Handle simple questions directly
                if prompt.lower() in ["hi", "hello", "hey"]:
                    return AssistantResponse(
                        action=ActionType.ANSWER,
                        clarify_data=ClarifyResponse(
                            message="Hello! How can I help you with your image today?",
                            suggested_prompts=[
                                "Show me information about this image",
                                "Increase the brightness of this image",
                                "Remove the object in the center of the image"
                            ]
                        )
                    )
                else:
                    return AssistantResponse(
                        action=ActionType.ANSWER,
                        clarify_data=ClarifyResponse(
                            message=f"I'm an image editing assistant. {prompt.capitalize() if prompt.endswith('?') else 'Can you please provide an image-related request?'}",
                            suggested_prompts=[
                                "Show me information about this image",
                                "Increase the brightness of this image",
                                "Remove the object in the center of the image"
                            ]
                        )
                    )
            
            elif action == ActionType.INFO:
                self.logger.info("Calling info agent")
                info_data = self.info_agent.analyze_image(image_path, prompt)
                response.info_data = info_data
                self.logger.info("Info agent completed")
                
            elif action == ActionType.GLOBAL_EDIT:
                self.logger.info("Calling global edit agent")
                edit_result = self.global_agent.edit_image(image_path, prompt)
                if "error" in edit_result:
                    self.logger.error(f"Global edit failed: {edit_result['error']}")
                    response.error = ErrorResponse(error=edit_result["error"])
                else:
                    self.logger.info("Global edit completed")
                    edit_data = EditResponse(
                        edited_image_path=edit_result["edited_image_path"],
                        edits_applied=edit_result["edits_applied"],
                        message=edit_result["message"]
                    )
                    response.edit_data = edit_data
                
            elif action == ActionType.LOCAL_EDIT:
                self.logger.info("Calling local edit agent")
                # Process local edit with object detection and inpainting
                local_result = self.local_agent.process_local_edit(image_path, prompt)
                
                # Convert to proper response format - handle both dict and BoundingBox objects
                detected_objects = []
                for obj in local_result.get("detected_objects", []):
                    if isinstance(obj, BoundingBox):
                        detected_objects.append(obj)
                    else:
                        detected_objects.append(BoundingBox(**obj))

                edited_regions = []
                for obj in local_result.get("edited_regions", []):
                    if isinstance(obj, BoundingBox):
                        edited_regions.append(obj)
                    else:
                        edited_regions.append(BoundingBox(**obj))
                
                local_edit_data = LocalEditResponse(
                    edited_image_path=local_result["edited_image_path"],
                    detected_objects=detected_objects,
                    edited_regions=edited_regions,
                    message=local_result["message"]
                )
                response.edit_data = local_edit_data
                self.logger.info("Local edit completed")
                    
            elif action == ActionType.CLARIFY:
                self.logger.info("Processing CLARIFY action")
                clarify_data = ClarifyResponse(
                    message="Please provide more specific details about what you'd like to do with the image.",
                    suggested_prompts=[
                        "Show me information about this image",
                        "Increase the brightness of this image",
                        "Remove the object in the center of the image"
                    ]
                )
                response.clarify_data = clarify_data
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {str(e)}")
            error_response = ErrorResponse(error=f"Processing failed", details=str(e))
            return AssistantResponse(action=ActionType.CLARIFY, error=error_response)