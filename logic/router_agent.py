from model.gemini import generate_with_schema
import enum
import logging
from pydantic import BaseModel

class ActionType(str, enum.Enum):
    ANSWER = "answer"
    INFO = "info"
    GLOBAL_EDIT = "global_edit"
    LOCAL_EDIT = "local_edit"
    CLARIFY = "clarify"
    QUIT = "quit"

class RouterResponse(BaseModel):
    action: ActionType

class AgentRouter:
    """Routes requests to appropriate agents based on user intent"""
    
    def __init__(self, client=None):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        # Client is not needed since we use the generate functions
        pass
    
    def route_request(self, image_path: str, prompt: str) -> ActionType:
        """Determine which agent should handle the request"""
        
        self.logger.info(f"Routing request - Prompt: {prompt[:50]}...")
        
        routing_prompt = f"""
        Analyze this user request and determine the appropriate action:
        User prompt: "{prompt}"
        
        Respond with ONE of these actions:
        - "answer": Simple questions or greetings that don't require image processing
        - "info": Get image information (resolution, histogram, metadata, description)
        - "global_edit": Apply global edits (brightness, contrast, color temperature, saturation)
        - "local_edit": Edit specific objects/regions (inpainting, object removal, object detection)
        - "clarify": Need more information from user
        - "quit": Exit the app
        
        Examples:
        - "What's in this image?" -> info
        - "Make it brighter" -> global_edit
        - "Remove the person" -> local_edit
        - "Hello" -> answer
        - "Can you help me?" -> clarify
        """
        
        try:
            self.logger.info("Calling Gemini API for request routing")
            response = generate_with_schema(
                prompt=routing_prompt,
                schema_class=RouterResponse,
                system_instruction="You are a routing agent that determines the appropriate action for image editing requests. Always respond with valid JSON containing one of the specified actions."
            )
            
            # Parse the response as a RouterResponse object
            router_response = RouterResponse.model_validate_json(response)
            self.logger.info(f"Routing completed - Action determined: {router_response.action}")
            return router_response.action
            
        except Exception as e:
            self.logger.error(f"Router error: {e}", exc_info=True)
            # Default to clarify if routing fails
            self.logger.info("Defaulting to CLARIFY action due to routing failure")
            return ActionType.CLARIFY