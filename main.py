import os
import logging
from logic.assistant import ImageEditingAssistant
from logic.router_agent import ActionType

def main():
    """Simple CLI for chatting with the Image Editing Assistant and managing images."""
    
    # Configure logging - only important actions
    logging.basicConfig(
        level=logging.WARNING,  # Reduce verbosity
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('image_assistant.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers to INFO for important actions only
    logging.getLogger('logic.assistant').setLevel(logging.INFO)
    logging.getLogger('logic.router_agent').setLevel(logging.WARNING)
    logging.getLogger('logic.info_agent').setLevel(logging.INFO)
    logging.getLogger('model.gemini').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Image Editing Assistant")
    
    loaded_image = None
    # Initialize the assistant
    assistant = ImageEditingAssistant()
    
    print("=== Image Editing Assistant ===\n")
    print("Commands:")
    print("  load <file_path>   Load an image")
    print("  clear              Clear the loaded image")
    print("  quit/bye           Exit the app")
    print("Type any other text to chat with the assistant.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            logger.info("User interrupted, exiting")
            print("\nExiting.")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "bye", "exit"]:
            logger.info("User requested quit")
            print("Goodbye!")
            break
            
        elif user_input.lower().startswith("load "):
            path = user_input[5:].strip()
            if not os.path.isfile(path):
                print(f"[Error] File not found: {path}")
            else:
                loaded_image = path
                print(f"[Loaded] {path}")
                
        elif user_input.lower() == "clear":
            loaded_image = None
            print("[Image cleared]")
            
        else:
            try:
                # Process request using the multi-agent system
                response = assistant.process_request(image_path=loaded_image, prompt=user_input)
                
                # Handle QUIT action from assistant
                if response.action == ActionType.QUIT:
                    print("Goodbye!")
                    break
                
                # Handle different response types based on the action
                elif response.action == ActionType.INFO and response.info_data:
                    # Format and display image information
                    print(f"Assistant: {response.info_data.description}")
                    
                elif response.action == ActionType.GLOBAL_EDIT and response.edit_data:
                    # Show global edit result
                    print(f"Assistant: {response.edit_data.message}")
                    print(f"[Edited image saved to: {response.edit_data.edited_image_path}]")
                    
                elif response.action == ActionType.LOCAL_EDIT and response.edit_data:
                    # Show local edit result
                    print(f"Assistant: {response.edit_data.message}")
                    print(f"[Edited image saved to: {response.edit_data.edited_image_path}]")
                    
                elif response.action == ActionType.CLARIFY and response.clarify_data:
                    # Show clarification message and suggestions
                    print(f"Assistant: {response.clarify_data.message}")
                    if response.clarify_data.suggested_prompts:
                        print("Suggested prompts:")
                        for i, suggestion in enumerate(response.clarify_data.suggested_prompts, 1):
                            print(f"  {i}. {suggestion}")
                            
                elif response.error:
                    # Show error message
                    print(f"[Error] {response.error.error}")
                    if response.error.details:
                        print(f"Details: {response.error.details}")
                        
                else:
                    # Generic response for simple answers
                    print(f"Assistant: I'll help you with that request.")
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                print(f"[Error] {e}")

    logger.info("Image Editing Assistant session ended")


if __name__ == "__main__":
    main()