import gradio as gr
import numpy as np
import os
import tempfile
import shutil
import logging
from PIL import Image
from typing import Optional, Tuple, List
from logic.assistant import ImageEditingAssistant
from logic.router_agent import ActionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioImageEditingUI:
    """Gradio UI for the Image Editing Assistant"""
    
    def __init__(self, use_gemini_local_edit: bool = False):
        self.assistant = ImageEditingAssistant(use_gemini_local_edit=use_gemini_local_edit)
        self.current_image_path: Optional[str] = None
        self.temp_dir = tempfile.mkdtemp()
        self.use_gemini_local_edit = use_gemini_local_edit
        logger.info(f"Temporary directory created: {self.temp_dir}")
        if use_gemini_local_edit:
            logger.info("Gradio UI initialized with Gemini Local Edit Agent")
        else:
            logger.info("Gradio UI initialized with Standard Local Edit Agent")
    
    def save_image_from_editor(self, image_data) -> Optional[str]:
        """Save image from ImageEditor to temporary file"""
        if image_data is None:
            return None
        
        try:
            # Handle different image data formats
            if isinstance(image_data, dict):
                # ImageEditor returns dict with 'background' and 'layers'
                img_array = image_data.get('background')
                if img_array is None:
                    return None
            else:
                img_array = image_data
            
            # Convert numpy array to PIL Image
            if isinstance(img_array, np.ndarray):
                img = Image.fromarray(img_array.astype('uint8'))
            else:
                return None
            
            # Save to temporary file
            temp_path = os.path.join(self.temp_dir, f"temp_image_{len(os.listdir(self.temp_dir))}.png")
            img.save(temp_path)
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
    
    def process_chat_message(self, message: str, history: List, image_data) -> Tuple[List, str, Optional[np.ndarray]]:
        """Process chat message and return updated history and potentially updated image"""
        if not message.strip():
            return history, "", image_data
        
        # Save current image if available
        image_path = None
        if image_data is not None:
            image_path = self.save_image_from_editor(image_data)
            self.current_image_path = image_path
            logger.info(f"Image saved for processing: {image_path}")
        else:
            logger.info("No image data provided, using current image path")
        
        # Add user message to history using messages format
        history = history or []
        history.append({"role": "user", "content": message})
        
        try:
            logger.info(f"Processing request: '{message}' with image: {image_path or self.current_image_path}")
            
            # Process request through assistant
            response = self.assistant.process_request(
                image_path=image_path or self.current_image_path, 
                prompt=message
            )
            
            logger.info(f"Assistant response action: {response.action}")
            
            # Format response based on action type
            assistant_response = self.format_assistant_response(response)
            
            # Add assistant response to history using messages format
            history.append({"role": "assistant", "content": assistant_response})
            
            # Check if we have an edited image to replace the current one
            updated_image = self.get_edited_image_from_response(response, image_data)
            
            if updated_image is not None and not np.array_equal(updated_image, image_data if image_data is not None else np.array([])):
                logger.info("Image was updated by the assistant")
            else:
                logger.info("No image update from assistant")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
            updated_image = image_data
        
        return history, "", updated_image
    
    def get_edited_image_from_response(self, response, current_image) -> Optional[np.ndarray]:
        """Extract edited image from response and return as numpy array"""
        edited_path = None
        
        # Check for edited image path in different response types
        if response.action == ActionType.GLOBAL_EDIT and response.edit_data:
            edited_path = response.edit_data.edited_image_path
        elif response.action == ActionType.LOCAL_EDIT and response.edit_data:
            edited_path = response.edit_data.edited_image_path
        
        # Load and return edited image if available
        if edited_path and os.path.exists(edited_path):
            try:
                img = Image.open(edited_path)
                return np.array(img)
            except Exception as e:
                logger.error(f"Error loading edited image: {e}")
        
        return current_image
    
    def format_assistant_response(self, response) -> str:
        """Format assistant response for display in chat"""
        if response.error:
            return f"❌ **Error**: {response.error.error}\n{response.error.details or ''}"
        
        if response.action == ActionType.INFO and response.info_data:
            info = response.info_data
            return f"📊 **Image Analysis**:\n\n{info.description}\n\n" \
                   f"**Dimensions**: {info.metadata.width}x{info.metadata.height}\n" \
                   f"**Format**: {info.metadata.format}\n" \
                   f"**Color Space**: {info.metadata.color_space}"
        
        elif response.action == ActionType.GLOBAL_EDIT and response.edit_data:
            edit = response.edit_data
            return f"✨ **Global Edit Complete**:\n\n{edit.message}\n\n" \
                   f"**Edits Applied**: {', '.join(edit.edits_applied)}\n" \
                   f"📸 *Image updated in editor*"
        
        elif response.action == ActionType.LOCAL_EDIT and response.edit_data:
            edit = response.edit_data
            objects_info = ""
            if edit.detected_objects:
                objects_info = f"\n**Objects Detected**: {len(edit.detected_objects)} items"
            
            return f"🎯 **Local Edit Complete**:\n\n{edit.message}{objects_info}\n" \
                   f"📸 *Image updated in editor*"
        
        elif response.action == ActionType.CLARIFY and response.clarify_data:
            clarify = response.clarify_data
            suggestions = ""
            if clarify.suggested_prompts:
                suggestions = "\n\n**Suggestions**:\n" + \
                            "\n".join([f"• {prompt}" for prompt in clarify.suggested_prompts])
            
            return f"💭 {clarify.message}{suggestions}"
        
        elif response.action == ActionType.ANSWER and response.clarify_data:
            return f"💬 {response.clarify_data.message}"
        
        return "✅ Request processed successfully!"
    
    def download_current_image(self, image_data) -> Optional[str]:
        """Prepare current image for download"""
        if image_data is None:
            return None
        
        try:
            # Save image to download
            if isinstance(image_data, dict):
                img_array = image_data.get('background')
            else:
                img_array = image_data
            
            if img_array is None:
                return None
            
            img = Image.fromarray(img_array.astype('uint8'))
            download_path = os.path.join(self.temp_dir, "download_image.png")
            img.save(download_path)
            return download_path
            
        except Exception as e:
            logger.error(f"Error preparing download: {e}")
            return None
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="Image Editing Assistant",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-container {
                height: 400px;
            }
            """
        ) as demo:
            
            gr.Markdown(
                """
                # 🎨 Image Editing Assistant
                
                Upload an image, draw masks for inpainting, and chat with the AI assistant to edit your images!
                
                **Features:**
                - 📤 Upload images or use the built-in drawing tools
                - 🖌️ Draw masks for precise inpainting operations  
                - 💬 Chat with AI for intelligent image editing
                - 📥 Download your edited results
                - 🔄 Edited images automatically replace the original
                """
            )
            
            with gr.Row():
                # Left column - Image Editor
                with gr.Column(scale=2):
                    gr.Markdown("### 🖼️ Image Editor")
                    
                    image_editor = gr.ImageEditor(
                        label="Upload or Edit Image",
                        type="numpy",
                        brush=gr.Brush(
                            default_size=20,
                            colors=["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"],
                            default_color="#FF0000"
                        ),
                        eraser=gr.Eraser(default_size=20),
                        layers=False,  # Disable layers for simplicity
                        sources=["upload", "webcam", "clipboard"],
                        transforms=["crop", "flip", "rotate"],
                        interactive=True,
                        show_fullscreen_button=True,
                        height=500
                    )
                    
                    with gr.Row():
                        download_btn = gr.Button("📥 Download Image", variant="primary")
                    
                    download_file = gr.File(
                        label="Download Ready",
                        visible=False,
                        interactive=False
                    )
                
                # Right column - Chat Interface
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 Chat with Assistant")
                    
                    chatbot = gr.Chatbot(
                        label="Image Editing Assistant",
                        height=400,
                        show_label=False,
                        container=True,
                        bubble_full_width=False,
                        avatar_images=(None, "🤖"),
                        type="messages"  # Fix the deprecation warning
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me to edit your image... (e.g., 'Make it brighter', 'Remove the background', 'What's in this image?')",
                            label="Message",
                            show_label=False,
                            scale=4,
                            container=False
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    gr.Markdown(
                        """
                        **Example prompts:**
                        - "What's in this image?"
                        - "Make the image brighter"
                        - "Remove the object in the center"
                        - "Increase the contrast"
                        - "Apply a vintage filter"
                        
                        **Tips:**
                        - Use the brush to draw masks for inpainting
                        - Use the eraser to remove unwanted brush strokes
                        - Edited images will automatically replace the original
                        """
                    )
            
            # Event handlers
            def send_message(message, history, image_data):
                return self.process_chat_message(message, history, image_data)
            
            def prepare_download(image_data):
                download_path = self.download_current_image(image_data)
                if download_path:
                    return gr.File(value=download_path, visible=True)
                return gr.File(visible=False)
            
            # Connect event handlers
            send_btn.click(
                fn=send_message,
                inputs=[msg_input, chatbot, image_editor],
                outputs=[chatbot, msg_input, image_editor]  # Now updates image_editor too
            )
            
            msg_input.submit(
                fn=send_message,
                inputs=[msg_input, chatbot, image_editor],
                outputs=[chatbot, msg_input, image_editor]  # Now updates image_editor too
            )
            
            download_btn.click(
                fn=prepare_download,
                inputs=[image_editor],
                outputs=[download_file]
            )
            
            # Welcome message
            demo.load(
                lambda: [{"role": "assistant", "content": "👋 Hello! I'm your Image Editing Assistant. Upload an image and tell me how you'd like to edit it!"}],
                outputs=[chatbot]
            )
        
        return demo
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        demo = self.create_interface()
        return demo.launch(**kwargs)
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary directory cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")

def main():
    """Launch the Gradio UI"""
    ui = GradioImageEditingUI()
    
    try:
        ui.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ui.cleanup()

if __name__ == "__main__":
    main()
