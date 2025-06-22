#!/usr/bin/env python3
"""
Launch script for the Image Editing Assistant Gradio UI
"""

import os
import sys
import logging
from gradio_ui import GradioImageEditingUI

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gradio_ui.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('gradio_ui').setLevel(logging.INFO)
    logging.getLogger('logic.assistant').setLevel(logging.INFO)
    logging.getLogger('logic.router_agent').setLevel(logging.WARNING)
    logging.getLogger('logic.info_agent').setLevel(logging.INFO)
    logging.getLogger('model.gemini').setLevel(logging.WARNING)

def main():
    """Main function to launch the UI"""
    print("🎨 Starting Image Editing Assistant UI...")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found. Please create one with your API keys.")
        print("   You can copy .env.example to .env and fill in your values.")
    
    try:
        # Create and launch UI
        ui = GradioImageEditingUI()
        logger.info("Launching Gradio UI...")
        
        print("\n🚀 Launching Image Editing Assistant UI...")
        print("📱 The interface will be available at: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop the server\n")
        
        ui.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
        logger.info("User interrupted, shutting down")
    except Exception as e:
        print(f"❌ Error starting UI: {e}")
        logger.error(f"Failed to start UI: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'ui' in locals():
            ui.cleanup()
        print("✅ Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
