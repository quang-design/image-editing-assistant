#!/usr/bin/env python3
"""
Example usage of the Image Editing Assistant
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from logic.assistant import ImageEditingAssistant

def main():
    """Example usage of the assistant"""
    
    # Initialize the assistant
    assistant = ImageEditingAssistant()
    
    # Example image path (replace with your image)
    image_path = "sample_image.jpg"
    
    print("🎨 Image Editing Assistant Example")
    print("=" * 40)
    
    # Example 1: Get image information
    print("\n📊 Example 1: Getting image information")
    response = assistant.process_request(image_path, "Tell me about this image")
    
    if response.info_data:
        metadata = response.info_data.metadata
        print(f"Image: {metadata.width}x{metadata.height} {metadata.format}")
        print(f"Description: {response.info_data.description[:200]}...")
    
    # Example 2: Global edit - brightness adjustment
    print("\n🌞 Example 2: Adjusting brightness")
    response = assistant.process_request(image_path, "Make this image 20% brighter")
    
    if response.edit_data and hasattr(response.edit_data, 'edited_image_path'):
        print(f"Edited image saved to: {response.edit_data.edited_image_path}")
        print(f"Applied edits: {response.edit_data.edits_applied}")
    
    # Example 3: Local edit - object removal
    print("\n🎯 Example 3: Object detection and removal")
    response = assistant.process_request(image_path, "Remove the person in the center")
    
    if response.edit_data and hasattr(response.edit_data, 'detected_objects'):
        print(f"Detected {len(response.edit_data.detected_objects)} objects")
        for obj in response.edit_data.detected_objects:
            print(f"  - {obj.label} (confidence: {obj.confidence:.2f})")
        
        if hasattr(response.edit_data, 'edited_image_path'):
            print(f"Edited image saved to: {response.edit_data.edited_image_path}")
    
    # Example 4: Simple conversation
    print("\n💬 Example 4: Simple greeting")
    response = assistant.process_request(image_path, "Hello!")
    
    if response.clarify_data:
        print(f"Response: {response.clarify_data.message}")
        print("Suggested prompts:")
        for prompt in response.clarify_data.suggested_prompts:
            print(f"  - {prompt}")

def interactive_mode():
    """Interactive mode for testing the assistant"""
    
    assistant = ImageEditingAssistant()
    
    print("🎨 Interactive Image Editing Assistant")
    print("Enter 'quit' to exit")
    print("=" * 40)
    
    # Get image path
    image_path = input("Enter image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📸 Using image: {image_path}")
    
    while True:
        try:
            prompt = input("\n🤖 What would you like to do? ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("🔄 Processing...")
            response = assistant.process_request(image_path, prompt)
            
            print(f"📋 Action: {response.action}")
            
            if response.error:
                print(f"❌ Error: {response.error.error}")
                continue
            
            if response.info_data:
                metadata = response.info_data.metadata
                print(f"📊 Image Info: {metadata.width}x{metadata.height} {metadata.format}")
                print(f"📝 Description: {response.info_data.description}")
                print(f"🎨 Dominant colors: {', '.join(response.info_data.dominant_colors[:3])}")
            
            elif response.edit_data:
                if hasattr(response.edit_data, 'edited_image_path'):
                    print(f"✅ {response.edit_data.message}")
                    print(f"💾 Saved to: {response.edit_data.edited_image_path}")
                    
                    if hasattr(response.edit_data, 'edits_applied'):
                        print(f"🔧 Applied: {', '.join(response.edit_data.edits_applied)}")
                    
                    if hasattr(response.edit_data, 'detected_objects'):
                        if response.edit_data.detected_objects:
                            print(f"🎯 Detected objects:")
                            for obj in response.edit_data.detected_objects:
                                print(f"   - {obj.label} (confidence: {obj.confidence:.2f})")
            
            elif response.clarify_data:
                print(f"❓ {response.clarify_data.message}")
                if response.clarify_data.suggested_prompts:
                    print("💡 Try these:")
                    for suggestion in response.clarify_data.suggested_prompts:
                        print(f"   - {suggestion}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
