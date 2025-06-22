#!/usr/bin/env python3
"""
Test script for the local edit agent to verify it's working properly
"""

import os
import sys
import logging
from PIL import Image
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.assistant import ImageEditingAssistant
from logic.router_agent import ActionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_local_edit_agent():
    """Test the local edit agent with a sample request"""
    
    print("🧪 Testing Local Edit Agent Connection...")
    
    # Initialize the assistant
    try:
        assistant = ImageEditingAssistant()
        print("✅ Assistant initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        return False
    
    # Test routing for local edit requests
    test_prompts = [
        "remove the person from the image",
        "delete the car",
        "remove the object in the center",
        "inpaint the background"
    ]
    
    print("\n🔀 Testing Router for Local Edit Actions...")
    for prompt in test_prompts:
        try:
            action = assistant.router.route_request("test_image.jpg", prompt)
            expected = ActionType.LOCAL_EDIT
            status = "✅" if action == expected else "❌"
            print(f"{status} '{prompt}' -> {action} (expected: {expected})")
        except Exception as e:
            print(f"❌ Router error for '{prompt}': {e}")
    
    # Test with a real image if available
    test_image_paths = [
        "test_images/sample.jpg",
        "test_images/sample.png", 
        "example.jpg",
        "example.png"
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if test_image:
        print(f"\n🖼️  Testing with real image: {test_image}")
        try:
            response = assistant.process_request(test_image, "remove the person")
            print(f"✅ Local edit request processed successfully")
            print(f"   Action: {response.action}")
            print(f"   Message: {response.edit_data.message if response.edit_data else 'No edit data'}")
            
            if response.edit_data and hasattr(response.edit_data, 'edited_image_path'):
                if os.path.exists(response.edit_data.edited_image_path):
                    print(f"✅ Edited image saved: {response.edit_data.edited_image_path}")
                else:
                    print(f"❌ Edited image not found: {response.edit_data.edited_image_path}")
                    
        except Exception as e:
            print(f"❌ Local edit processing failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No test image found. Skipping real image test.")
        print("   Place a test image in one of these locations:")
        for path in test_image_paths:
            print(f"   - {path}")
    
    print("\n🎯 Local Edit Agent Test Complete!")
    return True

def create_test_image():
    """Create a simple test image if none exists"""
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    test_path = os.path.join(test_dir, "sample.jpg")
    if not os.path.exists(test_path):
        # Create a simple test image
        img = Image.new('RGB', (400, 300), color='lightblue')
        img.save(test_path)
        print(f"✅ Created test image: {test_path}")
        return test_path
    
    return test_path

if __name__ == "__main__":
    print("🚀 Starting Local Edit Agent Test...\n")
    
    # Create a test image if needed
    create_test_image()
    
    # Run the test
    success = test_local_edit_agent()
    
    if success:
        print("\n✅ All tests completed!")
        print("\n💡 To test in the Gradio UI:")
        print("   1. Run: python gradio_ui.py")
        print("   2. Upload an image")
        print("   3. Try prompts like:")
        print("      - 'remove the person'")
        print("      - 'delete the car'") 
        print("      - 'remove the object in the center'")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
