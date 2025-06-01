#!/usr/bin/env python3
"""
Test script to verify the integrated image editing assistant works correctly.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from logic.assistant import ImageEditingAssistant

def test_assistant():
    """Test the integrated assistant with sample prompts"""
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key: export GEMINI_API_KEY='your-key-here'")
        return False
    
    # Initialize assistant
    print("🚀 Initializing Image Editing Assistant...")
    assistant = ImageEditingAssistant()
    
    # Test image path (you'll need to provide a real image)
    test_image = "test_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please provide a test image file or update the path in this script")
        return False
    
    print(f"📸 Using test image: {test_image}")
    
    # Test cases
    test_cases = [
        ("Hello", "Should route to ANSWER"),
        ("What's in this image?", "Should route to INFO"),
        ("Make this image brighter", "Should route to GLOBAL_EDIT"),
        ("Remove the person from this image", "Should route to LOCAL_EDIT"),
        ("I'm not sure what I want", "Should route to CLARIFY"),
    ]
    
    print("\n🧪 Running test cases...")
    
    for i, (prompt, expected) in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {prompt} ---")
        print(f"Expected: {expected}")
        
        try:
            response = assistant.process_request(test_image, prompt)
            print(f"✅ Action: {response.action}")
            
            if response.error:
                print(f"⚠️  Error: {response.error.error}")
                if response.error.details:
                    print(f"   Details: {response.error.details}")
            
            if response.info_data:
                print(f"📊 Info: {response.info_data.metadata.width}x{response.info_data.metadata.height} {response.info_data.metadata.format}")
                print(f"   Description: {response.info_data.description[:100]}...")
            
            if response.edit_data:
                if hasattr(response.edit_data, 'edited_image_path'):
                    print(f"🎨 Edit: {response.edit_data.edited_image_path}")
                    print(f"   Message: {response.edit_data.message}")
            
            if response.clarify_data:
                print(f"❓ Clarify: {response.clarify_data.message}")
                
        except Exception as e:
            print(f"❌ Error processing request: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Integration test completed!")
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'google.genai',
        'PIL',
        'cv2',
        'numpy',
        'pydantic'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install google-genai pillow opencv-python numpy pydantic")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Image Editing Assistant Integration Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Run the test
    if test_assistant():
        print("\n🎉 All tests passed! The assistant is ready to use.")
    else:
        print("\n💥 Some tests failed. Please check the configuration.")
        sys.exit(1)
