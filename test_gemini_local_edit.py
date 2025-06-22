#!/usr/bin/env python3
"""
Test script for the Gemini Local Edit Agent
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from logic.gemini_local_edit_agent import GeminiLocalEditAgent

def test_gemini_local_edit():
    """Test the Gemini local edit agent with a sample image"""
    
    # Check if GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        return False
    
    # Initialize the agent
    try:
        agent = GeminiLocalEditAgent()
        print("✓ Gemini Local Edit Agent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return False
    
    # Test with a sample image (you'll need to provide a test image)
    test_images_dir = project_root / "test_images"
    
    # Look for test images
    test_image = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for img_file in test_images_dir.glob(f"*{ext}"):
            test_image = str(img_file)
            break
        if test_image:
            break
    
    if not test_image:
        print("No test images found in test_images directory")
        print("Please add a test image to the test_images directory")
        return False
    
    print(f"Using test image: {test_image}")
    
    # Test prompts
    test_prompts = [
        "Add a red hat to the person in the image",
        "Change the color of the car to blue",
        "Remove the background and make it white",
        "Add sunglasses to the person",
        "Make the sky more dramatic with clouds"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1}: {prompt} ---")
        
        try:
            # Process the local edit request
            result = agent.process_local_edit(test_image, prompt)
            
            print(f"✓ Edit completed")
            print(f"  Original image: {test_image}")
            print(f"  Edited image: {result['edited_image_path']}")
            print(f"  Detected objects: {len(result['detected_objects'])}")
            print(f"  Message: {result['message']}")
            
            # Print detected objects
            for j, obj in enumerate(result['detected_objects']):
                print(f"    Object {j+1}: {obj.label} (confidence: {obj.confidence:.3f})")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            continue
        
        # Only run one test for now to avoid API quota issues
        break
    
    return True

def test_object_detection_only():
    """Test only the object detection functionality"""
    
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        return False
    
    try:
        agent = GeminiLocalEditAgent()
        print("✓ Agent initialized for detection test")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return False
    
    # Find test image
    test_images_dir = project_root / "test_images"
    test_image = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for img_file in test_images_dir.glob(f"*{ext}"):
            test_image = str(img_file)
            break
        if test_image:
            break
    
    if not test_image:
        print("No test images found")
        return False
    
    print(f"Testing object detection with: {test_image}")
    
    # Test detection with different prompts
    detection_prompts = [
        "Find all people in the image",
        "Detect cars and vehicles",
        "Identify objects that can be edited",
        "Find faces in the image"
    ]
    
    for prompt in detection_prompts:
        print(f"\n--- Detection Test: {prompt} ---")
        
        try:
            detected_objects = agent._detect_objects_with_gemini(test_image, prompt)
            print(f"✓ Detected {len(detected_objects)} objects")
            
            for i, obj in enumerate(detected_objects):
                print(f"  {i+1}. {obj.label} at ({obj.x1}, {obj.y1}, {obj.x2}, {obj.y2}) - confidence: {obj.confidence:.3f}")
                
        except Exception as e:
            print(f"✗ Detection failed: {e}")
        
        # Only run one detection test
        break
    
    return True

if __name__ == "__main__":
    print("=== Gemini Local Edit Agent Test ===\n")
    
    # Test object detection first
    print("1. Testing object detection...")
    test_object_detection_only()
    
    print("\n" + "="*50 + "\n")
    
    # Test full local edit functionality
    print("2. Testing full local edit...")
    test_gemini_local_edit()
    
    print("\n=== Test completed ===")
