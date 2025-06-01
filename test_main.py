import os
from logic.assistant import ImageEditingAssistant

def run_tests():
    """Run demo/test requests for the Image Editing Assistant."""
    assistant = ImageEditingAssistant()
    image_path = "test_images/test.png"
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Using placeholder for demonstration.")
        image_path = "placeholder.jpg"  # You'll need to provide an actual image
    print("=== Image Editing Assistant Demo ===\n")
    test_requests = [
        ("What's in this image?", "Image Analysis"),
        ("Make it brighter and more vibrant", "Global Edit - Brightness & Saturation"),
        ("Remove the person in the background", "Local Edit - Object Removal"),
        ("Add more contrast and make it warmer", "Global Edit - Contrast & Temperature"),
    ]
    for prompt, description in test_requests:
        print(f"\U0001F4CB Test: {description}")
        print(f"\U0001F4AC Prompt: '{prompt}'")
        try:
            result = assistant.process_request(image_path, prompt)
            print(f"\u2705 Result: {result}")
            if "edited_image" in result:
                print(f"\U0001F5BC\uFE0F  Edited image saved: {result['edited_image']}")
        except Exception as e:
            print(f"\u274C Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
