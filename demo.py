# Do imports here
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
from system import handle_conversation_turn

def chat():
  """Simulates a simple chat interface with conversation history in Google Colab."""
  conversation_history = ""
  while True:
    user_input = input("User: ")
    conversation_history += f"User: {user_input}\n"
    if user_input.lower() == "quit":
      print("Assistant: Goodbye!")
      break

    response_conv_turn = handle_conversation_turn(conversation_history)
    print(response_conv_turn)
    response = response_conv_turn

    #response = response_conv_turn.text
    conversation_history += f"Assistant: {response}\n"
    print(f"Assistant: {response}\n")

  return conversation_history

if __name__ == '__main__':
    chat()