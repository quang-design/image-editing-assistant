from sentence_transformers import SentenceTransformer
import pandas as pd
import google.generativeai as genai

genai.configure(api_key="*") # change to your api
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

GEMINI = genai.GenerativeModel('gemini-2.5-flash-preview-05-20', safety_settings = safety_settings)
response = GEMINI.generate_content("Hello")

class DummyResponse:
    def __init__(self, text):
        self.text = text

def join_list_into_string(my_list):
    result = ""
    for i, item in enumerate(my_list):
        result += f"{i+1}. {item}\n\n"
    return result.rstrip()


def separate_last_user_query(conversation):
    lines = conversation.splitlines()
    if not lines:
        return None
    user_queries = [line.split(": ", maxsplit=1)[1] for line in lines if line.startswith("User:")]
    return user_queries[-1] if user_queries else None

def choose_method_for_handling_user_query(conversation_text):

    user_last_query = separate_last_user_query(conversation_text)
    prompt = f"""You are an AI assistant tasked with processing a user's prompt in conjunction with an image. Your primary goal is to categorize the user's intent to determine the most appropriate next action or tool to call.
Given the following conversation:
{conversation_text}
User's last query:
{user_last_query}
What do you need to do to answer the user's last query helpfully and faithfully? (Please ask for clarifications only when necessary)
A. **Provide Information**Examples: "What are the dimensions of this image?", "Describe this scene.", "What objects are visible in this picture?", "Tell me the EXIF data."
B. **Request Clarification**Examples: "Make it look better." (How?), "Edit this part." (Which part, and how?), "Can you adjust the image?" (What kind of adjustment?).
C. **Perform Global Image Editing**Examples: "Convert this image to grayscale.", "Apply a warm filter to the whole picture.", "Increase the overall contrast.", "Make the image black and white."
D. **Perform Object-Specific Editing**Examples: "Remove the person in the background.", "Change the color of the blue car to red.", "Blur the face of the individual on the right.", "Inpaint the scratched area on the table."
If the prompt involves multiple steps (e.g., "Identify the cats and make them wear hats"), choose the category that represents the primary or most complex part of the request that needs to be initiated. Internal steps like "Query reformulation" should guide your decision towards one of these four actions.
Note: Please answer using letters (A. or B. or C. or D.)
    """

    response = GEMINI.generate_content(prompt)
    return response



def ask_for_clarification_questions(conversation_text):
    prompt = f"""Given the following conversation
{conversation_text}

Generate a clarification question given the conversation history, with the aim to be as helpful to the user as possible
    """
    response = GEMINI.generate_content(prompt)
    return response

def query_reformulation(conversation_text):
    user_last_query = separate_last_user_query(conversation_text)
    prompt = f"""Given the following conversation
{conversation_text}

User's last query:
{user_last_query}

Please rewrite the last user's query so that the rewritten query can be used to search and satisfy the user's information needs. It is best that the reformulated query is in the form of a question
    """

    response = GEMINI.generate_content(prompt)
    return response


def answer_user_directly(conversation_text):
    prompt = f"""You are an Assistant
Given the following conversation
{conversation_text}

Please give a response in a faithful and helpful manner
Assistant:
    """

    response = GEMINI.generate_content(prompt)
    return response


def check_if_context_is_relevant(query, string_context):
    prompt = f"""Given the following query and context. Is the context has relevant information to answer the query
Query: {query}

Context: {string_context}

Choose:
A. Yes
B. No
Note: Please answer using letters (A. or B.)
"""
    response = GEMINI.generate_content(prompt)
    if "A." in response.text or "A" == response.text: return True
    else: return False


def answer_query_with_context(query, conversation_text, contexts):
    string_context = join_list_into_string(contexts)

    context_relevant = check_if_context_is_relevant(query, string_context)

    if not context_relevant: return DummyResponse(text = "I am very sorry for the inconvenience, I cannot find the right information for your question")

    else:
        print(f"\n--------\nMESSAGE: {string_context}\n--------\n")
        prompt = f"""You are an Assistant
Given the following conversation history, reformulated user's query and context.

Conversation history:
{conversation_text}

Reformulated query:
{query}

Retrieved Context:
{string_context}

Please answer the user based on the information in the context
        """
        response = GEMINI.generate_content(prompt)
        return response


def handle_conversation_turn(conversation_history):
    method_for_answering = choose_method_for_handling_user_query(conversation_history)

    if "A." in method_for_answering.text or "A" == method_for_answering.text:
        print("\n--------\nLOGGING: answer query directly\n--------\n")
        return "ifnormation of image "

    elif "C." in method_for_answering.text or "C" == method_for_answering.text:

        return "image edit"

    elif "B." in method_for_answering.text or "B" == method_for_answering.text:
        print("\n--------\nLOGGING: asking clarification question\n--------\n")
        return ask_for_clarification_questions(conversation_history)
    elif "D." in method_for_answering.text or "D" == method_for_answering.text:

        return "image object edit"
    else:
        return method_for_answering
