#gemini_utils.py
from google import genai
from google.genai import types
import streamlit as st
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_genai_client = None

def initialize_google_genai_model(model_name: str, project=None, location=None):
    """Initializes Google Generative AI SDK and returns the client reference."""
    global _genai_client
    try:
        # Initialize the client (only once)
        if _genai_client is None:
            logging.info(f"Attempting to initialize Google Generative AI client with Vertex AI (project: {project}, location: {location}).")
            
            if not project or not location:
                logging.error("Missing project or location for Vertex AI initialization")
                return None
                
            # Initialize with Vertex AI configuration
            _genai_client = genai.Client(
                vertexai=True,
                project=project,
                location=location
            )
            logging.info(f"Google Generative AI client initialized with project {project} in {location}")
        else:
            logging.info("Google Generative AI client already initialized.")
        
        # Return the client instance
        logging.info(f"Using model: {model_name}")
        return _genai_client
    except Exception as e:
        st.error(f"Error initializing Google Generative AI SDK: {e}")
        logging.error(f"Error initializing Google Generative AI SDK: {e}", exc_info=True)
        return None

def generate_gemini_content(client, contents: list[dict], max_tokens: int):
    """Generates content using the provided Google GenAI client."""
    if not client:
        st.error("Client not initialized. Cannot generate content.")
        logging.error("generate_gemini_content called with an uninitialized client.")
        yield None
        return

    # Default model
    model_name = 'gemini-2.5-pro-preview-03-25'

    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]    

    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 0.95,
        seed = 0,
        max_output_tokens = 55819,
        response_modalities = ["TEXT"],
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )],
        tools = tools,
    )

    logging.info(f"Sending contents to Gemini API.")
    logging.info(f"Using model: {model_name}")

    try:
        # Generate content with streaming using client.models approach
        # Pass parameters directly instead of using generation_config dictionary
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config = generate_content_config
        )

        for chunk in response_stream:
            # Check for blocking
            if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                block_reason = chunk.prompt_feedback.block_reason.name
                logging.warning(f"Content generation potentially blocked: {block_reason}")
                st.warning(f"Content generation potentially blocked. Reason: {block_reason}")
                yield None
                break
            
            # Extract text from the chunk
            try:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
            except AttributeError:
                logging.warning(f"Chunk missing expected attributes: {chunk}")
                continue

    except StopIteration:
        logging.info("Content generation stream finished.")
    except Exception as e:
        st.error(f"Error during Gemini API call or streaming: {e}")
        logging.error(f"Error during Gemini API call or streaming: {e}", exc_info=True)
        yield None

def create_content(role: str, text: str) -> dict:
    """Creates a content dictionary for the google-genai SDK."""
    valid_roles = ['user', 'model']
    if role not in valid_roles:
        logging.warning(f"Invalid role '{role}' provided. Defaulting to 'user'.")
        role = 'user'
    
    if role == 'user':
        return types.UserContent(parts=[types.Part.from_text(text=text)])
    else:
        return types.ModelContent(parts=[types.Part.from_text(text=text)])