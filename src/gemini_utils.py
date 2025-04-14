import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Content,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
)
import streamlit as st
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Store initialized state globally within the module to avoid re-init
_vertexai_initialized = False

def initialize_vertex_model(project: str, location: str, model_name: str):
    """Initializes Vertex AI and returns a GenerativeModel instance."""
    global _vertexai_initialized
    try:
        # Initialize Vertex AI only once per session/run if needed
        if not _vertexai_initialized:
             vertexai.init(project=project, location=location)
             _vertexai_initialized = True
             logging.info(f"Vertex AI initialized for project {project} in {location}")
        else:
             logging.info(f"Vertex AI already initialized (Project: {project}, Location: {location})")

        # Load the specific model
        model = GenerativeModel(model_name=model_name)
        logging.info(f"GenerativeModel loaded: {model_name}")
        return model
    except Exception as e:
        st.error(f"Error initializing Vertex AI or loading model: {e}")
        logging.error(f"Error initializing Vertex AI or loading model: {e}", exc_info=True)
        # Reset initialization flag on error if needed, depending on desired retry behavior
        # _vertexai_initialized = False
        return None

def generate_gemini_content(model: GenerativeModel, contents: list[Content], max_tokens: int):
    """Generates content using the provided Vertex AI GenerativeModel instance."""
    if not model:
        st.error("Model not initialized. Cannot generate content.")
        logging.error("generate_gemini_content called with an uninitialized model.")
        yield None # Signal error
        return # Use return instead of yield None after yielding to stop generation

    # BLOCK_NONE might be too permissive, consider BLOCK_MEDIUM_AND_ABOVE or BLOCK_LOW_AND_ABOVE
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Define tools using Vertex AI SDK types if needed. Example (if you add a search tool):
    # from vertexai.generative_models import Tool, grounding
    # search_tool = Tool(google_search_retrieval=grounding.GoogleSearchRetrieval(disable_attribution=False))
    # tools = [search_tool]
    tools = [] # Keep as empty list if no tools are actively used

    # Only include parameters valid for vertexai.generative_models.GenerationConfig
    generation_config = GenerationConfig(
        temperature = 1.0, # Corrected default based on docs often being 1.0 for creative tasks
        top_p = 0.95,
        max_output_tokens = max_tokens,
    )

    try:
        # Use generate_content with stream=True
        # Pass safety_settings and tools directly here
        response_stream = model.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools, # Pass the correctly defined tools list (even if empty)
            stream=True
        )

        for chunk in response_stream:
            # Access text content correctly for Vertex AI stream chunks
            # Check if the chunk has candidates and the first candidate has content with parts
            try:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    part_text = chunk.candidates[0].content.parts[0].text
                    if part_text:
                       yield part_text
                # Handle potential errors or empty chunks gracefully
                elif chunk.prompt_feedback:
                     # Log or handle feedback if necessary (e.g., block reason)
                     logging.warning(f"Content generation potentially blocked: {chunk.prompt_feedback}")
                     st.warning(f"Content generation potentially blocked. Reason: {chunk.prompt_feedback.block_reason}")
                     # Decide if you want to yield an error signal or just stop
                     yield None # Signal potential issue
                     break # Stop processing more chunks for this response

            except StopIteration:
                 # Stream finished normally
                 break
            except Exception as chunk_e:
                 logging.error(f"Error processing chunk: {chunk_e} - Chunk Data: {chunk}", exc_info=True)
                 yield None # Signal error
                 break # Stop processing

        # Handle potential finish reasons or errors if needed (check Vertex SDK docs for details)
        # For example, check response_stream.candidates[0].finish_reason


    except Exception as e:
        st.error(f"Error calling generate_content: {e}")
        logging.error(f"Error calling generate_content: {e}", exc_info=True)
        yield None # Signal error


# Helper function to create Content objects easily (optional but useful)
def create_content(role: str, text: str) -> Content:
    """Creates a Vertex AI Content object."""
    return Content(role=role, parts=[Part.from_text(text=text)])