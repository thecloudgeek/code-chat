# app.py
import streamlit as st
# Change the import from gemini_utils
from src.gemini_utils import initialize_vertex_model, generate_gemini_content # Removed create_content helper as we build directly
from src.git_utils import combine_repo_files, DEFAULT_IGNORE_DIRS, DEFAULT_IGNORE_EXTS, DEFAULT_IGNORE_FILES
from src.config import get_config
from vertexai.generative_models import Part, Content # Keep these
import uuid
from streamlit_tags import st_tags
import os
from io import BytesIO
import logging

# --- PDF processing import (remains the same) ---
try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    # No warning here, handled in processing function if needed
# --- End Add ---

# --- MODIFIED function to process uploaded files ---
def process_uploaded_files(uploaded_files):
    """
    Reads content from uploaded files (txt, pdf, png, jpg).
    Returns a list of dictionaries, each containing file details.
    For images, 'data' contains bytes. For text/pdf, 'content' contains text.
    """
    processed_files_details = []
    if not uploaded_files:
        return []

    for uploaded_file in uploaded_files:
        file_details = {
            "name": uploaded_file.name,
            "type": uploaded_file.type,
            "content": None, # For text-based content
            "data": None,    # For raw byte data (images)
            "error": None
        }
        bytes_data = uploaded_file.getvalue()

        try:
            if file_details["type"] == "text/plain":
                try:
                    file_details["content"] = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    file_details["content"] = "[Error: Could not decode file as UTF-8]"
                    file_details["error"] = "Decoding Error"
            elif file_details["type"] == "application/pdf":
                if PYPDF2_AVAILABLE:
                    pdf_text = ""
                    try:
                        reader = PdfReader(BytesIO(bytes_data))
                        for i, page in enumerate(reader.pages):
                            pdf_text += f"--- Page {i+1} ---\n"
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text += page_text + "\n"
                            else:
                                pdf_text += "[No text extracted from this page]\n"
                        file_details["content"] = pdf_text
                    except Exception as pdf_e:
                        file_details["content"] = f"[Error reading PDF: {pdf_e}]\n"
                        file_details["error"] = "PDF Read Error"
                else:
                    file_details["content"] = "[PDF processing skipped: PyPDF2 not installed]\n"
                    file_details["error"] = "PyPDF2 Missing"
            elif file_details["type"] in ["image/png", "image/jpeg"]:
                # Store the raw bytes and mime type for image files
                file_details["data"] = bytes_data
                # No text 'content' for raw images
            else:
                 file_details["content"] = f"[Unsupported file type: {file_details['type']}]\n"
                 file_details["error"] = "Unsupported Type"

        except Exception as e:
            file_details["content"] = f"[Error processing file {file_details['name']}: {e}]\n"
            file_details["error"] = f"Processing Error: {e}"

        processed_files_details.append(file_details)

    return processed_files_details
# --- End MODIFIED function ---


def main():
    st.set_page_config(layout="wide")
    st.title("Local Code Dir & File Chat")

    # --- Session State Initialization (UPDATED) ---
    if "messages" not in st.session_state:
         st.session_state.messages = []
    if "repo_content" not in st.session_state:
         st.session_state.repo_content = None
    # Use a new key for structured file details
    if "processed_files_details" not in st.session_state:
         st.session_state.processed_files_details = []
    if 'session_id' not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "combined_ignore_dirs" not in st.session_state:
        st.session_state["combined_ignore_dirs"] = DEFAULT_IGNORE_DIRS[:]
    if "combined_ignore_exts" not in st.session_state:
        st.session_state["combined_ignore_exts"] = DEFAULT_IGNORE_EXTS[:]
    if "combined_ignore_files" not in st.session_state:
        st.session_state["combined_ignore_files"] = DEFAULT_IGNORE_FILES[:]
    if "gemini_model_instance" not in st.session_state:
        st.session_state["gemini_model_instance"] = None
    # --- End Session State ---

    # --- Sidebar (remains the same) ---
    with st.sidebar:
        st.header("Model Configuration")
        model_options = ["gemini-2.5-pro-preview-03-25"] # Ensure you select a vision-capable model like 1.5 Pro/Flash or 1.0 Pro Vision
        st.session_state["gemini_model"] = st.selectbox(
             "Select a model", # Update help text
             model_options,
             index = model_options.index(st.session_state.get("gemini_model", "gemini-2.5-pro-preview-03-25")) # Default to a vision capable one
         )

        max_tokens_value = st.slider(
            "Max Output Tokens",
            min_value=1024,
            max_value=65536,
            value=st.session_state.get("max_output_tokens", 8192),
            step=1024,
            help="Controls the maximum number of tokens the model can generate."
        )
        st.session_state["max_output_tokens"] = max_tokens_value

        st.header("Repo Processing Configuration")
        # st_tags widgets remain the same...
        st.session_state.combined_ignore_dirs = st_tags(label='Ignore Directories:', text='Press enter...', value=st.session_state.combined_ignore_dirs, suggestions=DEFAULT_IGNORE_DIRS, maxtags=-1, key='tags_ignore_dirs')
        st.session_state.combined_ignore_exts = st_tags(label='Ignore File Extensions:', text='Press enter...', value=st.session_state.combined_ignore_exts, suggestions=DEFAULT_IGNORE_EXTS, maxtags=-1, key='tags_ignore_exts')
        st.session_state.combined_ignore_files = st_tags(label='Ignore Specific Files:', text='Press enter...', value=st.session_state.combined_ignore_files, suggestions=[], maxtags=-1, key='tags_ignore_files')


    # --- Model Loading (remains largely the same, but ensure vision model selected) ---
    project = get_config().get("GOOGLE_CLOUD_PROJECT")
    location = get_config().get("GCP_LOCATION")
    selected_model_name = st.session_state["gemini_model"]

    # Check if model needs loading/reloading logic (remains the same)
    current_model_obj = st.session_state.get("gemini_model_instance")
    model_needs_update = False
    if current_model_obj is None:
        model_needs_update = True
    else:
        # ... (existing logic to check if model name changed) ...
         try:
             if hasattr(current_model_obj, '_model_name'):
                 loaded_model_name = current_model_obj._model_name.split('/')[-1]
                 if loaded_model_name != selected_model_name: model_needs_update = True
             elif hasattr(current_model_obj, 'model_name'):
                 loaded_model_name = current_model_obj.model_name.split('/')[-1]
                 if loaded_model_name != selected_model_name: model_needs_update = True
             else:
                 prev_selected_model = st.session_state.get("_prev_gemini_model", None)
                 if selected_model_name != prev_selected_model: model_needs_update = True
         except Exception:
             st.warning("Could not reliably determine loaded model name. Re-initializing on selection change.")
             model_needs_update = True


    if model_needs_update and project and location and selected_model_name:
        with st.spinner(f"Loading model {selected_model_name}..."):
            st.session_state["gemini_model_instance"] = initialize_vertex_model(
                project=project, location=location, model_name=selected_model_name
            )
            st.session_state["_prev_gemini_model"] = selected_model_name
            if st.session_state["gemini_model_instance"]:
                st.success(f"Model {selected_model_name} loaded.")
            else:
                st.error("Failed to load model. Check logs.")
    elif not (project and location):
        st.error("Google Cloud Project ID and Location must be configured.")


    # --- Input Section: Directory and File Upload (UPDATED logic) ---
    st.header("Input Context")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Option 1: Process Local Directory")
        # Directory processing logic remains the same...
        local_repo_path_input = st.text_input("Input the full path to a local directory containing the code:", key="local_dir_input")
        if local_repo_path_input:
            if st.button("Process Local Directory", key="process_dir_button"):
                # ... (spinner, validation, call combine_repo_files) ...
                 with st.spinner("Processing local directory..."):
                     if not os.path.isdir(local_repo_path_input):
                         st.error(f"Invalid path: '{local_repo_path_input}' is not a valid directory.")
                         st.session_state.repo_content = None
                     else:
                         try:
                             repo_files, token_count = combine_repo_files(
                                 local_repo_path_input,
                                 ignore_dirs=st.session_state.combined_ignore_dirs,
                                 ignore_exts=st.session_state.combined_ignore_exts,
                                 ignore_files=st.session_state.combined_ignore_files
                             )
                             st.session_state.repo_content = repo_files
                             st.success(f"Local directory processed! Tokens (approx): {token_count}")
                         except Exception as e:
                             st.error(f"Error processing directory files: {e}")
                             st.session_state.repo_content = None

        if st.session_state.get("repo_content") is not None:
             if st.session_state.repo_content: st.info("Local directory content loaded.")

    with col2:
        st.subheader("Option 2: Upload Files")
        uploaded_files = st.file_uploader(
            "Upload TXT, PDF, PNG, or JPG files:",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'png', 'jpg', 'jpeg'],
            key="file_uploader"
        )

        if uploaded_files:
            # Process button logic
            if st.button("Process Uploaded Files", key="process_files_button"):
                 with st.spinner("Processing uploaded files..."):
                     try:
                         # Process and store structured details
                         st.session_state.processed_files_details = process_uploaded_files(uploaded_files)
                         st.success(f"{len(uploaded_files)} file(s) processed and ready for context.")
                         # Optional: Clear uploader widget state if needed, but be careful with Streamlit's execution model
                         # st.rerun() might be needed or careful state management
                     except Exception as e:
                         st.error(f"Error processing uploaded files: {e}")
                         st.session_state.processed_files_details = [] # Clear on error

        # Display status for file uploads based on the new state key
        if st.session_state.get("processed_files_details"):
             num_files = len(st.session_state["processed_files_details"])
             num_images = sum(1 for f in st.session_state["processed_files_details"] if f.get("data"))
             num_texts = sum(1 for f in st.session_state["processed_files_details"] if f.get("content"))
             st.info(f"{num_files} file(s) processed: {num_texts} text/pdf, {num_images} image(s).")
        elif uploaded_files and not st.session_state.get("processed_files_details"):
             # If files were uploaded but button not pressed / processing failed silently
             st.warning("Files uploaded, press 'Process Uploaded Files' to add them to context.")


    st.divider()
    st.header("Chat Interface")

    # Display chat history (remains the same)
    for message in st.session_state.messages:
         with st.chat_message(message["role"]):
             # Render images stored in messages if needed (more complex UI)
             # For now, just display markdown content
             st.markdown(message["content"])

    # Handle user input (UPDATED message preparation)
    if prompt := st.chat_input("Ask a question about the code or uploaded files"):
        # Add user's text prompt to session state first
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
             st.markdown(prompt)

        # --- Prepare the messages for Gemini (UPDATED with multimodal structure) ---
        vertex_messages = []
        context_parts_text = [] # Accumulate only text context for the initial message

        # 1. Add directory context (if available) - as text
        if st.session_state.get("repo_content"):
            context_parts_text.append(
                "Context from the local code directory:\n"
                "---------------------------------------------------------\n"
                f"{st.session_state.repo_content}"
                "\n---------------------------------------------------------"
            )

        # 2. Add TEXT context from uploaded files (if available)
        uploaded_text_content = ""
        if st.session_state.get("processed_files_details"):
            for file_detail in st.session_state["processed_files_details"]:
                 if file_detail.get("content"): # Check if text content exists
                    uploaded_text_content += f"\n--- Uploaded File: {file_detail['name']} ({file_detail['type']}) ---\n"
                    uploaded_text_content += file_detail['content']
                    uploaded_text_content += "\n----------------------------------------------------------\n"

            if uploaded_text_content:
                context_parts_text.append(
                    "Context from uploaded files (Text Content Only):\n"
                    "---------------------------------------------------------\n"
                    f"{uploaded_text_content}"
                    "\n---------------------------------------------------------"
                )

        # 3. Create the initial context message (if any text context exists)
        if context_parts_text:
            full_text_context = "\n\n".join(context_parts_text)
            context_message_text = (
                f"{full_text_context}\n\n"
                "Based on the above context (code directory and text from uploaded files), and potentially uploaded images provided with the prompt, answer the following question:"
            )
            # Use Content/Part directly
            vertex_messages.append(Content(role="user", parts=[Part.from_text(text=context_message_text)]))
            # Model acknowledgement
            vertex_messages.append(Content(role="model", parts=[Part.from_text(text="Okay, I have loaded the provided context (code and text files). Ask your question, including any images you want me to analyze with it.")]))

        # 4. Add chat history (converting roles)
        # Iterate up to the *second to last* message (exclude the current user prompt added above)
        for m in st.session_state.messages[:-1]:
            vertex_role = "user" if m["role"] == "user" else "model"
            # Simple text history for now. More complex state needed if history involves images.
            vertex_messages.append(Content(role=vertex_role, parts=[Part.from_text(text=m["content"])]))

        # 5. Create the LATEST User Prompt Content (potentially multimodal)
        latest_user_parts = [Part.from_text(text=prompt)] # Start with the text prompt

        # Add image data parts from processed files to the *current* prompt
        images_added_to_prompt = []
        if st.session_state.get("processed_files_details"):
            for file_detail in st.session_state["processed_files_details"]:
                 if file_detail.get("data") and file_detail.get("type") in ["image/png", "image/jpeg"]:
                     try:
                         latest_user_parts.append(Part.from_data(data=file_detail["data"], mime_type=file_detail["type"]))
                         images_added_to_prompt.append(file_detail['name'])
                     except Exception as img_e:
                          st.error(f"Failed to prepare image '{file_detail['name']}' for model: {img_e}")
                          logging.error(f"Error creating Part.from_data for {file_detail['name']}: {img_e}", exc_info=True)

        if images_added_to_prompt:
             logging.info(f"Added {len(images_added_to_prompt)} image(s) to the current prompt: {', '.join(images_added_to_prompt)}")
             # Optionally notify user in UI, though it might be verbose
             # st.info(f"Sending images: {', '.join(images_added_to_prompt)} with your prompt.")

        # Add the final user Content object (text + images)
        vertex_messages.append(Content(role="user", parts=latest_user_parts))
        # --- End Message Preparation ---


        # --- Generate and display assistant response (logic remains similar) ---
        with st.chat_message("assistant"):
             model_instance = st.session_state.get("gemini_model_instance")
             if not model_instance:
                  st.error("Model is not initialized.")
                  st.stop()

             # Check if there's *any* context (code OR files) before first question
             has_repo_context = bool(st.session_state.get("repo_content"))
             has_file_context = bool(st.session_state.get("processed_files_details"))
             # Adjust check: allow question if any context exists or if it's not the very first interaction
             if not has_repo_context and not has_file_context and len(st.session_state.messages) <= 1:
                  st.warning("Please process a local directory or upload files first to provide context.")
                  # Remove the user message we optimistically added
                  st.session_state.messages.pop()
                  st.stop()

             try:
                response_stream = generate_gemini_content(
                     model_instance,
                     vertex_messages, # Pass the potentially multimodal messages
                     max_tokens=st.session_state["max_output_tokens"]
                 )
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    if chunk:
                         full_response += chunk
                         response_placeholder.markdown(full_response + "▌")
                    elif chunk is None:
                         st.error("An error occurred during content generation. Check logs.")
                         logging.error("Received None chunk, signaling generation error.")
                         break
                response_placeholder.markdown(full_response)
                if full_response:
                     # Add response to session state
                     st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                     # If generation yielded nothing or only None, maybe remove the user prompt?
                     # Or add a specific assistant message like "[No response generated]"
                     # For now, we just don't add an assistant message if full_response is empty
                     pass


             except Exception as e:
                 st.error(f"Failed to generate response: {e}")
                 logging.error(f"Exception during generate_gemini_content call: {e}", exc_info=True)
                 # Remove the user message if generation failed catastrophically right away


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()