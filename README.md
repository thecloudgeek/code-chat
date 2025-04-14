# Local Code & File Chat with Gemini (Vertex AI)

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Streamlit](https://img.shields.io/badge/Streamlit->=1.41-orange.svg)](https://streamlit.io)
[![Vertex AI](https://img.shields.io/badge/Vertex_AI->=1.71-green.svg)](https://cloud.google.com/vertex-ai)

This Streamlit application provides a chat interface powered by Google's Gemini 2.5 pro model (via Vertex AI) allowing you to interact with the model using context derived from:

1.  **Local Code Directories:** Scans a specified local directory, reads text-based files, and includes their content as context.
2.  **Uploaded Files:** Processes uploaded TXT, PDF, PNG, and JPG files, adding their text content (for TXT/PDF) or image data (for PNG/JPG) to the context for multimodal analysis.

The application combines code, text, and image context to answer your questions.

## Features

*   **Conversational AI:** Chat interface using Streamlit w/ Vertex AI to chat with a local directory containing code.
*   **Local Directory Context:** Process entire codebases or directories by providing a local path.
    *   Configurable ignore lists for directories, file extensions, and specific filenames.
    *   Handles large repositories by skipping files over a size limit (8MB).
    *   Gracefully skips binary files and files with decoding errors.
*   **File Upload Context:** Upload individual files for analysis:
    *   `.txt`: Extracts text content.
    *   `.pdf`: Extracts text content using PyPDF2 (if installed).
    *   `.png`, `.jpg`, `.jpeg`: Includes image data for multimodal analysis with compatible models.
*   **Multimodal Interaction:** Ask questions about code, text documents, and uploaded images within the same chat session.
*   **Configurable Model & Parameters:**
    *   Select the desired Gemini model from the available options via the sidebar.
    *   Adjust the `max_output_tokens` for the model's response length.
*   **Streaming Responses:** Displays the model's response as it's generated.
*   **Session Management:** Maintains chat history and loaded context within a user session.
*   **Docker Support:** Includes a `Dockerfile` for easy containerization and deployment.


## Prerequisites

1.  **Python:** Version 3.12 or later.
2.  **Pip:** Python package installer.
3.  **Google Cloud Project:**
    *   A Google Cloud Platform project with the **Vertex AI API enabled**.
    *   Billing enabled for the project.
4.  **Authentication:** You need to authenticate your local environment or the Docker container to use Google Cloud services. The recommended way for local development is using the `gcloud` CLI:
    ```bash
    gcloud auth application-default login
    ```
    This command stores credentials where the Vertex AI SDK can automatically find them.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires the following environment variables to be set:

*   `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
*   `GCP_LOCATION`: The Google Cloud region where you want to run Vertex AI models (e.g., `us-central1`).

You can set these variables in your shell before running the app:

```bash
# On macOS/Linux
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GCP_LOCATION="your-gcp-location"

# On Windows (Command Prompt)
set GOOGLE_CLOUD_PROJECT=your-gcp-project-id
set GCP_LOCATION=your-gcp-location

# On Windows (PowerShell)
$env:GOOGLE_CLOUD_PROJECT = "your-gcp-project-id"
$env:GCP_LOCATION = "your-gcp-location"
```

Alternatively, you can use a .env file and a library like python-dotenv if you prefer (requires modifying src/config.py).

## Running the App

Once the prerequisites are met, dependencies are installed, and environment variables are configured, run the Streamlit application:

`streamlit run app.py`

The application should open in your default web browser, typically at `http://localhost:8501`.

## Running with Docker

A Dockerfile is provided for containerizing the application.

1. **Build the Docker image:**

    `docker build -t gemini-code-chat .`

2. **Run the Docker container: Make sure to pass the necessary environment variables to the container.**
    ```
    docker run -p 8501:8501 \
    -e GOOGLE_CLOUD_PROJECT="your-gcp-project-id" \
    -e GCP_LOCATION="your-gcp-location" \
    --name gemini-chat-app \
    gemini-code-chat
    ```
   * Replace "your-gcp-project-id" and "your-gcp-location" with your actual GCP details.
   * The container needs access to Google Cloud credentials. If running locally after gcloud auth application-default login, Docker might pick them up automatically depending on your setup. If it doesn't, add `-v "$HOME/.config/gcloud/application_default_credentials.json":/gcp/creds.json:ro` and `-e GOOGLE_APPLICATION_CREDENTIALS=/gcp/creds.json` to the docker command. For deployments (e.g., Cloud Run, GKE), configure service accounts appropriately.

Access the application in your browser at http://localhost:8501.

## Usage

1. **Configure Model (Sidebar):**
   * Select the Gemini model you want to use.
   * Adjust the Max Output Tokens slider if needed.
   * Modify the `Ignore Directories`, `Ignore File Extensions`, and `Ignore Specific Files` lists if processing a local directory.
2. **Provide Context (Main Area):**
   * **Option 1:** Local Directory: Enter the full path to the local directory you want to analyze in the "Input the full path..." text box and click "Process Local Directory". Wait for the confirmation message.
   * **Option 2:** Upload Files: Drag and drop or browse to upload TXT, PDF, PNG, or JPG files. After uploading, click "Process Uploaded Files". Wait for the confirmation message.
   * You can use either or both options. The context from both sources will be combined if processed.
3. **Chat:**
   * Type your question in the chat input box at the bottom of the page and press Enter.
   * Your prompt, along with the processed code/text context and any currently processed images, will be sent to the Gemini model.
   * The model's response will appear in the chat history.

**Note on Images**: When you upload and process images, they are prepared for the model. They will be included with the next chat message you send. Subsequent messages will also include these images unless you clear the context or upload new files.

## How it Works

1. **Initialization:** The Streamlit app starts, initializes session state, and loads the selected Gemini model via src/gemini_utils.py using configured GCP credentials.
2. **Context Processing:**
   * Local Directory: If a path is provided and processed, `src/git_utils.py:combine_repo_files` walks the directory, reads allowed text files, concatenates their content into a single string, respecting ignore lists and size limits.
   * File Upload: If files are uploaded and processed, `app.py:process_uploaded_files` reads the content. Text is extracted from TXT/PDFs. Raw image bytes and MIME types are stored for PNG/JPGs.
   * Processed context (text string from directory, structured file details including text and image data) is stored in `st.session_state`.
3. **Chat Interaction:**
   * User submits a prompt via `st.chat_input`.
   * The application constructs a list of `Content` objects for the Vertex AI API:
      * An initial user message containing the combined text context (from the directory and uploaded text/PDF files).
      * A model acknowledgement message.
      * Previous messages from the chat history (st.session_state.messages).
      * The latest user message, containing:
         * The text prompt entered by the user.
         * Part objects for each currently processed image (Part.from_data(...)).
      * This list of Content objects is passed to src/gemini_utils.py:generate_gemini_content.
3. **API Call & Response:**
   * `generate_gemini_content` calls the `model.generate_content` method of the Vertex AI SDK with the constructed messages, configuration, and `stream=True`.
   * The app iterates through the response stream, yielding text chunks.
   * Streamlit updates the chat UI incrementally (st.empty(), st.markdown(...)).
   * The final assistant response is added to the session state history.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License
Apache 2.0