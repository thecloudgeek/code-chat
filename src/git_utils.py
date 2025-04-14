import os
import vertexai
from vertexai.generative_models import GenerativeModel

# Define default ignore lists as constants (remains the same)
DEFAULT_IGNORE_DIRS = ['.github']
DEFAULT_IGNORE_EXTS = ['.md']
DEFAULT_IGNORE_FILES = []

def count_tks(text_string, model_name: str = "gemini-pro"):
    """Counts tokens in a string using a Gemini model on Vertex AI."""
    vertexai.init()
    try:
        model = GenerativeModel("gemini-2.0-flash") # Or another suitable model
        response = model.count_tokens(text_string)
        return response.total_tokens
    except Exception as e:
        print(f"Error counting tokens: {e}. Falling back to character count / 4 (approx).")
        return len(text_string) // 4


# --- combine_repo_files remains largely the same, but REMOVE cleanup ---
def combine_repo_files(repo_path, ignore_dirs, ignore_exts, ignore_files):
    """
    Combines repository files from a given local path into a single text string,
    honoring ignore lists.
    """
    output_string = ""
    total_tokens = 0
    max_file_size = 1024 * 1024 * 8 # 8 MB limit

    # --- IMPORTANT: Check if the repo_path is valid before walking ---
    if not os.path.isdir(repo_path):
        print(f"Error: Provided path '{repo_path}' is not a valid directory.")
        return "", 0 # Return empty string and zero tokens

    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            if file in ignore_files:
                continue
            if any(file.endswith(ext) for ext in ignore_exts):
                continue

            file_path = os.path.join(root, file)
            if os.path.islink(file_path):
                print(f"filename: {os.path.relpath(file_path, repo_path)} Skipped (symbolic link)")
                continue
            try:
                file_size = os.path.getsize(file_path)
                if file_size > max_file_size:
                     print(f"filename: {os.path.relpath(file_path, repo_path)} Skipped due to file size > 8MB \n")
                     continue
                try:
                    with open(file_path, "r", encoding="utf-8") as file_content:
                        content = file_content.read()
                        relative_path = os.path.relpath(file_path, repo_path) # Use relative path
                        output_string += "\n--------------------------------------------------------- FILE START -------------------------------------------------------------------------------------\n\n"
                        output_string += f"filename: {relative_path}\n" # Use relative path
                        output_string += f"file_content:\n{content}\n"
                        output_string += "\n---------------------------------------------------------- FILE END --------------------------------------------------------------------------------------\n\n"
                except UnicodeDecodeError:
                    print(f"filename: {os.path.relpath(file_path, repo_path)} Skipped due to decoding error")
                except OSError as e_read: # Catch file read errors (e.g., permission denied)
                    print(f"filename: {os.path.relpath(file_path, repo_path)} Skipped due to read error: {e_read}")
            except OSError as e_stat: # Catch errors during os.path.getsize or os.path.islink (e.g., file disappears)
               print(f"filename: {os.path.relpath(file_path, repo_path)} Skipped due to filesystem error: {e_stat}")
            except Exception as e: # General catch-all
               print(f"filename: {os.path.relpath(file_path, repo_path)} Skipped due to unexpected error: {e}")

    # Count tokens *after* combining all content
    if output_string:
        total_tokens = count_tks(output_string)
    else:
        total_tokens = 0

    return output_string, total_tokens