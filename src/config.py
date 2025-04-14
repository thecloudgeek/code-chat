import os

def get_config():
    """Loads configuration from environment variables."""
    return {
        "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT"),
        "GCP_LOCATION": os.getenv("GCP_LOCATION", default="us-central1"),
    }