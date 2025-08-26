import vertexai
from vertexai.generative_models import GenerativeModel
import os
from google.oauth2 import service_account
from google.auth import default

# Authentication setup
def setup_authentication():
    """Set up authentication for Vertex AI"""
    
    # Option 1: Try to use service account key file if available
    service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if service_account_path and os.path.exists(service_account_path):
        print("Using service account key file for authentication")
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        return credentials
    
    # Option 2: Try default credentials (will work if VM has proper scopes)
    try:
        credentials, _ = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        print("Using default credentials")
        return credentials
    except Exception as e:
        print(f"Default credentials failed: {e}")
        return None

# Get project ID from environment or metadata server
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not project_id:
    # Try to get from metadata server if not in environment
    try:
        import requests
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
        headers = {"Metadata-Flavor": "Google"}
        response = requests.get(metadata_url, headers=headers, timeout=5)
        if response.status_code == 200:
            project_id = response.text
    except Exception:
        print("Could not retrieve project ID from metadata server")
        project_id = "your-project-id"  # Replace with your actual project ID

# Set the location (region) for Vertex AI
location = "us-central1"  # Change this to your preferred region if needed

try:
    # Set up authentication
    credentials = setup_authentication()
    
    # Initialize Vertex AI with explicit credentials
    if credentials:
        vertexai.init(project=project_id, location=location, credentials=credentials)
    else:
        # Fallback to default initialization (might fail if scopes are insufficient)
        vertexai.init(project=project_id, location=location)
    
    # Configure the model
    model = GenerativeModel('gemini-1.5-flash-001')

    # Make the API call
    prompt = "Explain how a Compute Engine VM works in simple terms."
    response = model.generate_content(prompt)
    print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting suggestions:")
    print("1. SCOPE ISSUE: Your VM doesn't have the required OAuth scopes.")
    print("   To fix this, you need to either:")
    print("   a) Stop the VM and modify it to include 'https://www.googleapis.com/auth/cloud-platform' scope")
    print("   b) Or create a service account key and set GOOGLE_APPLICATION_CREDENTIALS")
    print("")
    print("2. To create a service account key:")
    print("   - Go to IAM & Admin > Service Accounts in GCP Console")
    print("   - Create or select a service account with 'Vertex AI User' role")
    print("   - Create a JSON key and download it")
    print("   - Set: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
    print("")
    print("3. Alternative: Recreate the VM with proper scopes:")
    print("   - When creating VM, set 'Access scopes' to 'Allow full access to all Cloud APIs'")
    print("   - Or manually add 'https://www.googleapis.com/auth/cloud-platform' scope")
