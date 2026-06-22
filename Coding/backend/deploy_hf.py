import os
import sys

def deploy(token, space_name="axiom-backend", dataset_name="axiom-telemetry"):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub is not installed in the current environment.")
        print("Please install it by running: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    # 1. Create the private dataset
    dataset_id = f"{username}/{dataset_name}"
    print(f"Ensuring private dataset repo exists: {dataset_id}")
    try:
        create_repo(
            repo_id=dataset_id,
            repo_type="dataset",
            private=True,
            token=token,
            exist_ok=True
        )
        print(f"✓ Dataset repo {dataset_id} is ready.")
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        sys.exit(1)

    # 2. Create the public Space
    space_id = f"{username}/{space_name}"
    print(f"Ensuring public Space repo exists: {space_id}")
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=token,
            exist_ok=True
        )
        print(f"✓ Space repo {space_id} is ready.")
    except Exception as e:
        print(f"Failed to create Space: {e}")
        sys.exit(1)

    # 3. Upload the backend folder to the Space
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Uploading backend folder '{backend_dir}' to Space '{space_id}'...")
    try:
        api.upload_folder(
            folder_path=backend_dir,
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=[
                "**/__pycache__",
                "**/*.pyc",
                ".env",
                ".venv",
                "app/ml/feedback_data/feedback_log.json",
                "app/ml/feedback_data/quarantine_log.json",
                "app/ml/feedback_data/calibration_metrics.json",
                "app/ml/feedback_data/calibration_history.json",
                "app/ml/feedback_data/confusion_matrix.json",
                "app/ml/feedback_data/video_feedback_db.json",
                "app/ml/feedback_data/video_calibration_db.json",
                "deploy_hf.py"
            ]
        )
        print("✓ Upload complete!")
    except Exception as e:
        print(f"Failed to upload folder: {e}")
        sys.exit(1)

    # 4. Set Space variables
    print("Configuring Space environment variables...")
    try:
        api.add_space_variable(repo_id=space_id, key="AXIOM_MODE", value="host")
        api.add_space_secret(repo_id=space_id, key="AXIOM_HF_TOKEN", value=token)
        api.add_space_variable(repo_id=space_id, key="AXIOM_HF_DATASET_PATH", value=dataset_id)
        api.add_space_variable(repo_id=space_id, key="AXIOM_ALLOW_MODEL_DOWNLOAD", value="true")
        api.add_space_variable(repo_id=space_id, key="AXIOM_DEVICE", value="cpu")
        print("✓ Environment variables configured successfully.")
    except Exception as e:
        print(f"Failed to set Space secrets/variables: {e}")
        print("You can set these manually in the Space settings on huggingface.co.")

    space_url = f"https://huggingface.co/spaces/{space_id}"
    api_url = f"https://{username.lower()}-{space_name.lower()}.hf.space"
    print("\n" + "="*50)
    print("DEPLOYMENT INITIATED SUCCESSFULLY!")
    print(f"Space URL: {space_url}")
    print(f"FastAPI API Base URL: {api_url}")
    print("="*50)

if __name__ == "__main__":
    import os
    import getpass
    
    tok = os.environ.get("HF_TOKEN") or os.environ.get("AXIOM_HF_TOKEN")
    
    # Check if first CLI argument is the HF token (backward compatibility)
    first_arg_is_token = False
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1.startswith("hf_") or len(arg1) >= 30:
            first_arg_is_token = True
            
    if first_arg_is_token:
        tok = sys.argv[1]
        sp = sys.argv[2] if len(sys.argv) > 2 else "axiom-backend"
        ds = sys.argv[3] if len(sys.argv) > 3 else "axiom-telemetry"
    else:
        sp = sys.argv[1] if len(sys.argv) > 1 else "axiom-backend"
        ds = sys.argv[2] if len(sys.argv) > 2 else "axiom-telemetry"
        
    if not tok:
        tok = getpass.getpass("Enter Hugging Face token (leave empty to cancel): ").strip()
        if not tok:
            print("ERROR: Hugging Face token is required for deployment.")
            sys.exit(1)
            
    deploy(tok, sp, ds)
