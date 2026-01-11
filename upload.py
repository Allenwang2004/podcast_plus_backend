from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

def upload_lora_model():

    model_path = "./lora-llama-dialog"
    repo_name = "coconut19/llama-dialog-lora"
    
    api = HfApi()
    
    try:
        create_repo(
            repo_id=repo_name,
            exist_ok=True,
            private=False,
            repo_type="model"
        )
        print(f"Repository {repo_name} created/verified")
    except Exception as e:
        print(f"Repository creation error: {e}")
    
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model"
        )
        print(f"Model uploaded successfully to https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == "__main__":
    upload_lora_model()