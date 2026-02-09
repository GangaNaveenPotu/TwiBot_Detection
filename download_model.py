from huggingface_hub import snapshot_download
import os

def download_model():
    model_name = 'bert-base-uncased'
    local_dir = './bert-base-uncased'
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    print(f"Downloading model and tokenizer to {local_dir}")
    snapshot_download(repo_id=model_name, local_dir=local_dir)
    print("Download complete.")

if __name__ == '__main__':
    download_model()