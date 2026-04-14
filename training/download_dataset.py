import kagglehub
import os
import shutil

def download_and_prepare():
    print("Downloading PlantVillage dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print("Path to dataset files:", path)

    # Define our local training data directory
    target_dir = "./training/data/plantvillage"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}")

    # Note: In a real environment, we would move/symlink files here
    # For this simulation, we'll just document the path
    with open("./training/dataset_path.txt", "w") as f:
        f.write(path)
    
    print(f"Dataset path saved to ./training/dataset_path.txt")
    print("Next step: Run 'python training/train_classifier.py' to begin retraining.")

if __name__ == "__main__":
    download_and_prepare()
