import requests
import zipfile
from pathlib import Path

# Function to download a zip file from a URL
def download_zip(url, save_path):
    """This downloads zip files from the given URL, 
    unzip them and saves it to the specified path.
    """
    if not save_path.is_file():  # Only download if the file doesn't already exist
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded and saved at {save_path}.")
    else:
        print(f"{save_path} already exists. Skipping download.")

# Function to unzip a file to a specified directory
def unzip_file(zip_path, extract_to):
    """Unzips a zip file to the specified directory."""
    if not any(extract_to.iterdir()):  # Check if the directory is empty
        print(f"Unzipping {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzipped to {extract_to}.")
    else:
        print(f"{extract_to} is not empty. Skipping unzip.")

# Set up paths for the data folder, train data, and test data
data_path = Path("data/")
train_image_path = data_path / "train_data"
test_image_path = data_path / "test_data"

# Paths to save downloaded zip files
train_zip_path = data_path / "train_data.zip"
test_zip_path = data_path / "test_data.zip"

# Create directories if they don't exist
train_image_path.mkdir(parents=True, exist_ok=True)
test_image_path.mkdir(parents=True, exist_ok=True)

# URLs for train and test zip files
train_url = "https://github.com/glorious0119/Project-Oxford_IIIT_Pet_TorchVision/raw/master/experimentation_data/train_data.zip"
test_url = "https://github.com/glorious0119/Project-Oxford_IIIT_Pet_TorchVision/raw/master/experimentation_data/test_data.zip"

# Download train and test data using the function
download_zip(train_url, train_zip_path)
download_zip(test_url, test_zip_path)

# Unzip the downloaded files using the function
unzip_file(train_zip_path, train_image_path)
unzip_file(test_zip_path, test_image_path)

# Delete the zip files using Python instead of shell commands
if train_zip_path.is_file():
    train_zip_path.unlink()
if test_zip_path.is_file():
    test_zip_path.unlink()
