import urllib.request
import zipfile
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def download_and_extract(url, extract_to="data/raw"):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "heart_disease.zip")
    logging.info(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    logging.info("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(f"Data extracted to {extract_to}")
    os.remove(zip_path)

if __name__ == "__main__":
    DATA_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    download_and_extract(DATA_URL)
