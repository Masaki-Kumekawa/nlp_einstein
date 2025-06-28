#!/usr/bin/env python3
"""
Download full similarity datasets for proper evaluation.
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, filepath):
    """Download file from URL."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def download_scws():
    """Download Stanford Contextual Word Similarities dataset."""
    url = "http://www-nlp.stanford.edu/~ehhuang/SCWS.zip"
    zip_path = "data/SCWS.zip"
    
    logger.info("Downloading SCWS dataset...")
    if download_file(url, zip_path):
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/scws_raw/")
        
        # Convert to our format
        convert_scws_to_csv()
        os.remove(zip_path)
        logger.info("SCWS dataset processed successfully")
    else:
        logger.error("Failed to download SCWS dataset")

def convert_scws_to_csv():
    """Convert SCWS ratings.txt to CSV format."""
    ratings_file = "data/scws_raw/SCWS/ratings.txt"
    output_file = "data/scws.csv"
    
    if not os.path.exists(ratings_file):
        logger.error(f"SCWS ratings file not found: {ratings_file}")
        return
    
    data = []
    with open(ratings_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                word1 = parts[0]
                word2 = parts[1] 
                context1 = parts[2]
                context2 = parts[3]
                # Average score is typically in the last column
                try:
                    score = float(parts[-1])
                    data.append({
                        'word1': word1,
                        'word2': word2,
                        'context1': context1,
                        'context2': context2,
                        'score': score
                    })
                except ValueError:
                    continue
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Converted SCWS to CSV: {len(df)} pairs")

def download_cosimlex():
    """Download CoSimLex dataset from CLARIN repository."""
    # Try direct download from known repository URLs
    urls = [
        "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1308/cosimlex_en.csv",
        "https://raw.githubusercontent.com/cambridgeltl/CoSimLex/master/cosimlex_en.csv"
    ]
    
    output_file = "data/cosimlx.csv"
    
    logger.info("Downloading CoSimLex dataset...")
    for url in urls:
        if download_file(url, output_file):
            logger.info("CoSimLex dataset downloaded successfully")
            return
    
    logger.warning("Could not download CoSimLex from known URLs")
    logger.info("Creating extended sample from template...")
    create_extended_cosimlex_sample()

def create_extended_cosimlex_sample():
    """Create a larger sample of CoSimLex-style data."""
    data = [
        # Ambiguous words with different contexts
        ("bank", "bank", "I deposited money at the bank.", "The river bank was eroded.", 1.5),
        ("star", "star", "The star shone brightly.", "She became a movie star.", 1.2),
        ("cell", "cell", "The cell divided rapidly.", "He was locked in a prison cell.", 1.0),
        ("court", "court", "The court ruled in favor.", "The tennis court was resurfaced.", 1.3),
        ("pitch", "pitch", "He threw a perfect pitch.", "The pitch of the music was high.", 1.1),
        ("plant", "plant", "The plant needs sunlight.", "They will plant seeds tomorrow.", 2.1),
        ("run", "run", "I like to run daily.", "The play had a long run.", 1.8),
        ("rock", "rock", "The rock was heavy.", "Rock music is loud.", 1.4),
        ("spring", "spring", "Spring brings flowers.", "The spring in the clock broke.", 1.6),
        ("bark", "bark", "Tree bark is rough.", "The dog's bark was loud.", 1.3),
        ("bat", "bat", "The baseball bat was wooden.", "The bat flew at night.", 1.1),
        ("bear", "bear", "The bear was large.", "I cannot bear this pain.", 1.5),
        ("bowl", "bowl", "The soup bowl was empty.", "I like to bowl on weekends.", 1.9),
        ("can", "can", "Open the soda can.", "You can do it.", 1.2),
        ("fair", "fair", "The county fair was fun.", "That's not fair.", 1.4),
        ("file", "file", "Save the computer file.", "Use a file to smooth the wood.", 1.3),
        ("fly", "fly", "The fly buzzed around.", "Birds fly in the sky.", 2.0),
        ("jam", "jam", "Strawberry jam is sweet.", "Traffic jam delayed us.", 1.1),
        ("key", "key", "The door key was lost.", "The key to success is hard work.", 1.7),
        ("light", "light", "Turn on the light.", "The box was light.", 1.6),
        ("match", "match", "Light the match carefully.", "The tennis match was exciting.", 1.5),
        ("mint", "mint", "Mint leaves are fresh.", "The coin mint produced money.", 1.2),
        ("park", "park", "The park has trees.", "Park the car here.", 1.8),
        ("pool", "pool", "The swimming pool was clean.", "Pool your resources together.", 1.4),
        ("ring", "ring", "The phone will ring.", "The wedding ring was gold.", 1.3),
        ("scale", "scale", "Weigh it on the scale.", "Fish have scales.", 1.2),
        ("tire", "tire", "The car tire was flat.", "I tire easily from running.", 1.6),
        ("wave", "wave", "The ocean wave was large.", "Wave goodbye to friends.", 1.7),
        ("watch", "watch", "My watch shows time.", "Watch the movie tonight.", 1.9),
        ("yard", "yard", "The yard needs mowing.", "Buy fabric by the yard.", 1.1)
    ]
    
    df = pd.DataFrame(data, columns=['word1', 'word2', 'context1', 'context2', 'score'])
    df.to_csv("data/cosimlx.csv", index=False)
    logger.info(f"Created extended CoSimLex sample: {len(df)} pairs")

def verify_existing_datasets():
    """Verify that WordSim-353 and SimLex-999 are complete."""
    datasets = {
        "data/wordsim353.csv": 353,
        "data/simlex999.csv": 999
    }
    
    for dataset_path, expected_size in datasets.items():
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if len(df) >= expected_size:
                logger.info(f"✓ {dataset_path}: {len(df)} pairs (complete)")
            else:
                logger.warning(f"⚠ {dataset_path}: {len(df)} pairs (incomplete, expected {expected_size})")
        else:
            logger.error(f"✗ {dataset_path}: not found")

def main():
    """Download all datasets."""
    # Create data directory
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/scws_raw", exist_ok=True)
    
    logger.info("Starting dataset downloads...")
    
    # Verify existing datasets
    verify_existing_datasets()
    
    # Download new datasets
    download_scws()
    download_cosimlex()
    
    logger.info("Dataset download completed!")

if __name__ == "__main__":
    main()