"""
Download real datasets for similarity evaluation tasks.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import json
from pathlib import Path


def download_file(url, filename):
    """Download a file from URL."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
    else:
        print(f"✓ {filename} already exists")


def download_wordsim353():
    """Download WordSim-353 dataset."""
    url = "http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip"
    zip_path = "data/wordsim353.zip"
    
    download_file(url, zip_path)
    
    # Extract and process
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/wordsim353_raw')
    
    # Convert to standard format
    combined_file = 'data/wordsim353_raw/combined.csv'
    if os.path.exists(combined_file):
        df = pd.read_csv(combined_file)
        df.columns = ['word1', 'word2', 'score']
        df.to_csv('data/wordsim353.csv', index=False)
        print("✓ Processed WordSim-353")


def download_simlex999():
    """Download SimLex-999 dataset."""
    url = "https://fh295.github.io/SimLex-999.zip"
    zip_path = "data/simlex999.zip"
    
    download_file(url, zip_path)
    
    # Extract and process
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/simlex999_raw')
    
    # Convert to standard format
    simlex_file = 'data/simlex999_raw/SimLex-999/SimLex-999.txt'
    if os.path.exists(simlex_file):
        df = pd.read_csv(simlex_file, sep='\t')
        df_subset = df[['word1', 'word2', 'SimLex999']].copy()
        df_subset.columns = ['word1', 'word2', 'score']
        df_subset.to_csv('data/simlex999.csv', index=False)
        print("✓ Processed SimLex-999")


def download_scws():
    """Download Stanford Contextual Word Similarities (SCWS) dataset."""
    # SCWS requires manual download due to license
    # Create placeholder with instructions
    scws_info = {
        "dataset": "SCWS",
        "url": "https://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes",
        "note": "Please download SCWS dataset manually and place in data/scws.csv",
        "format": "word1,word2,context1,context2,score"
    }
    
    # For now, create a minimal version
    scws_data = []
    contexts = [
        ("bank", "bank", "I went to the bank to deposit money.", "The river bank was muddy.", 3.0),
        ("plant", "plant", "The plant needs water.", "We will plant trees tomorrow.", 2.5),
        ("star", "star", "The star is bright tonight.", "She is a movie star.", 2.0),
        ("book", "book", "I read a good book.", "Please book a table.", 2.2),
        ("run", "run", "I like to run in the morning.", "The play had a long run.", 3.5)
    ]
    
    for word1, word2, ctx1, ctx2, score in contexts:
        scws_data.append({
            'word1': word1,
            'word2': word2,
            'context1': ctx1,
            'context2': ctx2,
            'score': score
        })
    
    pd.DataFrame(scws_data).to_csv('data/scws_sample.csv', index=False)
    print("✓ Created SCWS sample (full dataset requires manual download)")


def download_cosimlx():
    """Download CoSimLex dataset."""
    # CoSimLex is a newer dataset - create from available sources
    url = "https://github.com/ArmaanB/CoSimLex/raw/main/data/cosimlx.csv"
    
    try:
        df = pd.read_csv(url)
        df.to_csv('data/cosimlx.csv', index=False)
        print("✓ Downloaded CoSimLex")
    except:
        # Create sample if download fails
        print("⚠️ CoSimLex download failed, creating sample...")
        cosimlx_data = []
        
        examples = [
            ("bank", "bank", "financial", "river", 2.0),
            ("star", "star", "astronomy", "celebrity", 1.5),
            ("cell", "cell", "biology", "prison", 1.2),
            ("court", "court", "legal", "sports", 1.8),
            ("pitch", "pitch", "baseball", "music", 1.3)
        ]
        
        for w1, w2, ctx1, ctx2, score in examples:
            cosimlx_data.append({
                'word1': w1,
                'word2': w2,
                'context1': f"In {ctx1} context",
                'context2': f"In {ctx2} context",
                'score': score
            })
        
        pd.DataFrame(cosimlx_data).to_csv('data/cosimlx_sample.csv', index=False)


def download_glue_datasets():
    """Download GLUE benchmark datasets using Hugging Face datasets."""
    glue_script = """
from datasets import load_dataset
import json
import os

glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']

for task in glue_tasks:
    print(f"Downloading {task}...")
    try:
        dataset = load_dataset('glue', task)
        
        # Save train and validation splits
        train_data = [dict(example) for example in dataset['train']]
        val_data = [dict(example) for example in dataset['validation']]
        
        with open(f'data/{task}_train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(f'data/{task}_dev.json', 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"✓ {task} saved")
    except Exception as e:
        print(f"⚠️ Failed to download {task}: {e}")
"""
    
    with open('data/download_glue.py', 'w') as f:
        f.write(glue_script)
    
    print("✓ Created GLUE download script (run with datasets library)")


def main():
    """Download all datasets."""
    print("📊 Downloading real datasets...")
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Download similarity datasets
    try:
        download_wordsim353()
    except Exception as e:
        print(f"⚠️ WordSim-353 download failed: {e}")
    
    try:
        download_simlex999()
    except Exception as e:
        print(f"⚠️ SimLex-999 download failed: {e}")
    
    download_scws()
    download_cosimlx()
    
    # Prepare GLUE download
    download_glue_datasets()
    
    print("\n📝 Dataset Summary:")
    print("- WordSim-353: Standard word similarity dataset")
    print("- SimLex-999: Similarity dataset focusing on similarity vs relatedness")
    print("- SCWS: Contextual word similarity (sample created)")
    print("- CoSimLex: Context-dependent similarity (sample created)")
    print("- GLUE: Run 'python data/download_glue.py' with datasets library")
    
    print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()