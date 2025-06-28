"""
Data pipeline for downloading and preparing datasets.
"""

import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import json
from pathlib import Path


class DataPipeline:
    """Handles data downloading and preprocessing."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def prepare_all_datasets(self):
        """Prepare all required datasets."""
        print("🔄 Preparing datasets...")
        
        # Prepare similarity datasets
        self.prepare_similarity_datasets()
        
        # Prepare GLUE datasets
        self.prepare_glue_datasets()
        
        print("✅ All datasets prepared!")
    
    def prepare_similarity_datasets(self):
        """Prepare word similarity datasets."""
        datasets = {
            'wordsim353': self.create_wordsim353,
            'simlex999': self.create_simlex999,
            'cosimlx': self.create_cosimlx,
            'scws': self.create_scws
        }
        
        for name, creator_func in datasets.items():
            filepath = self.data_dir / f"{name}.csv"
            if not filepath.exists():
                print(f"Creating {name} dataset...")
                df = creator_func()
                df.to_csv(filepath, index=False)
                print(f"✓ {name} saved to {filepath}")
            else:
                print(f"✓ {name} already exists")
    
    def create_wordsim353(self):
        """Create WordSim-353 dataset (synthetic version)."""
        np.random.seed(42)
        word_pairs = [
            ('car', 'automobile'), ('gem', 'jewel'), ('journey', 'voyage'),
            ('boy', 'lad'), ('coast', 'shore'), ('asylum', 'madhouse'),
            ('magician', 'wizard'), ('midday', 'noon'), ('furnace', 'stove'),
            ('food', 'fruit'), ('bird', 'cock'), ('bird', 'crane'),
            ('tool', 'implement'), ('brother', 'monk'), ('crane', 'implement'),
            ('lad', 'brother'), ('journey', 'car'), ('monk', 'oracle'),
            ('cemetery', 'woodland'), ('food', 'rooster'), ('coast', 'hill'),
            ('forest', 'graveyard'), ('shore', 'woodland'), ('monk', 'slave'),
            ('coast', 'forest'), ('lad', 'wizard'), ('chord', 'smile'),
            ('glass', 'magician'), ('noon', 'string'), ('rooster', 'voyage')
        ]
        
        data = []
        for _ in range(353):
            word1, word2 = word_pairs[np.random.randint(0, len(word_pairs))]
            score = np.random.uniform(0, 10)
            data.append({
                'word1': word1,
                'word2': word2,
                'score': score
            })
        
        return pd.DataFrame(data)
    
    def create_simlex999(self):
        """Create SimLex-999 dataset (synthetic version)."""
        np.random.seed(43)
        word_pairs = [
            ('old', 'new'), ('smart', 'intelligent'), ('hard', 'difficult'),
            ('happy', 'cheerful'), ('fast', 'quick'), ('easy', 'simple'),
            ('dark', 'light'), ('short', 'long'), ('bad', 'terrible'),
            ('big', 'huge'), ('beautiful', 'pretty'), ('clean', 'dirty'),
            ('expensive', 'costly'), ('quiet', 'silent'), ('thin', 'slim'),
            ('strange', 'weird'), ('wide', 'broad'), ('bad', 'awful'),
            ('certain', 'sure'), ('decent', 'proper'), ('exact', 'precise')
        ]
        
        data = []
        for _ in range(999):
            word1, word2 = word_pairs[np.random.randint(0, len(word_pairs))]
            score = np.random.uniform(0, 10)
            data.append({
                'word1': word1,
                'word2': word2,
                'score': score
            })
        
        return pd.DataFrame(data)
    
    def create_cosimlx(self):
        """Create CoSimLex dataset with contexts (synthetic version)."""
        np.random.seed(44)
        context_pairs = [
            ('bank', 'The bank of the river was muddy.', 'bank', 'I need to go to the bank to withdraw money.'),
            ('plant', 'The plant needs water.', 'plant', 'They will plant new trees tomorrow.'),
            ('light', 'The light from the sun is bright.', 'light', 'This bag is very light.'),
            ('bark', 'The dog began to bark loudly.', 'bark', 'The bark of the tree was rough.'),
            ('bat', 'He swung the bat at the ball.', 'bat', 'A bat flew out of the cave.'),
            ('spring', 'Spring is my favorite season.', 'spring', 'The spring in the mattress broke.'),
            ('match', 'They won the match easily.', 'match', 'Use a match to light the candle.'),
            ('fair', 'The weather is fair today.', 'fair', 'Everyone deserves a fair chance.'),
            ('rose', 'She rose from her chair.', 'rose', 'He gave her a red rose.'),
            ('leaves', 'The leaves are turning yellow.', 'leaves', 'She leaves for work at 8am.')
        ]
        
        data = []
        for _ in range(1024):
            word1, context1, word2, context2 = context_pairs[np.random.randint(0, len(context_pairs))]
            score = np.random.uniform(0, 10)
            data.append({
                'word1': word1,
                'word2': word2,
                'context1': context1,
                'context2': context2,
                'score': score
            })
        
        return pd.DataFrame(data)
    
    def create_scws(self):
        """Create Stanford Contextual Word Similarities dataset (synthetic version)."""
        np.random.seed(45)
        context_pairs = [
            ('run', 'I need to run to catch the bus.', 'run', 'The play had a successful run on Broadway.'),
            ('book', 'I love reading this book.', 'book', 'Please book a table for dinner.'),
            ('watch', 'I like to watch movies.', 'watch', 'My watch stopped working.'),
            ('play', 'Children love to play games.', 'play', 'We saw a play at the theater.'),
            ('right', 'Turn right at the corner.', 'right', 'You have the right to remain silent.'),
            ('mean', 'What does this word mean?', 'mean', 'Don\'t be mean to others.'),
            ('rock', 'The boat will rock in the waves.', 'rock', 'He threw a rock into the lake.'),
            ('present', 'She gave me a nice present.', 'present', 'Please present your findings.'),
            ('wind', 'The wind is very strong today.', 'wind', 'Please wind up the clock.'),
            ('tear', 'A tear rolled down her cheek.', 'tear', 'Be careful not to tear the paper.')
        ]
        
        data = []
        for _ in range(2003):
            word1, context1, word2, context2 = context_pairs[np.random.randint(0, len(context_pairs))]
            score = np.random.uniform(0, 10)
            data.append({
                'word1': word1,
                'word2': word2,
                'context1': context1,
                'context2': context2,
                'score': score
            })
        
        return pd.DataFrame(data)
    
    def prepare_glue_datasets(self):
        """Prepare GLUE benchmark datasets (synthetic versions)."""
        glue_tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']
        
        for task in glue_tasks:
            train_file = self.data_dir / f"{task}_train.json"
            dev_file = self.data_dir / f"{task}_dev.json"
            
            if not train_file.exists():
                print(f"Creating {task} dataset...")
                train_data, dev_data = self.create_glue_dataset(task)
                
                with open(train_file, 'w') as f:
                    json.dump(train_data, f, indent=2)
                with open(dev_file, 'w') as f:
                    json.dump(dev_data, f, indent=2)
                
                print(f"✓ {task} saved")
            else:
                print(f"✓ {task} already exists")
    
    def create_glue_dataset(self, task_name):
        """Create synthetic GLUE dataset."""
        np.random.seed(hash(task_name) % 100)
        
        if task_name == 'cola':
            # Corpus of Linguistic Acceptability
            sentences = [
                "The book was written by John.",
                "Book the was by written John.",
                "She gave him the book.",
                "Gave she book the him.",
                "The cat sat on the mat.",
                "On cat the sat mat the."
            ]
            train_data = []
            dev_data = []
            
            for i in range(1000):
                sentence = sentences[i % len(sentences)]
                label = 1 if i % 2 == 0 else 0  # Even indices are acceptable
                example = {'sentence': sentence, 'label': label}
                
                if i < 800:
                    train_data.append(example)
                else:
                    dev_data.append(example)
        
        elif task_name == 'sst2':
            # Stanford Sentiment Treebank
            positive_sentences = [
                "This movie is absolutely fantastic!",
                "I loved every moment of it.",
                "Best film I've seen this year.",
                "Highly recommend this masterpiece."
            ]
            negative_sentences = [
                "This movie was terrible.",
                "I wasted two hours of my life.",
                "Boring and predictable plot.",
                "Poor acting and direction."
            ]
            
            train_data = []
            dev_data = []
            
            for i in range(1000):
                if i % 2 == 0:
                    sentence = positive_sentences[i % len(positive_sentences)]
                    label = 1
                else:
                    sentence = negative_sentences[i % len(negative_sentences)]
                    label = 0
                
                example = {'sentence': sentence, 'label': label}
                
                if i < 800:
                    train_data.append(example)
                else:
                    dev_data.append(example)
        
        elif task_name in ['mrpc', 'qqp']:
            # Paraphrase detection
            paraphrase_pairs = [
                ("The cat is sleeping on the sofa.", "A feline is resting on the couch."),
                ("It's raining outside.", "The weather is rainy."),
                ("She went to the store.", "She visited the shop."),
                ("The food was delicious.", "The meal tasted great.")
            ]
            non_paraphrase_pairs = [
                ("The cat is sleeping.", "The dog is running."),
                ("It's sunny today.", "It's raining outside."),
                ("She went home.", "He stayed at work."),
                ("The food was cold.", "The service was excellent.")
            ]
            
            train_data = []
            dev_data = []
            
            for i in range(1000):
                if i % 2 == 0:
                    s1, s2 = paraphrase_pairs[i % len(paraphrase_pairs)]
                    label = 1
                else:
                    s1, s2 = non_paraphrase_pairs[i % len(non_paraphrase_pairs)]
                    label = 0
                
                example = {'sentence1': s1, 'sentence2': s2, 'label': label}
                
                if i < 800:
                    train_data.append(example)
                else:
                    dev_data.append(example)
        
        else:
            # Generic sentence pair task
            train_data = []
            dev_data = []
            
            for i in range(1000):
                s1 = f"This is sentence {i} for {task_name}."
                s2 = f"This is another sentence {i} for {task_name}."
                label = i % 2
                
                example = {'sentence1': s1, 'sentence2': s2, 'label': label}
                
                if i < 800:
                    train_data.append(example)
                else:
                    dev_data.append(example)
        
        return train_data, dev_data


if __name__ == "__main__":
    # Run data preparation
    pipeline = DataPipeline()
    pipeline.prepare_all_datasets()