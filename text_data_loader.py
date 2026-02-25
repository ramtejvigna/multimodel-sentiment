"""
PyTorch Dataset and DataLoader for Text Sentiment Analysis
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import text_config
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequence as pad_seq


class TextSentimentDataset(Dataset):
    """PyTorch Dataset for text sentiment analysis"""
    
    def __init__(self, texts, labels, vocab, preprocessor, config=None, augment=False):
        """
        Initialize dataset
        
        Args:
            texts: List of text strings
            labels: List of sentiment labels (int)
            vocab: Vocabulary instance
            preprocessor: TextPreprocessor instance
            config: Configuration module
            augment: Whether to apply data augmentation
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.config = config or text_config
        self.augment = augment
        
        # Preprocess and encode all texts
        self.encoded_texts = []
        for text in self.texts:
            tokens = self.preprocessor.preprocess(text)
            encoded = self.vocab.encode(tokens)
            # Pad/truncate
            encoded = pad_seq(
                encoded,
                self.config.MAX_SEQ_LENGTH,
                self.vocab.pad_idx,
                self.config.PADDING_METHOD
            )
            self.encoded_texts.append(encoded)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Returns:
            Tuple of (text_tensor, label_tensor)
        """
        text = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return text, label


def collate_batch(batch):
    """
    Custom collate function for batching
    
    Args:
        batch: List of (text, label) tuples
        
    Returns:
        Batched tensors
    """
    texts, labels = zip(*batch)
    
    # Stack texts and labels
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    
    return texts, labels


def load_imdb_data(data_dir):
    """
    Load IMDb dataset from directory structure
    
    Args:
        data_dir: Path to IMDb dataset directory
        
    Returns:
        Tuple of (texts, labels)
    """
    data_dir = Path(data_dir)
    
    texts = []
    labels = []
    
    # Load positive reviews
    pos_dir = data_dir / 'pos'
    if pos_dir.exists():
        for file_path in pos_dir.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)  # Positive
    
    # Load negative reviews
    neg_dir = data_dir / 'neg'
    if neg_dir.exists():
        for file_path in neg_dir.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)  # Negative
    
    return texts, labels


def load_csv_data(csv_path, text_column='text', label_column='label'):
    """
    Load dataset from CSV file
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple of (texts, labels)
    """
    df = pd.read_csv(csv_path)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    return texts, labels


def load_sentiment_csv_data(csv_path, text_column='text', sentiment_column='sentiment'):
    """
    Load sentiment dataset from CSV file (e.g., final_sentiment_data.csv)
    
    Handles sentiment values of -1 (negative) and 1 (positive) and converts
    them to binary labels 0 and 1 for classification.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column (default: 'text')
        sentiment_column: Name of sentiment column with -1/1 values (default: 'sentiment')
        
    Returns:
        Tuple of (texts, labels) where labels are 0 (negative) or 1 (positive)
    """
    df = pd.read_csv(csv_path)
    texts = df[text_column].tolist()
    sentiments = df[sentiment_column].tolist()
    
    # Convert sentiment: -1 -> 0 (negative), 1 -> 1 (positive)
    labels = [0 if s == -1 else 1 for s in sentiments]
    
    print(f"Loaded sentiment data from {csv_path}")
    print(f"  Total samples: {len(texts)}")
    print(f"  Positive samples: {sum(labels)}")
    print(f"  Negative samples: {len(labels) - sum(labels)}")
    
    return texts, labels


def create_text_data_loaders(
    train_path,
    test_path=None,
    val_path=None,
    vocab_path=None,
    batch_size=None,
    augment_train=True,
    data_format='csv',
    config=None
):
    """
    Create train, validation, and test data loaders
    
    Args:
        train_path: Path to training data
        test_path: Path to test data (optional)
        val_path: Path to validation data (optional)
        vocab_path: Path to saved vocabulary (optional)
        batch_size: Batch size for dataloaders
        augment_train: Whether to augment training data
        data_format: 'csv' or 'imdb' directory structure
        config: Configuration module
        
    Returns:
        Dictionary of dataloaders and vocabulary
    """
    config = config or text_config
    batch_size = batch_size or config.TEXT_BATCH_SIZE
    
    print("\nCreating text data loaders...")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Load data based on format
    if data_format == 'csv':
        train_texts, train_labels = load_csv_data(train_path)
        
        if val_path:
            val_texts, val_labels = load_csv_data(val_path)
        else:
            # Split training data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels,
                test_size=config.VAL_SPLIT,
                random_state=config.TEXT_RANDOM_SEED,
                stratify=train_labels
            )
        
        if test_path:
            test_texts, test_labels = load_csv_data(test_path)
        else:
            test_texts, test_labels = None, None
    
    elif data_format == 'sentiment_csv':
        # Load from final_sentiment_data.csv format
        train_texts, train_labels = load_sentiment_csv_data(train_path)
        
        if val_path:
            val_texts, val_labels = load_sentiment_csv_data(val_path)
        else:
            # Split training data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels,
                test_size=config.VAL_SPLIT,
                random_state=config.TEXT_RANDOM_SEED,
                stratify=train_labels
            )
        
        if test_path:
            test_texts, test_labels = load_sentiment_csv_data(test_path)
        else:
            test_texts, test_labels = None, None
    
    elif data_format == 'imdb':
        train_texts, train_labels = load_imdb_data(train_path)
        
        if test_path:
            test_texts, test_labels = load_imdb_data(test_path)
        else:
            test_texts, test_labels = None, None
        
        # Create validation split from training
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=config.VAL_SPLIT,
            random_state=config.TEXT_RANDOM_SEED,
            stratify=train_labels
        )
    
    else:
        raise ValueError(f"Unknown data format: {data_format}")
    
    print(f"Loaded data:")
    print(f"  Training samples:   {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    if test_texts:
        print(f"  Test samples:       {len(test_texts)}")
    
    # Build or load vocabulary
    if vocab_path and Path(vocab_path).exists():
        print(f"\nLoading vocabulary from {vocab_path}")
        vocab = Vocabulary(config)
        vocab.load(vocab_path)
    else:
        print("\nBuilding vocabulary from training data...")
        vocab = Vocabulary(config)
        vocab.build_from_texts(train_texts, preprocessor)
        
        # Save vocabulary
        if vocab_path:
            vocab.save(vocab_path)
        else:
            default_vocab_path = config.TEXT_MODELS_DIR / 'vocabulary.pkl'
            vocab.save(default_vocab_path)
    
    # Compute class weights for imbalanced data
    class_weights = None
    if config.TEXT_USE_CLASS_WEIGHTS:
        unique_labels = np.unique(train_labels)
        weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=train_labels
        )
        class_weights = torch.FloatTensor(weights)
        print(f"\nClass weights: {class_weights.numpy()}")
    
    # Create datasets
    train_dataset = TextSentimentDataset(
        train_texts, train_labels, vocab, preprocessor, config,
        augment=augment_train and config.TEXT_AUGMENTATION
    )
    
    val_dataset = TextSentimentDataset(
        val_texts, val_labels, vocab, preprocessor, config,
        augment=False
    )
    
    test_dataset = None
    if test_texts:
        test_dataset = TextSentimentDataset(
            test_texts, test_labels, vocab, preprocessor, config,
            augment=False
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TEXT_SHUFFLE_TRAIN,
        num_workers=config.TEXT_NUM_WORKERS,
        pin_memory=config.TEXT_PIN_MEMORY,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.TEXT_NUM_WORKERS,
        pin_memory=config.TEXT_PIN_MEMORY,
        collate_fn=collate_batch
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.TEXT_NUM_WORKERS,
            pin_memory=config.TEXT_PIN_MEMORY,
            collate_fn=collate_batch
        )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
    }
    
    if test_loader:
        dataloaders['test'] = test_loader
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    if test_loader:
        print(f"  Test batches:  {len(test_loader)}")
    
    return dataloaders, vocab, class_weights


if __name__ == '__main__':
    # Test data loading
    print("Testing text data loader...")
    
    # Create sample CSV file for testing
    import tempfile
    import csv
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        
        # Positive samples
        writer.writerow(['This movie is fantastic and amazing!', 1])
        writer.writerow(['I absolutely loved it. Best film ever!', 1])
        writer.writerow(['Great story, excellent acting. Highly recommend!', 1])
        writer.writerow(['Wonderful experience, will watch again!', 1])
        
        # Negative samples
        writer.writerow(['Terrible movie. Complete waste of time.', 0])
        writer.writerow(['I hated every minute of it. Awful!', 0])
        writer.writerow(['Boring and poorly made. Do not watch.', 0])
        writer.writerow(['Worst film I have ever seen. Horrible!', 0])
        
        temp_csv = f.name
    
    print(f"Created temporary CSV: {temp_csv}\n")
    
    try:
        # Create dataloaders
        dataloaders, vocab, class_weights = create_text_data_loaders(
            train_path=temp_csv,
            batch_size=4,
            data_format='csv'
        )
        
        # Test one batch
        print("\nTesting one batch from training data:")
        train_loader = dataloaders['train']
        texts, labels = next(iter(train_loader))
        
        print(f"  Batch shape: {texts.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sample text (encoded): {texts[0][:20]}...")
        print(f"  Sample label: {labels[0].item()}")
        
        # Decode sample
        decoded = vocab.decode(texts[0].tolist())
        print(f"  Decoded (first 10 tokens): {decoded[:10]}")
        
        print("\nTest completed successfully!")
        
    finally:
        # Cleanup
        import os
        os.remove(temp_csv)
