"""
Text Preprocessing Utilities for Sentiment Analysis
Handles text cleaning, tokenization, and vocabulary building
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import text_config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline for sentiment analysis"""
    
    def __init__(self, config=None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration module (defaults to text_config)
        """
        self.config = config or text_config
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if self.config.USE_LEMMATIZATION else None
        
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        if self.config.LOWERCASE:
            text = text.lower()
        
        # Remove URLs
        if self.config.REMOVE_URLS:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        if self.config.REMOVE_MENTIONS:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text)
        if self.config.REMOVE_HASHTAGS:
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove numbers
        if self.config.REMOVE_NUMBERS:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Use NLTK word tokenizer
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split
            tokens = text.split()
        
        # Remove punctuation if configured
        if self.config.REMOVE_PUNCTUATION:
            translator = str.maketrans('', '', string.punctuation)
            tokens = [token.translate(translator) for token in tokens]
            tokens = [token for token in tokens if token]  # Remove empty strings
        
        # Remove stopwords if configured
        if self.config.REMOVE_STOPWORDS:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Lemmatization if configured
        if self.config.USE_LEMMATIZATION and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        return tokens


class Vocabulary:
    """Vocabulary builder and manager"""
    
    def __init__(self, config=None):
        """
        Initialize vocabulary
        
        Args:
            config: Configuration module
        """
        self.config = config or text_config
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Add special tokens
        self.add_word(self.config.PAD_TOKEN)
        self.add_word(self.config.UNK_TOKEN)
        if hasattr(self.config, 'SOS_TOKEN'):
            self.add_word(self.config.SOS_TOKEN)
        if hasattr(self.config, 'EOS_TOKEN'):
            self.add_word(self.config.EOS_TOKEN)
        
        self.pad_idx = self.word2idx[self.config.PAD_TOKEN]
        self.unk_idx = self.word2idx[self.config.UNK_TOKEN]
    
    def add_word(self, word):
        """Add word to vocabulary"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_counts[word] += 1
    
    def build_from_texts(self, texts, preprocessor=None):
        """
        Build vocabulary from list of texts
        
        Args:
            texts: List of text strings
            preprocessor: TextPreprocessor instance
        """
        if preprocessor is None:
            preprocessor = TextPreprocessor(self.config)
        
        # Count all words
        print("Building vocabulary from texts...")
        for text in texts:
            tokens = preprocessor.preprocess(text)
            for token in tokens:
                self.word_counts[token] += 1
        
        # Filter by frequency and max vocab size
        valid_words = [
            word for word, count in self.word_counts.most_common()
            if count >= self.config.MIN_WORD_FREQ
        ]
        
        # Limit to max vocab size (excluding special tokens)
        max_words = self.config.MAX_VOCAB_SIZE - len(self.word2idx)
        valid_words = valid_words[:max_words]
        
        # Add to vocabulary
        for word in valid_words:
            if word not in self.word2idx:
                self.add_word(word)
        
        print(f"Vocabulary built: {len(self.word2idx)} words")
        print(f"  Special tokens: {len([w for w in self.word2idx if w.startswith('<')])}")
        print(f"  Regular words: {len(self.word2idx) - len([w for w in self.word2idx if w.startswith('<')])}")
    
    def encode(self, tokens):
        """
        Convert tokens to indices
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of indices
        """
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices):
        """
        Convert indices to tokens
        
        Args:
            indices: List of indices
            
        Returns:
            List of tokens
        """
        return [self.idx2word.get(idx, self.config.UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath):
        """Save vocabulary to file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'config': {
                    'PAD_TOKEN': self.config.PAD_TOKEN,
                    'UNK_TOKEN': self.config.UNK_TOKEN,
                }
            }, f)
        print(f"Vocabulary saved to {filepath}")
    
    def load(self, filepath):
        """Load vocabulary from file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.word_counts = data['word_counts']
        self.pad_idx = self.word2idx[self.config.PAD_TOKEN]
        self.unk_idx = self.word2idx[self.config.UNK_TOKEN]
        
        print(f"Vocabulary loaded from {filepath}: {len(self.word2idx)} words")


def pad_sequence(sequence, max_length, pad_value=0, method='post'):
    """
    Pad or truncate sequence to max_length
    
    Args:
        sequence: List of values
        max_length: Maximum sequence length
        pad_value: Value to use for padding
        method: 'pre' or 'post' padding/truncation
        
    Returns:
        Padded sequence
    """
    if len(sequence) > max_length:
        # Truncate
        if method == 'post':
            return sequence[:max_length]
        else:
            return sequence[-max_length:]
    elif len(sequence) < max_length:
        # Pad
        padding = [pad_value] * (max_length - len(sequence))
        if method == 'post':
            return sequence + padding
        else:
            return padding + sequence
    else:
        return sequence


def load_glove_embeddings(filepath, word2idx, embedding_dim):
    """
    Load GloVe embeddings for vocabulary
    
    Args:
        filepath: Path to GloVe file
        word2idx: Word to index mapping
        embedding_dim: Embedding dimension
        
    Returns:
        Embedding matrix (numpy array)
    """
    print(f"Loading GloVe embeddings from {filepath}...")
    
    # Initialize embedding matrix
    vocab_size = len(word2idx)
    embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    # Load embeddings
    found = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            
            if word in word2idx:
                vector = np.array(values[1:], dtype='float32')
                embeddings[word2idx[word]] = vector
                found += 1
    
    print(f"Found {found}/{vocab_size} words in GloVe embeddings")
    print(f"Coverage: {100 * found / vocab_size:.2f}%")
    
    return embeddings


if __name__ == '__main__':
    # Test preprocessing
    print("Testing text preprocessing...\n")
    
    # Sample texts
    texts = [
        "This movie was absolutely AMAZING! I loved every minute of it. 😊 #BestMovieEver",
        "Terrible film. Waste of time and money. @director please improve!",
        "It was okay... not great, not terrible. Just meh.",
        "http://example.com Check out this review! The acting was superb!!!",
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Process texts
    for i, text in enumerate(texts, 1):
        print(f"Text {i}:")
        print(f"  Original: {text}")
        print(f"  Cleaned:  {preprocessor.clean_text(text)}")
        print(f"  Tokens:   {preprocessor.preprocess(text)}")
        print()
    
    # Test vocabulary building
    print("\nTesting vocabulary building...")
    vocab = Vocabulary()
    vocab.build_from_texts(texts, preprocessor)
    
    # Test encoding
    test_text = "This movie was amazing!"
    tokens = preprocessor.preprocess(test_text)
    encoded = vocab.encode(tokens)
    decoded = vocab.decode(encoded)
    
    print(f"\nTest encoding:")
    print(f"  Text:    {test_text}")
    print(f"  Tokens:  {tokens}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    
    # Test padding
    print(f"\nTest padding:")
    padded = pad_sequence(encoded, max_length=10, pad_value=vocab.pad_idx)
    print(f"  Original length: {len(encoded)}")
    print(f"  Padded to 10:    {padded}")
