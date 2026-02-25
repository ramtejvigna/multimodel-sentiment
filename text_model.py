"""
LSTM-based Text Sentiment Classification Models
Supports bidirectional LSTM with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import text_config


class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_dim, attention_dim):
        """
        Initialize attention layer
        
        Args:
            hidden_dim: LSTM hidden dimension
            attention_dim: Attention layer dimension
        """
        super(AttentionLayer, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, lstm_output):
        """
        Apply attention mechanism
        
        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        
        return context_vector, attention_weights.squeeze(-1)


class LSTMSentimentClassifier(nn.Module):
    """LSTM-based sentiment classifier with attention"""
    
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        bidirectional=True,
        dropout=0.3,
        use_attention=True,
        attention_dim=128,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        """
        Initialize LSTM sentiment classifier
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of sentiment classes
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            attention_dim: Attention layer dimension
            pretrained_embeddings: Pretrained embedding matrix (optional)
            freeze_embeddings: Whether to freeze embeddings during training
        """
        super(LSTMSentimentClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(lstm_output_dim, attention_dim)
            classifier_input_dim = lstm_output_dim
        else:
            classifier_input_dim = lstm_output_dim
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),  # Lighter dropout in middle layers
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(128, num_classes)
        )
        
        # Layer normalization (optional enhancement)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
    
    def forward(self, text, return_attention=False):
        """
        Forward pass
        
        Args:
            text: Input text tensor (batch_size, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Logits (and attention weights if requested)
        """
        # Embedding
        embedded = self.embedding(text)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                hidden = hidden[-1]
            context = hidden
            attention_weights = None
        
        # Classification
        logits = self.classifier(context)
        
        if return_attention and attention_weights is not None:
            return logits, attention_weights
        else:
            return logits


class SimpleLSTMClassifier(nn.Module):
    """Simpler LSTM classifier without attention (baseline)"""
    
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        dropout=0.5,
        pretrained_embeddings=None
    ):
        """Initialize simple LSTM classifier"""
        super(SimpleLSTMClassifier, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
    
    def forward(self, text):
        """Forward pass"""
        embedded = self.embedding(text)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        logits = self.classifier(hidden)
        return logits


def create_text_model(
    vocab_size,
    num_classes=2,
    model_type='lstm_attention',
    pretrained_embeddings=None,
    config=None
):
    """
    Create text sentiment model
    
    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of sentiment classes
        model_type: 'lstm_attention' or 'simple_lstm'
        pretrained_embeddings: Pretrained embedding matrix (optional)
        config: Configuration module
        
    Returns:
        PyTorch model
    """
    config = config or text_config
    
    if model_type == 'lstm_attention':
        model = LSTMSentimentClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            num_classes=num_classes,
            bidirectional=config.LSTM_BIDIRECTIONAL,
            dropout=config.LSTM_DROPOUT,
            use_attention=config.USE_ATTENTION,
            attention_dim=config.ATTENTION_DIM,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=False
        )
        print(f"Created LSTM model with attention")
    
    elif model_type == 'simple_lstm':
        model = SimpleLSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.LSTM_HIDDEN_DIM,
            num_classes=num_classes,
            dropout=config.LSTM_DROPOUT,
            pretrained_embeddings=pretrained_embeddings
        )
        print(f"Created simple LSTM model")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")
    
    return total_params, trainable_params


if __name__ == '__main__':
    # Test model creation
    print("Testing text sentiment model...\n")
    
    # Test parameters
    vocab_size = 10000
    num_classes = 2
    batch_size = 8
    seq_length = 50
    
    # Create models
    models_to_test = ['lstm_attention', 'simple_lstm']
    
    for model_type in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {model_type}")
        print('='*60)
        
        # Create model
        model = create_text_model(
            vocab_size=vocab_size,
            num_classes=num_classes,
            model_type=model_type
        )
        
        # Count parameters
        count_parameters(model)
        
        # Test forward pass
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        if model_type == 'lstm_attention':
            output, attention = model(dummy_input, return_attention=True)
            print(f"\nOutput shape: {output.shape}")
            print(f"Attention shape: {attention.shape}")
        else:
            output = model(dummy_input)
            print(f"\nOutput shape: {output.shape}")
        
        print(f"Expected shape: torch.Size([{batch_size}, {num_classes}])")
        
        # Test probabilities
        probs = torch.softmax(output, dim=1)
        print(f"Probabilities sum (first sample): {probs[0].sum().item():.4f}")
        print(f"Sample prediction: {probs[0].tolist()}")
    
    print("\n" + "="*60)
    print("Model test completed!")
