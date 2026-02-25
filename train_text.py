"""
Training Script for Text Sentiment Analysis Model
Supports LSTM training with various optimizers and schedulers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import argparse
from tqdm import tqdm

import text_config
from text_model import create_text_model, count_parameters
from text_data_loader import create_text_data_loaders
from text_preprocessing import load_glove_embeddings, Vocabulary


class TextTrainer:
    """Training engine for text sentiment analysis"""
    
    def __init__(self, model, dataloaders, vocab, class_weights=None,
                 device=None, log_dir=None, config=None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            dataloaders: Dictionary with 'train' and 'val' loaders
            vocab: Vocabulary instance
            class_weights: Class weights for loss function
            device: Device to train on
            log_dir: Directory for TensorBoard logs
            config: Configuration module
        """
        self.model = model
        self.dataloaders = dataloaders
        self.vocab = vocab
        self.config = config or text_config
        self.device = device or torch.device(self.config.TEXT_DEVICE)
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        if class_weights is not None and self.config.TEXT_USE_CLASS_WEIGHTS:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted CrossEntropyLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print(f"Using standard CrossEntropyLoss")
        
        # Optimizer
        if self.config.TEXT_OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.TEXT_LEARNING_RATE,
                weight_decay=self.config.TEXT_WEIGHT_DECAY
            )
        elif self.config.TEXT_OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.TEXT_LEARNING_RATE,
                weight_decay=self.config.TEXT_WEIGHT_DECAY
            )
        elif self.config.TEXT_OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.TEXT_LEARNING_RATE,
                momentum=self.config.TEXT_MOMENTUM,
                weight_decay=self.config.TEXT_WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.TEXT_OPTIMIZER}")
        
        # Learning rate scheduler
        if self.config.TEXT_LR_SCHEDULER == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.TEXT_LR_STEP_SIZE,
                gamma=self.config.TEXT_LR_GAMMA
            )
        elif self.config.TEXT_LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.TEXT_NUM_EPOCHS,
                eta_min=self.config.TEXT_LR_MIN
            )
        elif self.config.TEXT_LR_SCHEDULER == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.TEXT_LR_GAMMA,
                patience=5
            )
        else:
            self.scheduler = None
        
        # TensorBoard writer
        self.writer = None
        if self.config.TEXT_USE_TENSORBOARD and log_dir:
            self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config.TEXT_USE_AMP else None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(
            self.dataloaders['train'],
            desc=f'Epoch {epoch+1}/{self.config.TEXT_NUM_EPOCHS} [Train]'
        )
        
        for batch_idx, (texts, labels) in enumerate(pbar):
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.TEXT_USE_AMP and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(texts)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.TEXT_GRAD_CLIP > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.TEXT_GRAD_CLIP
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.TEXT_GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.TEXT_GRAD_CLIP
                    )
                
                self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * texts.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += texts.size(0)
            
            # Update progress bar
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(
            self.dataloaders['val'],
            desc=f'Epoch {epoch+1}/{self.config.TEXT_NUM_EPOCHS} [Val]  '
        )
        
        with torch.no_grad():
            for texts, labels in pbar:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * texts.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += texts.size(0)
                
                # Save predictions for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                current_acc = running_corrects.double() / total_samples
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item(), all_preds, all_labels
    
    def train(self, num_epochs=None):
        """Complete training loop"""
        num_epochs = num_epochs or self.config.TEXT_NUM_EPOCHS
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.config.TEXT_OPTIMIZER}")
        print(f"Learning rate: {self.config.TEXT_LEARNING_RATE}")
        print(f"Batch size: {self.config.TEXT_BATCH_SIZE}")
        print(f"Mixed precision: {self.config.TEXT_USE_AMP}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc + self.config.TEXT_EARLY_STOPPING_MIN_DELTA:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model! Val Acc: {val_acc:.4f}")
            else:
                self.early_stop_counter += 1
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.early_stop_counter >= self.config.TEXT_EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        print(f"{'='*60}\n")
        
        # Final evaluation and metrics
        self.evaluate_final()
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'vocab_size': len(self.vocab),
        }
        
        # Save last checkpoint
        checkpoint_path = self.config.TEXT_MODELS_DIR / 'last_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.config.TEXT_MODELS_DIR / 'best_text_model.pth'
            torch.save(checkpoint, best_path)
    
    def evaluate_final(self):
        """Final evaluation with metrics"""
        print("Evaluating best model...")
        
        # Load best model
        best_model_path = self.config.TEXT_MODELS_DIR / 'best_text_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        _, val_acc, val_preds, val_labels = self.validate_epoch(self.best_epoch)
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        self.plot_confusion_matrix(cm)
        
        # Classification report
        target_names = [self.config.SENTIMENT_CLASSES[i] for i in range(self.config.NUM_SENTIMENT_CLASSES)]
        report = classification_report(
            val_labels,
            val_preds,
            target_names=target_names,
            digits=4
        )
        
        print("\nClassification Report:")
        print(report)
        
        # Save report
        report_path = self.config.TEXT_RESULTS_DIR / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Calculate F1 score
        f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"\nWeighted F1 Score: {f1:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.SENTIMENT_CLASSES,
            yticklabels=self.config.SENTIMENT_CLASSES
        )
        plt.title('Text Sentiment Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.config.TEXT_RESULTS_DIR / 'confusion_matrix_text.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Text Model: Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_title('Text Model: Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.config.TEXT_RESULTS_DIR / 'training_curves_text.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Text Sentiment Model')
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data (CSV or IMDb directory)')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (optional)')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation data (optional)')
    parser.add_argument('--data-format', type=str, default='csv',
                       choices=['csv', 'imdb'],
                       help='Data format: csv or imdb directory structure')
    parser.add_argument('--model', type=str, default='lstm_attention',
                       choices=['lstm_attention', 'simple_lstm'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=text_config.TEXT_NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=text_config.TEXT_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=text_config.TEXT_LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--use-glove', action='store_true',
                       help='Use pretrained GloVe embeddings')
    parser.add_argument('--glove-path', type=str, default=None,
                       help='Path to GloVe embeddings file')
    
    args = parser.parse_args()
    
    # Update config
    text_config.TEXT_BATCH_SIZE = args.batch_size
    text_config.TEXT_LEARNING_RATE = args.lr
    text_config.TEXT_NUM_EPOCHS = args.epochs
    
    print(f"\n{'='*60}")
    print(f"TEXT SENTIMENT ANALYSIS TRAINING")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Data format: {args.data_format}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Use GloVe: {args.use_glove}")
    print(f"{'='*60}\n")
    
    # Set random seed
    torch.manual_seed(text_config.TEXT_RANDOM_SEED)
    np.random.seed(text_config.TEXT_RANDOM_SEED)
    
    # Create data loaders
    print("Creating data loaders...")
    vocab_path = text_config.TEXT_MODELS_DIR / 'vocabulary.pkl'
    dataloaders, vocab, class_weights = create_text_data_loaders(
        train_path=args.train_data,
        test_path=args.test_data,
        val_path=args.val_data,
        vocab_path=vocab_path,
        batch_size=args.batch_size,
        data_format=args.data_format,
        config=text_config
    )
    
    # Load GloVe embeddings if requested
    pretrained_embeddings = None
    if args.use_glove:
        glove_path = args.glove_path or text_config.GLOVE_PATH
        if Path(glove_path).exists():
            print(f"\nLoading GloVe embeddings from {glove_path}...")
            pretrained_embeddings = load_glove_embeddings(
                glove_path,
                vocab.word2idx,
                text_config.EMBEDDING_DIM
            )
        else:
            print(f"\nWarning: GloVe file not found at {glove_path}")
            print("Training with random embeddings instead.")
    
    # Create model
    print("\nCreating model...")
    model = create_text_model(
        vocab_size=len(vocab),
        num_classes=text_config.NUM_SENTIMENT_CLASSES,
        model_type=args.model,
        pretrained_embeddings=pretrained_embeddings,
        config=text_config
    )
    
    count_parameters(model)
    
    # Create trainer
    log_dir = text_config.TEXT_LOGS_DIR / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    trainer = TextTrainer(
        model=model,
        dataloaders=dataloaders,
        vocab=vocab,
        class_weights=class_weights,
        log_dir=log_dir,
        config=text_config
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {text_config.TEXT_MODELS_DIR / 'best_text_model.pth'}")
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Results saved to: {text_config.TEXT_RESULTS_DIR}")


if __name__ == '__main__':
    main()
