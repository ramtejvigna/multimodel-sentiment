"""
Simple test script to verify multimodal sentiment analysis system
Run this after training both models
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        import config
        import text_config
        from text_preprocessing import TextPreprocessor, Vocabulary
        from text_model import create_text_model
        from fusion_model import create_fusion_model
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_text_preprocessing():
    """Test text preprocessing pipeline"""
    print("\nTesting text preprocessing...")
    try:
        from text_preprocessing import TextPreprocessor, Vocabulary
        import text_config
        
        preprocessor = TextPreprocessor()
        
        # Test text cleaning
        sample_text = "This is AMAZING! 😊 #BestDay http://example.com @user123"
        cleaned = preprocessor.clean_text(sample_text)
        tokens = preprocessor.preprocess(sample_text)
        
        print(f"  Original: {sample_text}")
        print(f"  Cleaned: {cleaned}")
        print(f"  Tokens: {tokens}")
        
        # Test vocabulary
        vocab = Vocabulary()
        texts = ["This is great!", "This is terrible!", "This is okay."]
        vocab.build_from_texts(texts, preprocessor)
        
        encoded = vocab.encode(tokens)
        print(f"  Encoded: {encoded[:5]}...")
        print(f"  Vocab size: {len(vocab)}")
        
        print("✓ Text preprocessing works")
        return True
    except Exception as e:
        print(f"✗ Text preprocessing failed: {e}")
        return False


def test_fusion_models():
    """Test fusion model creation"""
    print("\nTesting fusion models...")
    try:
        import torch
        from fusion_model import create_fusion_model
        
        # Test late fusion
        late_fusion = create_fusion_model('late', text_weight=0.4, face_weight=0.6)
        
        text_probs = torch.softmax(torch.randn(2, 2), dim=1)
        face_probs = torch.softmax(torch.randn(2, 7), dim=1)
        
        result = late_fusion(text_probs, face_probs)
        print(f"  Late fusion result type: {type(result)}")
        
        print("✓ Fusion models work")
        return True
    except Exception as e:
        print(f"✗ Fusion models failed: {e}")
        return False


def test_models_exist():
    """Check if trained models exist"""
    print("\nChecking for trained models...")
    
    import config
    import text_config
    
    face_model_path = config.MODELS_DIR / 'best_model.pth'
    text_model_path = text_config.TEXT_MODELS_DIR / 'best_text_model.pth'
    vocab_path = text_config.TEXT_MODELS_DIR / 'vocabulary.pkl'
    
    face_exists = face_model_path.exists()
    text_exists = text_model_path.exists()
    vocab_exists = vocab_path.exists()
    
    if face_exists:
        print(f"  ✓ Face model found: {face_model_path}")
    else:
        print(f"  ✗ Face model NOT found: {face_model_path}")
        print(f"    Run: python train.py --model resnet18 --epochs 50")
    
    if text_exists:
        print(f"  ✓ Text model found: {text_model_path}")
    else:
        print(f"  ✗ Text model NOT found: {text_model_path}")
        print(f"    Run: python download_dataset.py")
        print(f"    Then: python train_text.py --train-data text_dataset/aclImdb/train --data-format imdb")
    
    if vocab_exists:
        print(f"  ✓ Vocabulary found: {vocab_path}")
    else:
        print(f"  ✗ Vocabulary NOT found: {vocab_path}")
    
    return face_exists and text_exists and vocab_exists


def test_multimodal_analyzer():
    """Test multimodal analyzer (if models exist)"""
    print("\nTesting multimodal analyzer...")
    
    if not test_models_exist():
        print("  ⚠ Skipping multimodal test - models not trained yet")
        return False
    
    try:
        from multimodal_inference import create_multimodal_analyzer
        
        print("  Loading models...")
        analyzer = create_multimodal_analyzer()
        
        # Test text analysis
        print("\n  Testing text analysis...")
        text_result = analyzer.analyze_text("I am so happy today!")
        print(f"    Prediction: {text_result['prediction']}")
        print(f"    Confidence: {text_result['confidence']:.2%}")
        
        print("\n✓ Multimodal analyzer works!")
        print("\nNext steps:")
        print("  1. Test face analysis with an image")
        print("  2. Test multimodal analysis")
        print("  3. Run web app: python app.py")
        
        return True
    except Exception as e:
        print(f"✗ Multimodal analyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Multimodal Sentiment Analysis - System Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Text Preprocessing", test_text_preprocessing()))
    results.append(("Fusion Models", test_fusion_models()))
    results.append(("Multimodal Analyzer", test_multimodal_analyzer()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All systems operational!")
        print("\nYou can now:")
        print("  - Run web app: python app.py")
        print("  - Test inference: python multimodal_inference.py")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    
    print("="*60)


if __name__ == '__main__':
    main()
