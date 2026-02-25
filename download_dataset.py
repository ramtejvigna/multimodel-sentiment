"""
Utility script to download and prepare IMDb sentiment dataset
"""

import os
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import text_config


def download_file(url, destination):
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        destination: Path to save file
    """
    print(f"Downloading from {url}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"Downloaded to {destination}")


def extract_tar_gz(tar_path, extract_to):
    """
    Extract .tar.gz file
    
    Args:
        tar_path: Path to .tar.gz file
        extract_to: Directory to extract to
    """
    print(f"Extracting {tar_path}...")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    
    print(f"Extracted to {extract_to}")


def download_imdb_dataset(output_dir=None):
    """
    Download and prepare IMDb dataset
    
    Args:
        output_dir: Directory to save dataset (default: text_dataset/)
    """
    output_dir = Path(output_dir) if output_dir else text_config.TEXT_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download URL
    url = text_config.IMDB_URL
    tar_filename = 'aclImdb_v1.tar.gz'
    tar_path = output_dir / tar_filename
    
    # Check if already downloaded
    extracted_dir = output_dir / 'aclImdb'
    if extracted_dir.exists():
        print(f"IMDb dataset already exists at {extracted_dir}")
        return extracted_dir
    
    # Download
    if not tar_path.exists():
        download_file(url, tar_path)
    else:
        print(f"Archive already exists: {tar_path}")
    
    # Extract
    extract_tar_gz(tar_path, output_dir)
    
    # Verify extraction
    if extracted_dir.exists():
        train_dir = extracted_dir / 'train'
        test_dir = extracted_dir / 'test'
        
        print(f"\nDataset structure:")
        print(f"  Train directory: {train_dir}")
        print(f"  Test directory: {test_dir}")
        
        # Count files
        if train_dir.exists():
            pos_count = len(list((train_dir / 'pos').glob('*.txt')))
            neg_count = len(list((train_dir / 'neg').glob('*.txt')))
            print(f"\nTraining samples:")
            print(f"  Positive: {pos_count}")
            print(f"  Negative: {neg_count}")
            print(f"  Total: {pos_count + neg_count}")
        
        if test_dir.exists():
            pos_count = len(list((test_dir / 'pos').glob('*.txt')))
            neg_count = len(list((test_dir / 'neg').glob('*.txt')))
            print(f"\nTest samples:")
            print(f"  Positive: {pos_count}")
            print(f"  Negative: {neg_count}")
            print(f"  Total: {pos_count + neg_count}")
        
        # Cleanup tar file (optional)
        # tar_path.unlink()
        
        return extracted_dir
    else:
        raise RuntimeError(f"Failed to extract dataset to {extracted_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download IMDb sentiment dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for dataset')
    
    args = parser.parse_args()
    
    print("="*60)
    print("IMDb Dataset Downloader")
    print("="*60)
    
    dataset_dir = download_imdb_dataset(args.output_dir)
    
    print("\n" + "="*60)
    print("Download completed!")
    print(f"Dataset location: {dataset_dir}")
    print("\nTo train text model, run:")
    print(f"  python train_text.py --train-data {dataset_dir}/train --test-data {dataset_dir}/test --data-format imdb")
    print("="*60)
