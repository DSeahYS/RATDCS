"""
Batch Processing for Large-Scale Feature Extraction

This module provides batch processing capabilities for extracting features from
large datasets of light curves with memory-efficient processing, checkpointing,
and progress tracking.

Based on RATDCS ARCHITECTURE.md Section 3.1.2 - Exoplanet Candidate Classifier
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_extractor import FeatureExtractor
from .preprocess import LightCurvePreprocessor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process large batches of light curves with checkpointing and memory management.
    
    Supports:
    - Batch processing with configurable batch size
    - Checkpointing for resumable processing
    - Progress tracking with tqdm
    - Memory profiling and optimization
    - Distributed processing support (future)
    
    Attributes:
        batch_size (int): Number of files to process per batch
        checkpoint_dir (Path): Directory for checkpoint files
        output_dir (Path): Directory for output files
        feature_extractor (FeatureExtractor): Feature extraction instance
        preprocessor (LightCurvePreprocessor): Preprocessing instance
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "data/processed",
        feature_set: str = 'comprehensive',
        n_jobs: int = 4,
        **extractor_kwargs
    ):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of light curves to process per batch
            checkpoint_dir: Directory for checkpoint files (default: output_dir/checkpoints)
            output_dir: Output directory for processed features
            feature_set: TSFresh feature set to use
            n_jobs: Number of parallel jobs
            **extractor_kwargs: Additional arguments for FeatureExtractor
        """
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            feature_set=feature_set,
            n_jobs=n_jobs,
            **extractor_kwargs
        )
        
        self.preprocessor = self.feature_extractor.preprocessor
        
        logger.info(f"Initialized BatchProcessor with batch_size={batch_size}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def save_checkpoint(
        self,
        checkpoint_name: str,
        data: Dict
    ):
        """
        Save processing checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint file
            data: Dictionary with checkpoint data
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        # Convert numpy types for JSON serialization
        checkpoint_data = {}
        for key, value in data.items():
            if isinstance(value, (np.ndarray, pd.Series)):
                checkpoint_data[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                checkpoint_data[key] = float(value)
            else:
                checkpoint_data[key] = value
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_name: str
    ) -> Optional[Dict]:
        """
        Load processing checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint file
        
        Returns:
            Dictionary with checkpoint data, or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        if not checkpoint_path.exists():
            return None
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return data
    
    def get_processed_files(
        self,
        checkpoint_name: str
    ) -> List[str]:
        """
        Get list of already processed files from checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint
        
        Returns:
            List of processed file paths
        """
        checkpoint = self.load_checkpoint(checkpoint_name)
        if checkpoint and 'processed_files' in checkpoint:
            return checkpoint['processed_files']
        return []
    
    def process_batch(
        self,
        fits_files: List[Union[str, Path]],
        labels: Optional[List[int]] = None,
        batch_start: int = 0
    ) -> pd.DataFrame:
        """
        Process a single batch of FITS files.
        
        Args:
            fits_files: List of FITS file paths
            labels: Optional labels for each file
            batch_start: Starting index for this batch
        
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Processing batch of {len(fits_files)} files (starting at index {batch_start})")
        
        # Extract features
        features, valid_ids, valid_labels = self.feature_extractor.extract_from_fits(
            fits_files,
            labels=labels
        )
        
        # Add labels if provided
        if valid_labels is not None:
            features['label'] = valid_labels
        
        # Add IDs
        features['id'] = valid_ids
        
        # Add batch information
        features['batch_index'] = batch_start // self.batch_size
        
        return features
    
    def process_dataset(
        self,
        fits_files: List[Union[str, Path]],
        output_filename: str,
        labels: Optional[List[int]] = None,
        checkpoint_name: Optional[str] = None,
        resume: bool = True
    ) -> Dict:
        """
        Process entire dataset in batches.
        
        Args:
            fits_files: List of all FITS file paths
            output_filename: Output CSV filename
            labels: Optional labels for classification
            checkpoint_name: Name for checkpoint files (default: based on output_filename)
            resume: Whether to resume from checkpoint if available
        
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        
        # Set checkpoint name
        if checkpoint_name is None:
            checkpoint_name = Path(output_filename).stem
        
        # Check for existing checkpoint
        processed_files = []
        if resume:
            processed_files = self.get_processed_files(checkpoint_name)
            if processed_files:
                logger.info(f"Resuming from checkpoint: {len(processed_files)} files already processed")
        
        # Filter out already processed files
        remaining_files = [f for f in fits_files if str(f) not in processed_files]
        
        if not remaining_files:
            logger.info("All files already processed!")
            return {'status': 'complete', 'message': 'All files already processed'}
        
        logger.info(f"Processing {len(remaining_files)} files in batches of {self.batch_size}")
        
        # Process in batches
        all_features = []
        total_batches = (len(remaining_files) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(remaining_files))
            
            batch_files = remaining_files[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end] if labels else None
            
            try:
                # Process batch
                batch_features = self.process_batch(
                    batch_files,
                    batch_labels,
                    batch_start=len(processed_files) + batch_start
                )
                
                all_features.append(batch_features)
                
                # Update checkpoint
                processed_files.extend([str(f) for f in batch_files])
                self.save_checkpoint(checkpoint_name, {
                    'processed_files': processed_files,
                    'last_batch': batch_idx,
                    'total_batches': total_batches,
                    'timestamp': time.time()
                })
                
                logger.debug(f"Completed batch {batch_idx + 1}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Save checkpoint before raising error
                self.save_checkpoint(f"{checkpoint_name}_error", {
                    'processed_files': processed_files,
                    'failed_batch': batch_idx,
                    'error': str(e),
                    'timestamp': time.time()
                })
                raise
        
        # Combine all batches
        if all_features:
            logger.info("Combining all batches...")
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # Save to CSV
            output_path = self.output_dir / output_filename
            combined_features.to_csv(output_path, index=False)
            logger.info(f"Saved combined features to {output_path}")
        else:
            logger.warning("No features extracted")
            combined_features = pd.DataFrame()
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        stats = {
            'status': 'success',
            'n_files_processed': len(remaining_files),
            'n_files_total': len(fits_files),
            'n_samples': len(combined_features),
            'n_features': len(combined_features.columns) - 2,  # Exclude id and label
            'output_file': str(self.output_dir / output_filename),
            'elapsed_time_seconds': elapsed_time,
            'files_per_second': len(remaining_files) / elapsed_time if elapsed_time > 0 else 0,
            'checkpoint_name': checkpoint_name
        }
        
        # Save final checkpoint and stats
        self.save_checkpoint(f"{checkpoint_name}_final", stats)
        
        logger.info(f"Processing complete: {stats['n_files_processed']} files in {elapsed_time:.1f}s")
        logger.info(f"Throughput: {stats['files_per_second']:.2f} files/second")
        
        return stats
    
    def estimate_memory_usage(
        self,
        n_files: int,
        avg_points_per_lc: int = 10000
    ) -> Dict[str, float]:
        """
        Estimate memory usage for processing.
        
        Args:
            n_files: Number of files to process
            avg_points_per_lc: Average points per light curve
        
        Returns:
            Dictionary with memory estimates in GB
        """
        # Rough estimates
        bytes_per_point = 4 + 4 + 4  # time (f64) + flux (f32) + err (f32)
        lc_memory_gb = (n_files * avg_points_per_lc * bytes_per_point) / (1024**3)
        
        # TSFresh features (rough estimate: ~1KB per feature)
        feature_info = self.feature_extractor.get_feature_info()
        n_features = feature_info['n_features']
        features_memory_gb = (n_files * n_features * 8) / (1024**3)  # float64
        
        # Add overhead (2x for processing)
        total_memory_gb = (lc_memory_gb + features_memory_gb) * 2
        
        return {
            'light_curves_gb': lc_memory_gb,
            'features_gb': features_memory_gb,
            'total_estimated_gb': total_memory_gb,
            'recommended_batch_size': max(10, int(16 / total_memory_gb * n_files))  # Assume 16GB available
        }


def process_directory_batch(
    input_dir: Union[str, Path],
    output_filename: str,
    batch_size: int = 100,
    pattern: str = "*.fits",
    max_files: Optional[int] = None,
    labels_file: Optional[Union[str, Path]] = None,
    checkpoint_name: Optional[str] = None,
    **processor_kwargs
) -> Dict:
    """
    Process all FITS files in a directory with batch processing.
    
    Args:
        input_dir: Directory containing FITS files
        output_filename: Output CSV filename
        batch_size: Files per batch
        pattern: Glob pattern for FITS files
        max_files: Maximum files to process (for testing)
        labels_file: Optional CSV with labels
        checkpoint_name: Name for checkpoints
        **processor_kwargs: Additional arguments for BatchProcessor
    
    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_dir)
    
    # Find FITS files
    fits_files = sorted(input_dir.glob(pattern))
    if max_files:
        fits_files = fits_files[:max_files]
    
    logger.info(f"Found {len(fits_files)} FITS files in {input_dir}")
    
    # Load labels if provided
    labels = None
    if labels_file:
        labels_df = pd.read_csv(labels_file)
        label_map = dict(zip(labels_df['id'], labels_df['label']))
        labels = [label_map.get(Path(f).stem, 0) for f in fits_files]
        logger.info(f"Loaded labels for {len(labels)} files")
    
    # Initialize batch processor
    processor = BatchProcessor(
        batch_size=batch_size,
        **processor_kwargs
    )
    
    # Estimate memory usage
    mem_estimate = processor.estimate_memory_usage(len(fits_files))
    logger.info(f"Estimated memory usage: {mem_estimate['total_estimated_gb']:.2f} GB")
    if mem_estimate['total_estimated_gb'] > 32:
        logger.warning(f"High memory usage expected! Consider reducing batch size or using recommended: "
                      f"{mem_estimate['recommended_batch_size']}")
    
    # Process dataset
    stats = processor.process_dataset(
        fits_files,
        output_filename,
        labels=labels,
        checkpoint_name=checkpoint_name
    )
    
    return stats