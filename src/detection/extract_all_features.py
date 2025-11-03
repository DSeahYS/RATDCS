"""
Master Script for Feature Extraction Pipeline

Command-line interface for extracting features from Kepler and TESS light curves
using the complete RATDCS feature extraction pipeline.

Usage:
    python -m src.detection.extract_all_features --dataset kepler --max-files 100
    python -m src.detection.extract_all_features --dataset tess --batch-size 50
    python -m src.detection.extract_all_features --all

Based on RATDCS ARCHITECTURE.md Section 3.1.2 - Exoplanet Candidate Classifier
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import yaml

from .batch_processor import BatchProcessor, process_directory_batch
from .feature_extractor import FeatureExtractor
from .feature_selection import FeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_extraction.log')
    ]
)
logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    """
    Complete feature extraction pipeline orchestrator.
    
    Manages the end-to-end process of extracting features from raw FITS files,
    including preprocessing, feature extraction, selection, and validation.
    
    Attributes:
        config (Dict): Configuration dictionary
        input_dir (Path): Directory with raw FITS files
        output_dir (Path): Directory for processed features
        feature_set (str): TSFresh feature set to use
        batch_size (int): Batch size for processing
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        dataset: str = "kepler",
        feature_set: str = "comprehensive",
        batch_size: int = 100,
        n_jobs: int = 4
    ):
        """
        Initialize the feature extraction pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            dataset: Dataset to process ('kepler' or 'tess')
            feature_set: Feature set to extract ('comprehensive', 'efficient', 'minimal')
            batch_size: Number of files per batch
            n_jobs: Number of parallel jobs
        """
        self.dataset = dataset.lower()
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Set paths
        self.input_dir = Path(f"data/raw/{self.dataset}")
        self.output_dir = Path("data/processed")
        
        logger.info(f"Initialized FeatureExtractionPipeline for {self.dataset}")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'preprocessing': {
                'normalize_method': 'median',
                'outlier_sigma': 5.0,
                'min_valid_points': 100,
                'fill_gaps': False,
                'detrend': True
            },
            'feature_extraction': {
                'n_jobs': self.n_jobs,
                'chunksize': 10,
                'disable_progressbar': False
            },
            'feature_selection': {
                'method': 'tsfresh',
                'correlation_threshold': 0.95
            },
            'batch_processing': {
                'checkpoint_enabled': True,
                'resume': True
            }
        }
    
    def extract_features(
        self,
        max_files: Optional[int] = None,
        labels_file: Optional[str] = None
    ) -> Dict:
        """
        Extract features from FITS files.
        
        Args:
            max_files: Maximum number of files to process (None = all)
            labels_file: Optional CSV file with labels
        
        Returns:
            Dictionary with extraction statistics
        """
        logger.info(f"Starting feature extraction for {self.dataset}...")
        logger.info(f"Feature set: {self.feature_set}")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Determine output filename
        output_filename = f"{self.dataset}_features_{self.feature_set}.csv"
        
        # Process dataset in batches
        stats = process_directory_batch(
            input_dir=self.input_dir,
            output_filename=output_filename,
            batch_size=self.batch_size,
            pattern="*.fits",
            max_files=max_files,
            labels_file=labels_file,
            checkpoint_name=f"{self.dataset}_{self.feature_set}",
            output_dir=self.output_dir,
            feature_set=self.feature_set,
            n_jobs=self.n_jobs,
            **self.config['preprocessing']
        )
        
        return stats
    
    def select_features(
        self,
        features_file: str,
        method: Optional[str] = None
    ) -> Dict:
        """
        Perform feature selection on extracted features.
        
        Args:
            features_file: Path to features CSV file
            method: Selection method (default: from config)
        
        Returns:
            Dictionary with selection statistics
        """
        method = method or self.config['feature_selection']['method']
        
        logger.info(f"Starting feature selection using method: {method}")
        
        # Initialize selector
        selector = FeatureSelector(
            method=method,
            correlation_threshold=self.config['feature_selection']['correlation_threshold']
        )
        
        # Determine output filename
        input_path = self.output_dir / features_file
        output_filename = f"{self.dataset}_features_selected_{method}.csv"
        output_path = self.output_dir / output_filename
        
        # Select features
        stats = selector.select_and_save(
            features_path=input_path,
            output_path=output_path
        )
        
        logger.info(f"Feature selection complete: {stats['n_features_selected']}/{stats['n_features_input']} features")
        
        return stats
    
    def run_full_pipeline(
        self,
        max_files: Optional[int] = None,
        labels_file: Optional[str] = None,
        perform_selection: bool = True
    ) -> Dict:
        """
        Run the complete feature extraction pipeline.
        
        Args:
            max_files: Maximum files to process
            labels_file: Optional labels CSV
            perform_selection: Whether to perform feature selection
        
        Returns:
            Dictionary with pipeline statistics
        """
        start_time = time.time()
        
        # Step 1: Extract features
        logger.info("=" * 60)
        logger.info("STEP 1: Feature Extraction")
        logger.info("=" * 60)
        extraction_stats = self.extract_features(
            max_files=max_files,
            labels_file=labels_file
        )
        
        # Step 2: Feature selection (optional)
        selection_stats = None
        if perform_selection and extraction_stats['status'] == 'success':
            logger.info("=" * 60)
            logger.info("STEP 2: Feature Selection")
            logger.info("=" * 60)
            
            features_file = Path(extraction_stats['output_file']).name
            selection_stats = self.select_features(features_file)
        
        # Compile pipeline statistics
        elapsed_time = time.time() - start_time
        
        pipeline_stats = {
            'dataset': self.dataset,
            'feature_set': self.feature_set,
            'extraction': extraction_stats,
            'selection': selection_stats,
            'total_elapsed_time': elapsed_time,
            'status': 'success'
        }
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info(f"Features extracted: {extraction_stats.get('n_features', 0)}")
        if selection_stats:
            logger.info(f"Features selected: {selection_stats.get('n_features_selected', 0)}")
        
        return pipeline_stats
    
    def get_feature_info(self) -> Dict:
        """
        Get information about extractable features.
        
        Returns:
            Dictionary with feature information
        """
        extractor = FeatureExtractor(
            feature_set=self.feature_set,
            n_jobs=1
        )
        
        info = extractor.get_feature_info()
        logger.info(f"Feature set '{self.feature_set}' contains {info['n_features']} features")
        
        return info


def main():
    """Command-line interface for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract time-series features from Kepler/TESS light curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from Kepler data
  python -m src.detection.extract_all_features --dataset kepler --max-files 100
  
  # Extract features from TESS data with labels
  python -m src.detection.extract_all_features --dataset tess --labels data/tess_labels.csv
  
  # Process both datasets
  python -m src.detection.extract_all_features --all
  
  # Use efficient feature set for faster processing
  python -m src.detection.extract_all_features --dataset kepler --feature-set efficient
  
  # Get feature information
  python -m src.detection.extract_all_features --info --feature-set comprehensive
        """
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['kepler', 'tess'],
        help='Dataset to process (kepler or tess)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process both Kepler and TESS datasets'
    )
    
    # Feature extraction options
    parser.add_argument(
        '--feature-set',
        type=str,
        default='comprehensive',
        choices=['comprehensive', 'efficient', 'minimal'],
        help='Feature set to extract (default: comprehensive = 789 features)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process (for testing)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of files to process per batch (default: 100)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=4,
        help='Number of parallel jobs (default: 4)'
    )
    
    # Data options
    parser.add_argument(
        '--labels',
        type=str,
        help='CSV file with labels (columns: id, label)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    # Feature selection
    parser.add_argument(
        '--no-selection',
        action='store_true',
        help='Skip feature selection step'
    )
    
    parser.add_argument(
        '--selection-method',
        type=str,
        choices=['tsfresh', 'mutual_info', 'f_test', 'random_forest', 'all'],
        help='Feature selection method (default: tsfresh)'
    )
    
    # Information
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display feature information and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle --info flag
    if args.info:
        pipeline = FeatureExtractionPipeline(
            config_path=args.config,
            feature_set=args.feature_set,
            n_jobs=1
        )
        info = pipeline.get_feature_info()
        print(f"\nFeature Set: {info['feature_set']}")
        print(f"Number of Features: {info['n_features']}")
        print(f"\nFirst 10 features:")
        for i, feat in enumerate(info['feature_names'][:10], 1):
            print(f"  {i}. {feat}")
        print(f"  ... and {info['n_features'] - 10} more")
        return
    
    # Validate arguments
    if not args.all and not args.dataset:
        parser.error("Either --dataset or --all must be specified")
    
    # Determine datasets to process
    datasets = ['kepler', 'tess'] if args.all else [args.dataset]
    
    # Process each dataset
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset.upper()} Dataset")
        logger.info(f"{'='*60}\n")
        
        try:
            # Initialize pipeline
            pipeline = FeatureExtractionPipeline(
                config_path=args.config,
                dataset=dataset,
                feature_set=args.feature_set,
                batch_size=args.batch_size,
                n_jobs=args.n_jobs
            )
            
            # Update selection method if specified
            if args.selection_method:
                pipeline.config['feature_selection']['method'] = args.selection_method
            
            # Run pipeline
            results = pipeline.run_full_pipeline(
                max_files=args.max_files,
                labels_file=args.labels,
                perform_selection=not args.no_selection
            )
            
            all_results[dataset] = results
            
        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}", exc_info=True)
            all_results[dataset] = {'status': 'failed', 'error': str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        if results['status'] == 'success':
            print(f"  ✓ Status: Success")
            print(f"  ✓ Files processed: {results['extraction'].get('n_files_processed', 0)}")
            print(f"  ✓ Features extracted: {results['extraction'].get('n_features', 0)}")
            if results.get('selection'):
                print(f"  ✓ Features selected: {results['selection'].get('n_features_selected', 0)}")
            print(f"  ✓ Time: {results['total_elapsed_time']:.1f}s")
        else:
            print(f"  ✗ Status: Failed")
            print(f"  ✗ Error: {results.get('error', 'Unknown error')}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()