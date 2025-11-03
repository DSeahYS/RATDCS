"""
TSFresh Feature Extraction for Exoplanet Detection

This module extracts 789 time-series features from Kepler and TESS light curves using TSFresh,
matching the methodology from Malik et al. (2022) that achieved 0.948 AUC.

Based on RATDCS ARCHITECTURE.md Section 3.1.2 - Exoplanet Candidate Classifier

Reference:
    Malik et al. (2022) - Deep Transfer Learning for Exoplanet Classification
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from .preprocess import LightCurvePreprocessor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract time-series features from light curves using TSFresh.
    
    Extracts 789 features including statistical, spectral, time-domain,
    and complexity features for exoplanet detection.
    
    Attributes:
        feature_set (str): Feature set to use ('comprehensive', 'efficient', 'minimal')
        n_jobs (int): Number of parallel jobs for feature extraction
        chunksize (int): Chunk size for parallel processing
        disable_progressbar (bool): Whether to disable progress bar
        preprocessor (LightCurvePreprocessor): Light curve preprocessor instance
    """
    
    # Feature set configurations
    FEATURE_SETS = {
        'comprehensive': ComprehensiveFCParameters(),
        'efficient': EfficientFCParameters(),
        'minimal': {
            'mean': None,
            'median': None,
            'std': None,
            'variance': None,
            'minimum': None,
            'maximum': None,
            'length': None,
            'sum_values': None,
            'abs_energy': None,
            'mean_abs_change': None,
            'mean_change': None,
            'mean_second_derivative_central': None,
            'quantile': [{'q': 0.1}, {'q': 0.9}],
            'autocorrelation': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
            'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40}],
            'partial_autocorrelation': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
            'fft_coefficient': [{'coeff': 0}, {'coeff': 1}, {'coeff': 2}],
            'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}],
            'ar_coefficient': [{'coeff': 0, 'k': 10}, {'coeff': 1, 'k': 10}],
            'count_above_mean': None,
            'count_below_mean': None,
            'sample_entropy': None,
            'approximate_entropy': [{'m': 2, 'r': 0.2}],
            'cid_ce': [{'normalize': True}],
            'symmetry_looking': [{'r': 0.2}],
        }
    }
    
    def __init__(
        self,
        feature_set: str = 'comprehensive',
        n_jobs: int = 4,
        chunksize: int = 10,
        disable_progressbar: bool = False,
        normalize_method: str = "median",
        outlier_sigma: float = 5.0
    ):
        """
        Initialize the feature extractor.
        
        Args:
            feature_set: Feature set to use ('comprehensive' for 789 features,
                        'efficient' for faster extraction, 'minimal' for testing)
            n_jobs: Number of parallel jobs (default: 4)
            chunksize: Chunk size for parallel processing
            disable_progressbar: Disable tqdm progress bar
            normalize_method: Flux normalization method
            outlier_sigma: Sigma threshold for outlier detection
        """
        if feature_set not in self.FEATURE_SETS:
            raise ValueError(f"Unknown feature_set: {feature_set}. "
                           f"Choose from: {list(self.FEATURE_SETS.keys())}")
        
        self.feature_set = feature_set
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.disable_progressbar = disable_progressbar
        
        # Initialize preprocessor
        self.preprocessor = LightCurvePreprocessor(
            normalize_method=normalize_method,
            outlier_sigma=outlier_sigma
        )
        
        logger.info(f"Initialized FeatureExtractor with '{feature_set}' feature set")
        logger.info(f"Parallel processing: {n_jobs} jobs, chunk size: {chunksize}")
    
    def prepare_dataframe(
        self,
        light_curves: List[Dict[str, np.ndarray]],
        ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare time series dataframe in TSFresh format.
        
        TSFresh requires data in long format with columns: id, time, flux
        
        Args:
            light_curves: List of light curve dictionaries with 'time' and 'flux'
            ids: Optional list of identifiers for each light curve
        
        Returns:
            DataFrame in TSFresh format with columns: id, time, flux
        """
        if ids is None:
            ids = [f"lc_{i}" for i in range(len(light_curves))]
        
        if len(ids) != len(light_curves):
            raise ValueError(f"Length mismatch: {len(ids)} ids vs {len(light_curves)} light curves")
        
        # Build long-format dataframe
        rows = []
        for lc_id, lc in zip(ids, light_curves):
            for t, f in zip(lc['time'], lc['flux']):
                rows.append({
                    'id': lc_id,
                    'time': t,
                    'flux': f
                })
        
        df = pd.DataFrame(rows)
        logger.debug(f"Prepared TSFresh dataframe: {len(df)} rows, {len(light_curves)} time series")
        
        return df
    
    def extract(
        self,
        light_curves: List[Dict[str, np.ndarray]],
        ids: Optional[List[str]] = None,
        column_id: str = 'id',
        column_sort: str = 'time',
        column_value: str = 'flux'
    ) -> pd.DataFrame:
        """
        Extract features from light curves using TSFresh.
        
        Args:
            light_curves: List of preprocessed light curve dictionaries
            ids: Optional identifiers for each light curve
            column_id: Name of ID column (default: 'id')
            column_sort: Name of time column (default: 'time')
            column_value: Name of flux column (default: 'flux')
        
        Returns:
            DataFrame with extracted features (rows=light curves, columns=features)
        """
        # Prepare data in TSFresh format
        df = self.prepare_dataframe(light_curves, ids)
        
        # Get feature calculation settings
        fc_parameters = self.FEATURE_SETS[self.feature_set]
        
        logger.info(f"Extracting features from {len(light_curves)} light curves...")
        
        # Extract features
        features = extract_features(
            df,
            column_id=column_id,
            column_sort=column_sort,
            column_value=column_value,
            default_fc_parameters=fc_parameters,
            n_jobs=self.n_jobs,
            chunksize=self.chunksize,
            disable_progressbar=self.disable_progressbar,
            impute_function=impute
        )
        
        logger.info(f"Extracted {features.shape[1]} features for {features.shape[0]} light curves")
        
        return features
    
    def extract_from_fits(
        self,
        fits_files: List[Union[str, Path]],
        labels: Optional[List[int]] = None,
        preprocess: bool = True
    ) -> Tuple[pd.DataFrame, List[str], Optional[List[int]]]:
        """
        Extract features directly from FITS files.
        
        Args:
            fits_files: List of FITS file paths
            labels: Optional list of classification labels (1=exoplanet, 0=non-exoplanet)
            preprocess: Whether to apply preprocessing
        
        Returns:
            Tuple of (features_df, valid_ids, valid_labels)
        """
        light_curves = []
        valid_ids = []
        valid_labels = [] if labels is not None else None
        
        logger.info(f"Processing {len(fits_files)} FITS files...")
        
        for i, fits_file in enumerate(tqdm(fits_files, desc="Reading FITS files")):
            try:
                if preprocess:
                    lc, metrics = self.preprocessor.preprocess(fits_file)
                else:
                    lc = self.preprocessor.read_fits(fits_file)
                    lc = self.preprocessor.remove_nans(lc)
                
                light_curves.append(lc)
                valid_ids.append(Path(fits_file).stem)
                
                if labels is not None:
                    valid_labels.append(labels[i])
                    
            except Exception as e:
                logger.warning(f"Failed to process {fits_file}: {e}")
                continue
        
        if not light_curves:
            raise ValueError("No valid light curves found")
        
        logger.info(f"Successfully loaded {len(light_curves)} light curves")
        
        # Extract features
        features = self.extract(light_curves, ids=valid_ids)
        
        return features, valid_ids, valid_labels
    
    def extract_and_save(
        self,
        fits_files: List[Union[str, Path]],
        output_path: Union[str, Path],
        labels: Optional[List[int]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Extract features and save to CSV.
        
        Args:
            fits_files: List of FITS file paths
            output_path: Output CSV file path
            labels: Optional classification labels
            metadata: Optional metadata dictionary to save
        
        Returns:
            Dictionary with extraction statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract features
        features, valid_ids, valid_labels = self.extract_from_fits(
            fits_files,
            labels=labels
        )
        
        # Add labels if provided
        if valid_labels is not None:
            features['label'] = valid_labels
        
        # Add IDs as a column
        features['id'] = valid_ids
        
        # Reorder columns to put id and label first
        cols = ['id']
        if 'label' in features.columns:
            cols.append('label')
        cols.extend([c for c in features.columns if c not in ['id', 'label']])
        features = features[cols]
        
        # Save to CSV
        features.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        
        # Compute statistics
        stats = {
            'n_samples': len(features),
            'n_features': len(features.columns) - (2 if 'label' in features.columns else 1),
            'feature_names': [c for c in features.columns if c not in ['id', 'label']],
            'output_file': str(output_path),
            'feature_set': self.feature_set
        }
        
        if valid_labels is not None:
            stats['n_positive'] = sum(valid_labels)
            stats['n_negative'] = len(valid_labels) - sum(valid_labels)
            stats['class_balance'] = sum(valid_labels) / len(valid_labels)
        
        return stats
    
    def get_feature_info(self) -> Dict[str, int]:
        """
        Get information about the feature set.
        
        Returns:
            Dictionary with feature set statistics
        """
        # Create a dummy time series to extract features
        dummy_lc = {
            'time': np.arange(100),
            'flux': np.random.randn(100)
        }
        
        df = self.prepare_dataframe([dummy_lc], ids=['dummy'])
        fc_parameters = self.FEATURE_SETS[self.feature_set]
        
        features = extract_features(
            df,
            column_id='id',
            column_sort='time',
            column_value='flux',
            default_fc_parameters=fc_parameters,
            disable_progressbar=True,
            n_jobs=1
        )
        
        return {
            'feature_set': self.feature_set,
            'n_features': features.shape[1],
            'feature_names': list(features.columns)
        }


def extract_features_batch(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    labels_file: Optional[Union[str, Path]] = None,
    pattern: str = "*.fits",
    max_files: Optional[int] = None,
    **extractor_kwargs
) -> Dict[str, any]:
    """
    Extract features from all FITS files in a directory.
    
    Args:
        input_dir: Directory containing FITS files
        output_path: Output CSV file path
        labels_file: Optional CSV file with columns 'id' and 'label'
        pattern: Glob pattern for FITS files (default: '*.fits')
        max_files: Maximum number of files to process
        **extractor_kwargs: Arguments for FeatureExtractor
    
    Returns:
        Dictionary with extraction statistics
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
        # Create mapping from filename to label
        label_map = dict(zip(labels_df['id'], labels_df['label']))
        labels = [label_map.get(Path(f).stem, 0) for f in fits_files]
        logger.info(f"Loaded labels for {len(labels)} files")
    
    # Initialize extractor
    extractor = FeatureExtractor(**extractor_kwargs)
    
    # Extract and save features
    metadata = {
        'input_dir': str(input_dir),
        'n_files': len(fits_files),
        'pattern': pattern,
        'feature_set': extractor.feature_set,
        'extractor_config': {
            'n_jobs': extractor.n_jobs,
            'chunksize': extractor.chunksize
        }
    }
    
    stats = extractor.extract_and_save(
        fits_files,
        output_path,
        labels=labels,
        metadata=metadata
    )
    
    return stats