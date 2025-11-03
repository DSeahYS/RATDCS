"""
Light Curve Preprocessing for Exoplanet Detection

This module provides preprocessing utilities for Kepler and TESS light curve data,
including FITS file reading, flux normalization, outlier detection, and data quality metrics.

Based on RATDCS ARCHITECTURE.md Section 3.1.2 - Exoplanet Candidate Classifier
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import stats, signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class LightCurvePreprocessor:
    """
    Preprocess light curve data from FITS files.
    
    Handles FITS reading, flux normalization, outlier removal, and data quality assessment
    for both Kepler and TESS light curves.
    
    Attributes:
        normalize_method (str): Normalization method ('median', 'mean', 'minmax')
        outlier_sigma (float): Sigma threshold for outlier detection
        min_valid_points (int): Minimum number of valid data points required
        fill_gaps (bool): Whether to interpolate gaps in time series
    """
    
    def __init__(
        self,
        normalize_method: str = "median",
        outlier_sigma: float = 5.0,
        min_valid_points: int = 100,
        fill_gaps: bool = False
    ):
        """
        Initialize the preprocessor.
        
        Args:
            normalize_method: Method for flux normalization ('median', 'mean', 'minmax')
            outlier_sigma: Sigma threshold for outlier detection (default: 5.0)
            min_valid_points: Minimum valid points required after preprocessing
            fill_gaps: Whether to interpolate missing data points
        """
        self.normalize_method = normalize_method
        self.outlier_sigma = outlier_sigma
        self.min_valid_points = min_valid_points
        self.fill_gaps = fill_gaps
        
        logger.info(f"Initialized LightCurvePreprocessor with {normalize_method} normalization")
    
    def read_fits(self, fits_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Read light curve data from FITS file.
        
        Supports both Kepler (PDCSAP_FLUX) and TESS (SAP_FLUX) formats.
        
        Args:
            fits_path: Path to FITS file
        
        Returns:
            Dictionary with 'time', 'flux', and 'flux_err' arrays
        
        Raises:
            ValueError: If FITS file is invalid or missing required columns
        """
        fits_path = Path(fits_path)
        
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")
        
        try:
            with fits.open(fits_path, memmap=False) as hdul:
                # Get light curve data from first extension
                if len(hdul) < 2:
                    raise ValueError(f"Invalid FITS file: {fits_path}")
                
                data = hdul[1].data
                
                # Try Kepler format first (PDCSAP_FLUX)
                if 'PDCSAP_FLUX' in data.columns.names:
                    time = data['TIME']
                    flux = data['PDCSAP_FLUX']
                    flux_err = data.get('PDCSAP_FLUX_ERR', np.ones_like(flux))
                # Try TESS format (SAP_FLUX or PDCSAP_FLUX)
                elif 'SAP_FLUX' in data.columns.names:
                    time = data['TIME']
                    flux = data['SAP_FLUX']
                    flux_err = data.get('SAP_FLUX_ERR', np.ones_like(flux))
                else:
                    raise ValueError(f"No recognized flux column in {fits_path}")
                
                # Convert to numpy arrays
                time = np.array(time, dtype=np.float64)
                flux = np.array(flux, dtype=np.float32)
                flux_err = np.array(flux_err, dtype=np.float32)
                
                return {
                    'time': time,
                    'flux': flux,
                    'flux_err': flux_err
                }
                
        except Exception as e:
            logger.error(f"Error reading FITS file {fits_path}: {e}")
            raise
    
    def remove_nans(self, light_curve: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Remove NaN and infinite values from light curve.
        
        Args:
            light_curve: Dictionary with 'time', 'flux', 'flux_err'
        
        Returns:
            Cleaned light curve dictionary
        """
        time = light_curve['time']
        flux = light_curve['flux']
        flux_err = light_curve['flux_err']
        
        # Create mask for valid values
        valid_mask = (
            np.isfinite(time) &
            np.isfinite(flux) &
            np.isfinite(flux_err) &
            (flux > 0)  # Physical constraint
        )
        
        n_removed = np.sum(~valid_mask)
        if n_removed > 0:
            logger.debug(f"Removed {n_removed} invalid data points")
        
        return {
            'time': time[valid_mask],
            'flux': flux[valid_mask],
            'flux_err': flux_err[valid_mask]
        }
    
    def remove_outliers(self, light_curve: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Remove outliers using sigma clipping.
        
        Args:
            light_curve: Dictionary with 'time', 'flux', 'flux_err'
        
        Returns:
            Light curve with outliers removed
        """
        flux = light_curve['flux']
        
        # Sigma clipping
        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        sigma = 1.4826 * mad  # Robust standard deviation
        
        outlier_mask = np.abs(flux - median) < self.outlier_sigma * sigma
        
        n_removed = np.sum(~outlier_mask)
        if n_removed > 0:
            logger.debug(f"Removed {n_removed} outliers ({n_removed/len(flux)*100:.1f}%)")
        
        return {
            'time': light_curve['time'][outlier_mask],
            'flux': flux[outlier_mask],
            'flux_err': light_curve['flux_err'][outlier_mask]
        }
    
    def normalize_flux(self, light_curve: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize flux values.
        
        Args:
            light_curve: Dictionary with 'time', 'flux', 'flux_err'
        
        Returns:
            Normalized light curve
        """
        flux = light_curve['flux']
        flux_err = light_curve['flux_err']
        
        if self.normalize_method == "median":
            norm_value = np.median(flux)
            normalized_flux = flux / norm_value
            normalized_err = flux_err / norm_value
            
        elif self.normalize_method == "mean":
            norm_value = np.mean(flux)
            normalized_flux = flux / norm_value
            normalized_err = flux_err / norm_value
            
        elif self.normalize_method == "minmax":
            min_flux = np.min(flux)
            max_flux = np.max(flux)
            normalized_flux = (flux - min_flux) / (max_flux - min_flux)
            normalized_err = flux_err / (max_flux - min_flux)
            
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
        
        logger.debug(f"Normalized flux using {self.normalize_method} method")
        
        return {
            'time': light_curve['time'],
            'flux': normalized_flux.astype(np.float32),
            'flux_err': normalized_err.astype(np.float32)
        }
    
    def interpolate_gaps(
        self,
        light_curve: Dict[str, np.ndarray],
        max_gap: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate small gaps in time series.
        
        Args:
            light_curve: Dictionary with 'time', 'flux', 'flux_err'
            max_gap: Maximum gap size to interpolate (in days)
        
        Returns:
            Light curve with interpolated values
        """
        time = light_curve['time']
        flux = light_curve['flux']
        flux_err = light_curve['flux_err']
        
        # Find gaps
        time_diff = np.diff(time)
        median_cadence = np.median(time_diff)
        gap_mask = time_diff > max_gap
        
        if not np.any(gap_mask):
            return light_curve  # No gaps to fill
        
        # Interpolate
        f_flux = interp1d(time, flux, kind='linear', fill_value='extrapolate')
        f_err = interp1d(time, flux_err, kind='linear', fill_value='extrapolate')
        
        # Create uniform time grid
        time_uniform = np.arange(time[0], time[-1], median_cadence)
        flux_interp = f_flux(time_uniform)
        err_interp = f_err(time_uniform)
        
        logger.debug(f"Interpolated {np.sum(gap_mask)} gaps")
        
        return {
            'time': time_uniform,
            'flux': flux_interp.astype(np.float32),
            'flux_err': err_interp.astype(np.float32)
        }
    
    def detrend_flux(
        self,
        light_curve: Dict[str, np.ndarray],
        window_length: int = 101,
        polyorder: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Remove long-term trends using Savitzky-Golay filter.
        
        Args:
            light_curve: Dictionary with 'time', 'flux', 'flux_err'
            window_length: Window length for Savitzky-Golay filter (must be odd)
            polyorder: Polynomial order for fitting
        
        Returns:
            Detrended light curve
        """
        flux = light_curve['flux']
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Apply Savitzky-Golay filter
        if len(flux) >= window_length:
            trend = signal.savgol_filter(flux, window_length, polyorder)
            detrended_flux = flux - trend + np.median(flux)
        else:
            logger.warning(f"Too few points for detrending ({len(flux)} < {window_length})")
            detrended_flux = flux
        
        return {
            'time': light_curve['time'],
            'flux': detrended_flux.astype(np.float32),
            'flux_err': light_curve['flux_err']
        }
    
    def calculate_quality_metrics(self, light_curve: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate data quality metrics.
        
        Args:
            light_curve: Dictionary with 'time', 'flux', 'flux_err'
        
        Returns:
            Dictionary of quality metrics
        """
        flux = light_curve['flux']
        time = light_curve['time']
        
        metrics = {
            'n_points': len(flux),
            'time_span_days': float(time[-1] - time[0]) if len(time) > 1 else 0.0,
            'mean_flux': float(np.mean(flux)),
            'std_flux': float(np.std(flux)),
            'median_flux': float(np.median(flux)),
            'mad_flux': float(np.median(np.abs(flux - np.median(flux)))),
            'snr': float(np.median(flux) / np.std(flux)) if np.std(flux) > 0 else 0.0,
            'completeness': float(len(flux) / (len(flux) + np.sum(~np.isfinite(flux))))
        }
        
        return metrics
    
    def preprocess(
        self,
        fits_path: Union[str, Path],
        detrend: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            fits_path: Path to FITS file
            detrend: Whether to apply detrending
        
        Returns:
            Tuple of (preprocessed_light_curve, quality_metrics)
        
        Raises:
            ValueError: If preprocessing fails or data quality is insufficient
        """
        try:
            # Read FITS file
            lc = self.read_fits(fits_path)
            
            # Remove NaN values
            lc = self.remove_nans(lc)
            
            # Check minimum data points
            if len(lc['flux']) < self.min_valid_points:
                raise ValueError(
                    f"Insufficient data points: {len(lc['flux'])} < {self.min_valid_points}"
                )
            
            # Remove outliers
            lc = self.remove_outliers(lc)
            
            # Normalize flux
            lc = self.normalize_flux(lc)
            
            # Optional: Fill gaps
            if self.fill_gaps:
                lc = self.interpolate_gaps(lc)
            
            # Optional: Detrend
            if detrend:
                lc = self.detrend_flux(lc)
            
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(lc)
            
            logger.debug(f"Preprocessed {fits_path.name}: {metrics['n_points']} points")
            
            return lc, metrics
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {fits_path}: {e}")
            raise


def preprocess_light_curve_batch(
    fits_files: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    **preprocessor_kwargs
) -> List[Tuple[Path, Dict[str, np.ndarray], Dict[str, float]]]:
    """
    Preprocess multiple light curves in batch.
    
    Args:
        fits_files: List of FITS file paths
        output_dir: Optional directory to save preprocessed data
        **preprocessor_kwargs: Arguments for LightCurvePreprocessor
    
    Returns:
        List of tuples: (fits_path, light_curve, quality_metrics)
    """
    preprocessor = LightCurvePreprocessor(**preprocessor_kwargs)
    results = []
    
    for fits_file in fits_files:
        try:
            lc, metrics = preprocessor.preprocess(fits_file)
            results.append((Path(fits_file), lc, metrics))
            
            # Optionally save preprocessed data
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{Path(fits_file).stem}_preprocessed.npz"
                np.savez(output_file, **lc, **{'metrics_' + k: v for k, v in metrics.items()})
                
        except Exception as e:
            logger.warning(f"Failed to preprocess {fits_file}: {e}")
            continue
    
    logger.info(f"Successfully preprocessed {len(results)}/{len(fits_files)} files")
    return results