# RATDCS Data Acquisition Guide

**Version:** 1.0.0  
**Last Updated:** 2025-11-02  
**Status:** Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Installation & Setup](#installation--setup)
4. [Quick Start](#quick-start)
5. [Downloader Modules](#downloader-modules)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Overview

The RATDCS data acquisition system downloads real astronomical data from three major sources:

- **Kepler/K2**: Exoplanet light curves from MAST archive
- **TESS**: TESS Objects of Interest (TOI) data and stellar parameters
- **ZTF**: Zwicky Transient Facility asteroid survey images

All downloaders feature:
- ✅ Automatic integrity verification
- ✅ Progress tracking with progress bars
- ✅ Comprehensive error handling
- ✅ Network retry logic with exponential backoff
- ✅ Resumable downloads (ZTF)
- ✅ API rate limiting (TESS)
- ✅ Detailed logging

### MVP Requirements

As specified in the architecture:
- **Kepler:** 100+ targets (~3GB)
- **TESS:** 150+ targets (~22GB)
- **ZTF:** 100+ images (~15GB)
- **Total:** ~40GB for MVP dataset

---

## Data Sources

### 1. Kepler/K2 Mission

**Archive:** MAST (Mikulski Archive for Space Telescopes)  
**API:** astroquery.mast  
**Data Format:** FITS time-series  
**Coverage:** 2009-2018 (Kepler), 2014-2018 (K2)

**Key Features:**
- Confirmed exoplanet light curves
- High photometric precision (20 ppm)
- 30-minute cadence (long cadence)
- Pre-processed PDCSAP flux

### 2. TESS Mission

**Archive:** MAST / TIC (TESS Input Catalog)  
**API:** astroquery.mast + REST API  
**Data Format:** FITS light curves + JSON metadata  
**Coverage:** 2018-present (ongoing)

**Key Features:**
- TOI (TESS Objects of Interest) candidates
- Full-frame images and target pixel files
- 2-minute and 20-second cadences
- Stellar parameter catalog

### 3. ZTF Survey

**Archive:** IRSA (NASA/IPAC Infrared Science Archive)  
**API:** IRSA IBE (Image-Based Exploration)  
**Data Format:** FITS imaging  
**Coverage:** 2018-present (ongoing)

**Key Features:**
- Wide-field survey (47 sq deg per exposure)
- 3-band photometry (g, r, i)
- Sub-arcsecond resolution
- Nightly cadence for northern sky

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
cd RATDCS
pip install -r requirements.txt
```

### Required Packages

The data acquisition modules require:

```python
astropy>=6.0.0          # FITS file handling
astroquery>=0.4.6       # MAST archive queries
requests>=2.31.0        # HTTP requests
tqdm>=4.66.0            # Progress bars
pyyaml>=6.0.1           # Configuration
pandas>=2.1.0           # Data manipulation
numpy>=1.26.0           # Numerical operations
```

These are already included in [`requirements.txt`](../requirements.txt).

### Verify Installation

```python
# Test imports
from src.data import KeplerDownloader, TESSDownloader, ZTFDownloader

print("✓ All downloaders imported successfully")
```

---

## Quick Start

### Download All Datasets (MVP Scale)

```bash
cd RATDCS

# Download all datasets with default settings
python -m src.data.download_all_data --all

# This will download:
# - 100 Kepler targets (~3GB)
# - 150 TESS targets (~22GB)
# - 100 ZTF images (~15GB)
# Total: ~40GB
```

### Download Individual Datasets

```bash
# Kepler only
python -m src.data.download_all_data --kepler --kepler-targets 100

# TESS only
python -m src.data.download_all_data --tess --tess-targets 150

# ZTF only (specific date range)
python -m src.data.download_all_data --ztf \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --ztf-max-images 100
```

---

## Downloader Modules

### 1. Kepler Downloader

**Module:** `src.data.kepler_downloader.py`  
**Class:** `KeplerDownloader`

#### Features

- Queries confirmed exoplanets from MAST
- Downloads FITS time-series data
- Verifies FITS integrity (column checks, NaN detection)
- Removes corrupted files automatically
- Supports both Kepler and K2 missions

#### Usage

```python
from src.data import KeplerDownloader

# Initialize downloader
downloader = KeplerDownloader(
    output_dir="data/raw/kepler",
    verify_fits=True,
    max_retries=3
)

# Download light curves
successful, failed = downloader.download_light_curves(
    max_targets=100,
    mission="Kepler"  # or "K2"
)

# Get statistics
stats = downloader.get_download_stats()
print(f"Downloaded {stats['total_files']} files")
print(f"Total size: {stats['total_size_gb']:.2f} GB")
```

#### CLI Usage

```bash
python -m src.data.kepler_downloader \
    --max-targets 100 \
    --output data/raw/kepler \
    --mission Kepler
```

#### Output Structure

```
data/raw/kepler/
├── kplr001234567-2009123456789_llc.fits
├── kplr002345678-2009234567890_llc.fits
└── ...
```

### 2. TESS Downloader

**Module:** `src.data.tess_downloader.py`  
**Class:** `TESSDownloader`

#### Features

- Queries TOI catalog from TIC
- Downloads light curve FITS files
- Extracts and saves stellar parameters
- API rate limiting (default: 0.5s delay)
- Retry logic with exponential backoff

#### Usage

```python
from src.data import TESSDownloader

# Initialize downloader
downloader = TESSDownloader(
    output_dir="data/raw/tess",
    rate_limit_delay=0.5,
    max_retries=3
)

# Download TOI data
successful, failed = downloader.download_toi_data(
    max_targets=150,
    download_fits=True
)

# Get statistics
stats = downloader.get_download_stats()
print(f"FITS files: {stats['total_fits_files']}")
print(f"Metadata files: {stats['total_metadata_files']}")
print(f"Total size: {stats['total_size_gb']:.2f} GB")
```

#### CLI Usage

```bash
python -m src.data.tess_downloader \
    --max-targets 150 \
    --output data/raw/tess \
    --rate-limit 0.5
```

#### Output Structure

```
data/raw/tess/
├── fits/
│   ├── tess12345678-s0001-1-1-0120-s_lc.fits
│   └── ...
└── metadata/
    ├── TIC12345678_metadata.json
    └── ...
```

#### Metadata Schema

```json
{
  "tic_id": "TIC12345678",
  "stellar_parameters": {
    "teff": 5778.0,
    "logg": 4.44,
    "radius": 1.0,
    "mass": 1.0,
    "tmag": 10.5,
    "distance": 100.0
  },
  "observations": [...],
  "download_timestamp": "2025-11-02T14:30:00Z"
}
```

### 3. ZTF Downloader

**Module:** `src.data.ztf_downloader.py`  
**Class:** `ZTFDownloader`

#### Features

- Queries ZTF public data releases
- Downloads FITS imaging data
- MD5 checksum verification
- Resumable downloads for large files
- Supports date range filtering
- Field ID filtering

#### Usage

```python
from src.data import ZTFDownloader

# Initialize downloader
downloader = ZTFDownloader(
    output_dir="data/raw/ztf",
    chunk_size=8192*1024,  # 8MB chunks
    verify_checksum=True,
    resume_downloads=True
)

# Check disk space first
required_gb = 15.0
if not downloader.check_disk_space(required_gb):
    print("Insufficient disk space!")
    exit(1)

# Download images
successful, failed = downloader.download_images(
    start_date="2024-01-01",
    end_date="2024-01-31",
    field_ids=[100, 200, 300],
    max_images=100
)

# Get statistics
stats = downloader.get_download_stats()
print(f"Total images: {stats['total_images']}")
print(f"Total size: {stats['total_size_gb']:.2f} GB")
```

#### CLI Usage

```bash
python -m src.data.ztf_downloader \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --field-ids 100 200 300 \
    --max-images 100 \
    --output data/raw/ztf
```

#### Output Structure

```
data/raw/ztf/
├── images/
│   ├── ztf_20240101_field000100_ccd01_sci.fits
│   ├── ztf_20240101_field000100_ccd02_sci.fits
│   └── ...
├── metadata/
└── temp/  # Temporary files (auto-cleaned)
```

---

## Usage Examples

### Example 1: Basic Download Pipeline

```python
from src.data import KeplerDownloader, TESSDownloader, ZTFDownloader

# Download Kepler data
print("Downloading Kepler data...")
kepler = KeplerDownloader(output_dir="data/raw/kepler")
k_success, k_fail = kepler.download_light_curves(max_targets=100)
print(f"Kepler: {k_success} successful, {k_fail} failed")

# Download TESS data
print("\nDownloading TESS data...")
tess = TESSDownloader(output_dir="data/raw/tess")
t_success, t_fail = tess.download_toi_data(max_targets=150)
print(f"TESS: {t_success} successful, {t_fail} failed")

# Download ZTF data
print("\nDownloading ZTF data...")
ztf = ZTFDownloader(output_dir="data/raw/ztf")
z_success, z_fail = ztf.download_images(
    start_date="2024-01-01",
    end_date="2024-01-31",
    max_images=100
)
print(f"ZTF: {z_success} successful, {z_fail} failed")
```

### Example 2: Using the Orchestrator

```python
from src.data.download_all_data import DataDownloadOrchestrator

# Initialize orchestrator
orchestrator = DataDownloadOrchestrator(
    base_dir="data/raw",
    config_path="config/default.yaml"
)

# Check disk space
if not orchestrator.check_disk_space(required_gb=50.0):
    print("Insufficient disk space!")
    exit(1)

# Download all datasets
results = orchestrator.download_all(
    kepler_targets=100,
    tess_targets=150,
    ztf_start_date="2024-01-01",
    ztf_end_date="2024-01-31",
    ztf_max_images=100
)

# Print summary
print("\n=== Download Summary ===")
print(f"Total successful: {results['summary']['total_successful']}")
print(f"Total failed: {results['summary']['total_failed']}")
print(f"Total size: {results['summary']['total_size_gb']:.2f} GB")
print(f"Total time: {results['summary']['total_elapsed_time']:.1f} seconds")
```

### Example 3: Incremental Downloads

```python
from src.data import ZTFDownloader

# Download 1 month at a time
downloader = ZTFDownloader(output_dir="data/raw/ztf")

months = [
    ("2024-01-01", "2024-01-31"),
    ("2024-02-01", "2024-02-29"),
    ("2024-03-01", "2024-03-31"),
]

for start_date, end_date in months:
    print(f"\nDownloading {start_date} to {end_date}...")
    success, failed = downloader.download_images(
        start_date=start_date,
        end_date=end_date,
        max_images=30
    )
    print(f"Month complete: {success} successful, {failed} failed")
```

---

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Data directories
RATDCS_DATA_DIR=data/raw

# Download settings
KEPLER_MAX_TARGETS=100
TESS_MAX_TARGETS=150
ZTF_MAX_IMAGES=100

# Network settings
MAX_RETRIES=3
RATE_LIMIT_DELAY=0.5
CHUNK_SIZE=8388608  # 8MB
```

### YAML Configuration

The downloaders read from `config/default.yaml`:

```yaml
data_pipeline:
  preprocessing:
    normalize: true
    cache_enabled: true
    prefetch_buffer_size: 10
  
  validation:
    split_ratio: [0.8, 0.1, 0.1]  # train, val, test
    shuffle_buffer_size: 10000
    random_seed: 42
```

---

## Testing

### Run Unit Tests

```bash
# Run all data acquisition tests
pytest tests/unit/data/ -v

# Run specific downloader tests
pytest tests/unit/data/test_kepler_downloader.py -v
pytest tests/unit/data/test_tess_downloader.py -v
pytest tests/unit/data/test_ztf_downloader.py -v

# Run with coverage
pytest tests/unit/data/ --cov=src.data --cov-report=html
```

### Mock Testing

All unit tests use mocked API calls to avoid actual downloads:

```python
@patch('src.data.kepler_downloader.Observations.query_criteria')
def test_query_returns_observations(mock_query, downloader):
    mock_table = Table({'obsid': ['obs1', 'obs2']})
    mock_query.return_value = mock_table
    
    observations = downloader.query_confirmed_exoplanets(max_targets=2)
    assert len(observations) == 2
```

---

## Troubleshooting

### Common Issues

#### 1. Network Timeouts

**Symptoms:** Downloads fail with timeout errors

**Solutions:**
```python
# Increase retry count
downloader = KEplerDownloader(max_retries=5)

# For TESS, increase rate limit delay
downloader = TESSDownloader(rate_limit_delay=1.0)
```

#### 2. Disk Space Errors

**Symptoms:** "No space left on device"

**Solutions:**
```bash
# Check available space
df -h

# Use different directory
python -m src.data.download_all_data --all --base-dir /mnt/external

# Clean temp files
rm -rf data/raw/*/temp data/raw/*/cache
```

#### 3. Corrupted FITS Files

**Symptoms:** Verification failures, read errors

**Solutions:**
- Corrupted files are automatically removed and logged
- Check `data_download.log` for details
- Re-run download to retry failed files

#### 4. API Rate Limiting

**Symptoms:** "429 Too Many Requests"

**Solutions:**
```python
# Increase delay between requests
downloader = TESSDownloader(rate_limit_delay=2.0)

# Reduce batch size
downloader.download_toi_data(max_targets=50)  # Smaller batches
```

### Debugging

Enable DEBUG logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run downloaders
downloader = KeplerDownloader(...)
```

Check logs:

```bash
# View download log
tail -f data_download.log

# Search for errors
grep "ERROR" data_download.log

# Check specific target
grep "TIC12345" data_download.log
```

---

## Best Practices

### 1. Start Small

Begin with small downloads to verify setup:

```bash
# Test with 10 targets first
python -m src.data.download_all_data --kepler --kepler-targets 10
```

### 2. Monitor Disk Space

```python
import shutil

stat = shutil.disk_usage("data/raw")
available_gb = stat.free / (1024**3)
print(f"Available: {available_gb:.2f} GB")
```

### 3. Use Logging

Always check logs after downloads:

```bash
# Summary of results
grep "Download complete" data_download.log

# Check for failures
grep "failed" data_download.log
```

### 4. Incremental Downloads

For large datasets, download incrementally:

```python
# Download in batches
for i in range(0, 500, 100):
    downloader.download_light_curves(max_targets=100)
    print(f"Completed batch {i//100 + 1}")
```

### 5. Verify Data Integrity

After downloads, verify data:

```python
from astropy.io import fits

# Check random files
import glob
import random

files = glob.glob("data/raw/kepler/*.fits")
sample = random.sample(files, min(10, len(files)))

for f in sample:
    with fits.open(f) as hdul:
        print(f"✓ {f}: {len(hdul)} HDUs")
```

---

## Data Volume Expectations

### MVP Scale (Development/Testing)

| Dataset | Count | Size | Download Time* |
|---------|-------|------|----------------|
| Kepler  | 100   | 3 GB | ~40 min |
| TESS    | 150   | 22 GB | ~5 hours |
| ZTF     | 100   | 15 GB | ~3.5 hours |
| **Total** | **350** | **40 GB** | **~9 hours** |

*Assuming 10 Mbps connection

### Full Scale (Production)

| Dataset | Count | Size | Notes |
|---------|-------|------|-------|
| Kepler  | 1000  | 30 GB | All confirmed exoplanets |
| TESS    | 1000  | 150 GB | Full TOI catalog |
| ZTF     | yearly | 300 GB | One year of observations |
| **Total** | **2000+** | **480 GB** | Full operational dataset |

---

## Performance Optimization

### Parallel Downloads

Use multiple terminals for different datasets:

```bash
# Terminal 1
python -m src.data.kepler_downloader --max-targets 100

# Terminal 2
python -m src.data.tess_downloader --max-targets 150

# Terminal 3
python -m src.data.ztf_downloader --start-date 2024-01-01 --end-date 2024-01-31
```

### Network Optimization

```python
# Increase chunk size for faster downloads
downloader = ZTFDownloader(chunk_size=16*1024*1024)  # 16MB

# Reduce verification overhead
downloader = KeplerDownloader(verify_fits=False)  # Not recommended
```

---

## Contact & Support

- **Documentation:** This file + [`data/raw/README.md`](../data/raw/README.md)
- **Logs:** Check `data_download.log` for detailed information
- **Issues:** Report on GitHub Issues
- **Architecture:** See [`ARCHITECTURE.md`](ARCHITECTURE.md) for system overview

---

**Last Updated:** 2025-11-02  
**RATDCS Version:** 1.0.0  
**Authors:** RATDCS Team