# RATDCS Raw Data Directory

This directory contains raw astronomical data downloaded from various archives for use in RATDCS detection and classification pipelines.

## Directory Structure

```
data/raw/
├── kepler/           # Kepler/K2 exoplanet light curves
│   ├── *.fits        # FITS time-series light curve files
│   └── cache/        # Temporary download cache (auto-cleaned)
│
├── tess/             # TESS exoplanet data
│   ├── fits/         # TESS light curve FITS files
│   │   └── *.fits    
│   └── metadata/     # TOI stellar parameters (JSON)
│       └── *_metadata.json
│
└── ztf/              # ZTF asteroid survey images
    ├── images/       # ZTF FITS images
    │   └── *.fits
    ├── metadata/     # Image metadata
    └── temp/         # Temporary download files (auto-cleaned)
```

## Data Sources

### Kepler Light Curves (`kepler/`)

**Source:** MAST Archive (Mikulski Archive for Space Telescopes)  
**Data Type:** FITS time-series files  
**Content:** Exoplanet transit light curves from Kepler/K2 missions  
**Expected Volume:** ~3GB for 100 targets (30MB per target)

**File Naming Convention:**
```
kplr<KIC_ID>-<timestamp>_llc.fits
```

**Data Structure:**
- HDU 0: Primary header with observation metadata
- HDU 1: Binary table with columns:
  - `TIME`: Barycentric Julian Date
  - `PDCSAP_FLUX`: Pre-search Data Conditioning flux
  - `PDCSAP_FLUX_ERR`: Flux uncertainty
  - `SAP_QUALITY`: Quality flags

**Usage:**
```python
from src.data import KeplerDownloader

downloader = KeplerDownloader(output_dir="data/raw/kepler")
downloader.download_light_curves(max_targets=100)
```

### TESS Data (`tess/`)

**Source:** MAST Archive / TESS Input Catalog (TIC)  
**Data Type:** FITS light curves + JSON metadata  
**Content:** TESS Objects of Interest (TOI) light curves and stellar parameters  
**Expected Volume:** ~22GB for 150 targets (150MB per target)

**File Naming Convention:**
```
FITS: tess<TIC_ID>-s<sector>-<timestamp>_lc.fits
JSON: TIC<id>_metadata.json
```

**Metadata Structure (JSON):**
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
  "observations": [
    {
      "obs_id": "...",
      "sector": 1,
      "s_ra": 290.5,
      "s_dec": 45.2
    }
  ],
  "download_timestamp": "2025-11-02T14:30:00Z"
}
```

**Usage:**
```python
from src.data import TESSDownloader

downloader = TESSDownloader(output_dir="data/raw/tess")
downloader.download_toi_data(max_targets=150)
```

### ZTF Images (`ztf/`)

**Source:** ZTF Public Data Releases (IRSA/Caltech)  
**Data Type:** FITS imaging data  
**Content:** Wide-field survey images for asteroid detection  
**Expected Volume:** ~15GB for 100 images (150MB per image)  
**Annual Capacity:** ~300GB per year of observations

**File Naming Convention:**
```
ztf_<YYYYMMDD>_field<field_id>_ccd<ccd_id>_sci.fits
```

**Image Properties:**
- **Resolution:** 2048×2048 pixels (typical)
- **Field of View:** ~47 square degrees
- **Filters:** g, r, i bands
- **Cadence:** Multiple exposures per night

**Usage:**
```python
from src.data import ZTFDownloader

downloader = ZTFDownloader(output_dir="data/raw/ztf")
downloader.download_images(
    start_date="2024-01-01",
    end_date="2024-01-31",
    max_images=100
)
```

## Data Validation

All downloaders include integrity verification:

- **Kepler:** FITS file structure validation, column presence checks, NaN detection
- **TESS:** JSON schema validation, FITS integrity checks
- **ZTF:** MD5 checksum verification (when available), FITS header validation

Corrupted or incomplete files are automatically removed and logged.

## Download All Data

To download all datasets at once:

```bash
cd RATDCS
python -m src.data.download_all_data --all
```

Or download specific datasets:

```bash
# Kepler only
python -m src.data.download_all_data --kepler --kepler-targets 100

# TESS only
python -m src.data.download_all_data --tess --tess-targets 150

# ZTF only
python -m src.data.download_all_data --ztf --start-date 2024-01-01 --end-date 2024-01-31
```

## Disk Space Requirements

| Dataset | Targets/Images | Expected Size | Full Dataset |
|---------|----------------|---------------|--------------|
| Kepler  | 100 targets    | ~3 GB         | ~30 GB (1000 targets) |
| TESS    | 150 targets    | ~22 GB        | ~150 GB (1000 TOIs) |
| ZTF     | 100 images     | ~15 GB        | ~300 GB (yearly) |
| **Total** | **MVP Scale** | **~40 GB**   | **~480 GB (full scale)** |

**Recommendation:** Ensure at least 60GB of free disk space before starting downloads (includes 50% buffer for temporary files).

## Performance Considerations

### Download Times (Estimated)

Assuming 10 Mbps connection:
- Kepler (100 targets): ~40 minutes
- TESS (150 targets): ~5 hours
- ZTF (100 images): ~3.5 hours

### Optimization Tips

1. **Parallel Downloads:** Use multiple terminal windows to download different datasets simultaneously
2. **Resume Support:** ZTF downloader supports resuming interrupted downloads
3. **Rate Limiting:** TESS downloader includes API rate limiting (configurable)
4. **Disk I/O:** Use SSD storage for faster FITS file operations

## Data Retention Policy

- **Raw Data:** Retained permanently for reproducibility
- **Temporary Files:** Auto-cleaned after successful downloads
- **Cache Directories:** Cleared automatically; can be manually cleaned if needed

## Troubleshooting

### Common Issues

**Issue:** "Insufficient disk space"  
**Solution:** Free up space or use `--base-dir` to specify different location

**Issue:** "Network timeout"  
**Solution:** Increase `max_retries` in downloader initialization or check network connection

**Issue:** "Corrupted FITS file"  
**Solution:** Files are automatically re-downloaded. Check logs in `data_download.log`

**Issue:** "API rate limit exceeded"  
**Solution:** Increase `rate_limit_delay` for TESS downloader (default: 0.5s)

### Logs

Download logs are stored in:
- Console output (INFO level)
- `data_download.log` (DEBUG level with full details)

## Data Access Examples

### Reading Kepler Light Curves

```python
from astropy.io import fits
import numpy as np

# Read FITS file
with fits.open('data/raw/kepler/kplr001234567_llc.fits') as hdul:
    data = hdul[1].data
    time = data['TIME']
    flux = data['PDCSAP_FLUX']
    
    # Remove NaN values
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time_clean = time[mask]
    flux_clean = flux[mask]
```

### Reading TESS Metadata

```python
import json

# Read metadata
with open('data/raw/tess/metadata/TIC12345_metadata.json') as f:
    metadata = json.load(f)
    
    teff = metadata['stellar_parameters']['teff']
    radius = metadata['stellar_parameters']['radius']
    sectors = [obs['sector'] for obs in metadata['observations']]
```

### Reading ZTF Images

```python
from astropy.io import fits

# Read ZTF FITS image
with fits.open('data/raw/ztf/images/ztf_20240101_field100_ccd01_sci.fits') as hdul:
    image_data = hdul[0].data
    header = hdul[0].header
    
    # Extract WCS information
    ra_center = header['CRVAL1']
    dec_center = header['CRVAL2']
```

## Data Citation

When using this data in publications, please cite the original sources:

- **Kepler:** Thompson et al. (2016), Kepler Data Release Notes
- **TESS:** Ricker et al. (2015), JATIS, 1, 014003
- **ZTF:** Bellm et al. (2019), PASP, 131, 018002

## Contact & Support

For data-related issues:
- Check logs first: `data_download.log`
- Review docs: `docs/DATA_ACQUISITION.md`
- Report issues: GitHub Issues

---

**Last Updated:** 2025-11-02  
**RATDCS Version:** 1.0.0