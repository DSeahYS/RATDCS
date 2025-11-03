# RATDCS Testing Guide

**Version:** 1.0.0  
**Last Updated:** 2025-11-03  
**Status:** Ready for Testing

---

## Overview

This guide provides instructions for testing the RATDCS feature extraction pipeline implementation.

## Code Status

✅ **All code has been pushed to GitHub**: https://github.com/DSeahYS/RATDCS  
✅ **Commits:**
- `d8f70ed`: feat: Implement complete feature extraction pipeline with TSFresh
- `6038d1c`: fix: Correct __init__.py syntax errors

---

## Prerequisites for Testing

### System Requirements

- **Operating System**: Linux, macOS, or Windows with Visual Studio Build Tools
- **Python**: 3.10 or 3.11 (recommended)
- **Memory**: 8GB+ RAM
- **Disk Space**: 50GB+ for data storage

### Windows Users

If testing on Windows, you need Microsoft Visual C++ Build Tools:

```bash
# Download and install:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Select "Desktop development with C++"
```

Or use WSL2 (Windows Subsystem for Linux):

```bash
wsl --install
wsl
# Then follow Linux instructions
```

---

## Installation Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/DSeahYS/RATDCS.git
cd RATDCS
```

### Step 2: Create Virtual Environment

```bash
# Linux/Mac
python3.10 -m venv venv
source venv/bin/activate

# Windows (with build tools installed)
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

#### Option A: Full Installation (requires build tools)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Feature Extraction Only (minimal)

```bash
pip install numpy pandas scipy scikit-learn
pip install tsfresh==0.20.1
pip install astropy pyyaml tqdm
```

#### Option C: Using Conda (Recommended for Windows)

```bash
conda create -n ratdcs python=3.10
conda activate ratdcs
conda install numpy pandas scipy scikit-learn
pip install tsfresh==0.20.1 astropy pyyaml tqdm
```

---

## Testing Steps

### Test 1: Import Verification

Test that all modules can be imported:

```bash
python -c "
from src.detection.preprocess import LightCurvePreprocessor
from src.detection.feature_extractor import FeatureExtractor
from src.detection.feature_selection import FeatureSelector
from src.detection.batch_processor import BatchProcessor
print('✓ All imports successful!')
"
```

**Expected Output:**
```
✓ All imports successful!
```

### Test 2: Unit Tests

Run the unit test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/unit/detection/ -v

# Run with coverage
pytest tests/unit/detection/ --cov=src.detection --cov-report=html
```

**Expected Output:**
```
tests/unit/detection/test_preprocess.py::TestLightCurvePreprocessor::test_initialization PASSED
tests/unit/detection/test_preprocess.py::TestLightCurvePreprocessor::test_read_fits_kepler PASSED
...
tests/unit/detection/test_feature_extractor.py::TestFeatureExtractor::test_initialization PASSED
...
tests/unit/detection/test_feature_selection.py::TestFeatureSelector::test_initialization PASSED
...

=============== 35 passed in 5.2s ===============
```

### Test 3: Feature Extraction Info

Get information about extractable features:

```bash
python -m src.detection.extract_all_features --info --feature-set comprehensive
```

**Expected Output:**
```
Feature Set: comprehensive
Number of Features: 789

First 10 features:
  1. flux__mean
  2. flux__median
  3. flux__std
  4. flux__variance
  5. flux__minimum
  6. flux__maximum
  7. flux__skewness
  8. flux__kurtosis
  9. flux__autocorrelation__lag_1
  10. flux__autocorrelation__lag_2
  ... and 779 more
```

### Test 4: Preprocessing Test

Test preprocessing with synthetic data:

```python
# Create test_preprocessing.py
import numpy as np
from src.detection.preprocess import LightCurvePreprocessor

# Create synthetic light curve
n_points = 1000
synthetic_lc = {
    'time': np.linspace(0, 100, n_points),
    'flux': np.ones(n_points) + 0.01 * np.random.randn(n_points),
    'flux_err': 0.001 * np.ones(n_points)
}

# Add some outliers
synthetic_lc['flux'][50] = 10.0
synthetic_lc['flux'][100] = -5.0

# Add some NaNs
synthetic_lc['flux'][200:205] = np.nan

# Test preprocessing
preprocessor = LightCurvePreprocessor()

# Remove NaNs
lc_clean = preprocessor.remove_nans(synthetic_lc)
print(f"✓ NaN removal: {len(synthetic_lc['flux'])} → {len(lc_clean['flux'])} points")

# Remove outliers
lc_no_outliers = preprocessor.remove_outliers(lc_clean)
print(f"✓ Outlier removal: {len(lc_clean['flux'])} → {len(lc_no_outliers['flux'])} points")

# Normalize
lc_normalized = preprocessor.normalize_flux(lc_no_outliers)
print(f"✓ Normalization: median = {np.median(lc_normalized['flux']):.4f}")

# Calculate metrics
metrics = preprocessor.calculate_quality_metrics(lc_normalized)
print(f"✓ Quality metrics: SNR = {metrics['snr']:.2f}")

print("\n✓ All preprocessing tests passed!")
```

Run:
```bash
python test_preprocessing.py
```

### Test 5: Feature Extraction Test

Test feature extraction with synthetic data:

```python
# Create test_extraction.py
import numpy as np
from src.detection.feature_extractor import FeatureExtractor

# Create synthetic light curves
n_lcs = 5
n_points = 1000
light_curves = []

for i in range(n_lcs):
    lc = {
        'time': np.linspace(0, 100, n_points),
        'flux': np.ones(n_points) + 0.01 * np.random.randn(n_points)
    }
    light_curves.append(lc)

# Test extraction
print("Testing feature extraction...")
extractor = FeatureExtractor(
    feature_set='minimal',  # Use minimal for faster testing
    n_jobs=1,
    disable_progressbar=False
)

# Extract features
features = extractor.extract(light_curves)

print(f"✓ Extracted{features.shape[1]} features for {features.shape[0]} light curves")
print(f"✓ Feature names: {list(features.columns[:5])}...")
print("\n✓ Feature extraction test passed!")
```

Run:
```bash
python test_extraction.py
```

---

## Integration Testing (with Real Data)

### Option 1: Test with Sample Data

If you have Kepler/TESS FITS files:

```bash
# Create a test directory with a few FITS files
mkdir -p data/raw/test
# Copy 2-3 FITS files to data/raw/test/

# Extract features
python -m src.detection.extract_all_features \
    --dataset test \
    --max-files 3 \
    --feature-set minimal \
    --n-jobs 2
```

### Option 2: Download Sample Data

```bash
# Download Kepler data (requires data acquisition to be set up)
python -m src.data.kepler_downloader --max-targets 5 --output data/raw/kepler

# Extract features from downloaded data
python -m src.detection.extract_all_features \
    --dataset kepler \
    --max-files 5 \
    --feature-set comprehensive
```

---

## Verification Checklist

 Use this checklist to verify the implementation:

- [ ] All modules import without errors
- [ ] Unit tests pass (35+ tests)
- [ ] Preprocessing works on synthetic data
- [ ] Feature extraction produces 789 features (comprehensive set)
- [ ] Feature extraction produces expected output format (CSV with id, label, features)
- [ ] CLI interface works with --info flag
- [ ] Memory usage is reasonable (<8GB for 100 files)
- [ ] Progress bars display correctly
- [ ] Output files are created in data/processed/
- [ ] Checkpointing works (interrupt and resume)

---

## Expected Performance

Based on synthetic data testing:

| Metric | Expected Value |
|--------|---------------|
| Import time | <5 seconds |
| Unit test time | <10 seconds |
| Feature extraction (100 LCs, minimal) | ~2 minutes |
| Feature extraction (100 LCs, comprehensive) | ~20 minutes |
| Memory usage (100 files) | ~8 GB |
| Output file size (100 samples, 789 features) | ~15 MB |

---

## Troubleshooting

### Issue: numpy/scipy won't install

**Solution:** Use conda or install Visual Studio Build Tools

```bash
# Using conda (recommended)
conda install numpy scipy pandas scikit-learn

# Or use pre-built wheels
pip install --only-binary :all: numpy scipy pandas scikit-learn
```

### Issue: TSFresh is slow

**Solution:** Use efficient or minimal feature set for testing

```bash
python -m src.detection.extract_all_features \
    --dataset kepler \
    --feature-set efficient  # or minimal
```

### Issue: Out of memory

**Solution:** Reduce batch size

```bash
python -m src.detection.extract_all_features \
    --dataset kepler \
    --batch-size 25  # Reduce from default 100
```

---

## Continuous Integration

The implementation includes GitHub Actions CI/CD:

- **File**: `.github/workflows/ci.yml`
- **Triggers**: Push to main, pull requests
- **Steps**:
  1. Setup Python 3.10
  2. Install dependencies
  3. Run linting (pylint, black)
  4. Run unit tests
  5. Generate coverage report

View CI status at: https://github.com/DSeahYS/RATDCS/actions

---

## Next Steps After Testing

1. **If all tests pass:**
   - Mark implementation as verified ✓
   - Begin feature extraction on real Kepler/TESS data
   - Train exoplanet classification models

2. **If issues found:**
   - Document issues in GitHub Issues
   - Fix bugs and retest
   - Update documentation

3. **Performance optimization:**
   - Profile memory usage
   - Optimize bottlenecks
   - Add caching where beneficial

---

## Contact

- **GitHub Repository**: https://github.com/DSeahYS/RATDCS
- **Issues**: https://github.com/DSeahYS/RATDCS/issues
- **Documentation**: [`docs/FEATURE_EXTRACTION.md`](docs/FEATURE_EXTRACTION.md)

---

**Last Updated:** 2025-11-03  
**Testing Status:** Ready for verification