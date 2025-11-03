# Processed Data Directory

This directory contains processed feature data extracted from Kepler and TESS light curves.

## Directory Structure

```
data/processed/
├── README.md                           # This file
├── kepler_features_comprehensive.csv   # Kepler 789-feature set
├── kepler_features_comprehensive.json  # Kepler feature metadata
├── kepler_features_selected_tsfresh.csv  # Selected Kepler features
├── kepler_features_selected_tsfresh.json # Selection metadata
├── tess_features_comprehensive.csv     # TESS 789-feature set
├── tess_features_comprehensive.json    # TESS feature metadata
├── tess_features_selected_tsfresh.csv  # Selected TESS features
├── tess_features_selected_tsfresh.json # Selection metadata
├── feature_metadata.json               # Global feature descriptions
└── checkpoints/                        # Processing checkpoints
    ├── kepler_comprehensive.json
    ├── kepler_comprehensive_final.json
    ├── tess_comprehensive.json
    └── tess_comprehensive_final.json
```

## File Formats

### Feature CSV Files

Feature CSV files contain one row per light curve with the following structure:

```csv
id,label,flux__mean,flux__std,flux__variance,...
kplr001234567-2009123456789_llc,1,0.9998,0.0012,0.0000014,...
kplr002345678-2009234567890_llc,0,1.0001,0.0015,0.0000023,...
```

**Columns:**
- `id`: Light curve identifier (KIC ID for Kepler, TIC ID for TESS)
- `label`: Classification label (1 = exoplanet, 0 = non-exoplanet)
- `flux__*`: TSFresh-extracted features (789 total for comprehensive set)

### Metadata JSON Files

Metadata files contain information about the extraction process:

```json
{
  "input_dir": "data/raw/kepler",
  "n_files": 100,
  "pattern": "*.fits",
  "feature_set": "comprehensive",
  "extractor_config": {
    "n_jobs": 4,
    "chunksize": 10
  },
  "timestamp": "2025-11-03T12:00:00Z"
}
```

## Feature Sets

### Comprehensive (789 features)

The comprehensive feature set extracts 789 time-series features using TSFresh's `ComprehensiveFCParameters()`:

- **Statistical features** (82): mean, median, std, variance, quantiles, moments
- **Autocorrelation features** (120): ACF and PACF at various lags
- **Spectral features** (400+): FFT coefficients, spectral entropy
- **Complexity features** (50): Approximate entropy, sample entropy
- **Linear features** (40): AR coefficients, trends
- **Other features** (97): Change detection, distribution analysis

### Selected Features

After feature selection, the number of features is typically reduced to ~100-300 most relevant features based on:

1. **TSFresh relevance test**: p-value based selection (FDR corrected)
2. **Correlation filtering**: Remove features with correlation > 0.95
3. **Mutual information**: Select high MI with target variable

## Data Statistics

### Expected File Sizes

| Dataset | # Samples | # Features | File Size (CSV) |
|---------|-----------|-----------|----------------|
| Kepler (100) | 100 | 789 | ~15 MB |
| Kepler (1000) | 1000 | 789 | ~150 MB |
| TESS (100) | 100 | 789 | ~15 MB |
| TESS (1000) | 1000 | 789 | ~150 MB |
| Selected (100) | 100 | ~200 | ~4 MB |

### Typical Class Distribution

**Kepler:**
- Exoplanet (label=1): ~10-15%
- Non-exoplanet (label=0): ~85-90%

**TESS:**
- TOI Candidate (label=1): ~20-30%
- Non-candidate (label=0): ~70-80%

## Usage

### Load Features

```python
import pandas as pd

# Load Kepler features
kepler_features = pd.read_csv('data/processed/kepler_features_comprehensive.csv')

# Separate features and labels
X = kepler_features.drop(columns=['id', 'label'])
y = kepler_features['label']
ids = kepler_features['id']

print(f"Loaded {len(X)} samples with {len(X.columns)} features")
print(f"Class distribution: {y.value_counts()}")
```

### Load with Metadata

```python
import json
import pandas as pd

# Load features
features = pd.read_csv('data/processed/kepler_features_comprehensive.csv')

# Load metadata
with open('data/processed/kepler_features_comprehensive.json', 'r') as f:
    metadata = json.load(f)

print(f"Feature set: {metadata['feature_set']}")
print(f"Extracted from: {metadata['input_dir']}")
print(f"Processing time: {metadata.get('elapsed_time_seconds', 'N/A')}s")
```

### Train a Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load features
df = pd.read_csv('data/processed/kepler_features_selected_tsfresh.csv')
X = df.drop(columns=['id', 'label'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

## Data Quality

### Quality Checks

All light curves undergo preprocessing with quality checks:

1. **Minimum data points**: ≥100 valid points after cleaning
2. **NaN removal**: All NaN and infinite values removed
3. **Outlier removal**: 5-sigma clipping using robust MAD
4. **Normalization**: Median flux normalization
5. **SNR threshold**: Signal-to-noise ratio computed

### Quality Metrics

Each processed light curve has associated quality metrics (stored in metadata):

```json
{
  "n_points": 1000,
  "time_span_days": 90.5,
  "mean_flux": 1.0,
  "std_flux": 0.0012,
  "snr": 833.3,
  "completeness": 0.98
}
```

## Troubleshooting

### Missing Features

If features CSV is empty or has fewer columns than expected:

1. Check `feature_extraction.log` for errors
2. Verify FITS files are valid and contain TIME, FLUX columns
3. Check preprocessing parameters (min_valid_points)
4. Try with `feature_set='minimal'` first for testing

### NaN Values

TSFresh automatically imputes NaN values, but if many NaNs persist:

1. Check light curve quality (insufficient data points)
2. Increase `min_valid_points` in preprocessing
3. Review feature extraction warnings in logs

### Large File Sizes

If CSV files are too large:

1. Use feature selection to reduce dimensionality
2. Process in smaller batches
3. Save to Parquet format instead of CSV:

```python
df.to_parquet('data/processed/features.parquet', compression='gzip')
```

## Data Versioning

Track data versions for reproducibility:

```bash
# Create a version info file
echo "RATDCS v1.0.0" > data/processed/VERSION.txt
echo "Extracted: $(date)" >> data/processed/VERSION.txt
echo "TSFresh: 0.20.1" >> data/processed/VERSION.txt
echo "Feature set: comprehensive (789 features)" >> data/processed/VERSION.txt
```

## References

- **TSFresh Documentation**: https://tsfresh.readthedocs.io/
- **Feature Extraction Guide**: [`docs/FEATURE_EXTRACTION.md`](../../docs/FEATURE_EXTRACTION.md)
- **Architecture**: [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md)
- **Data Acquisition**: [`docs/DATA_ACQUISITION.md`](../../docs/DATA_ACQUISITION.md)

---

**Last Updated:** 2025-11-03  
**RATDCS Version:** 1.0.0