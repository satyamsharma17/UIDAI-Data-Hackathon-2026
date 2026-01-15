# Quick Start Guide

## Setup (5 minutes)

### 1. Create Virtual Environment
```bash
cd "/Users/satyamsharma/Satverse AI/UIDAI Data Hackathon 2026"
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Analysis

### Option 1: Quick Analysis Script (Fastest)
```bash
python run_analysis.py
```

This will:
- Load 20% sample of data (adjustable)
- Clean and preprocess
- Generate processed datasets
- Show summary statistics

### Option 2: Jupyter Notebooks (Detailed Analysis)

**Run notebooks in order:**

```bash
jupyter notebook
```

1. **01_data_loading_preprocessing.ipynb**
   - Load all datasets
   - Validate data quality
   - Clean and enhance data
   - Save processed files

2. **02_exploratory_data_analysis.ipynb**
   - Univariate analysis
   - Bivariate relationships
   - Trivariate interactions
   - Generate visualizations

3. **03_temporal_spatial_analysis.ipynb** (create this)
   - Time series analysis
   - Geographic patterns
   - Seasonal decomposition

4. **04_anomaly_detection.ipynb** (create this)
   - Statistical outliers
   - Temporal anomalies
   - ML-based detection

5. **05_predictive_modeling.ipynb** (create this)
   - Forecasting models
   - Model comparison
   - Demand prediction

6. **06_insights_recommendations.ipynb** (create this)
   - Compile all findings
   - Generate recommendations
   - Create final report

---

## Expected Outputs

### Directory Structure
```
outputs/
├── figures/               # All visualizations (PNG)
│   ├── 01_age_distribution_enrolment.png
│   ├── 02_daily_enrolments_distribution.png
│   ├── 03_top_states_enrolments.png
│   └── ...
├── insights/              # Text reports
│   ├── eda_summary.txt
│   ├── anomaly_report.txt
│   └── recommendations.txt
├── enrolment_processed.parquet
├── demographic_processed.parquet
└── biometric_processed.parquet
```

---

## Customization

### Adjust Sample Size

In `run_analysis.py`:
```python
main(sample_frac=0.2)  # 20% of data
main(sample_frac=1.0)  # 100% of data (full analysis)
```

In notebooks:
```python
SAMPLE_FRACTION = 0.2  # Change to desired fraction
```

### Select Specific States

```python
# In any notebook
states_of_interest = ['Maharashtra', 'Karnataka', 'Tamil Nadu']
filtered_df = df[df['state'].isin(states_of_interest)]
```

---

## Troubleshooting

### Memory Issues
If running out of memory:
1. Reduce `sample_frac` to 0.1 (10%)
2. Process one dataset at a time
3. Use chunked processing in data_loader

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Date Parsing Errors
Data loader handles DD-MM-YYYY format automatically. If issues persist, check CSV encoding.

---

## Performance Tips

1. **Start with 10% sample** for initial exploration
2. **Run full analysis overnight** (may take hours)
3. **Use SSD** for faster file I/O
4. **Close other applications** to free memory
5. **Save processed data** to avoid reloading

---

## Creating the Final Submission PDF

### Recommended Workflow:

1. **Run all notebooks** (in order)
2. **Export notebooks to HTML**:
   ```bash
   jupyter nbconvert --to html notebooks/*.ipynb
   ```

3. **Compile in Word/Google Docs**:
   - Problem statement and approach
   - Dataset description
   - Methodology (from notebooks)
   - Copy visualizations from outputs/figures
   - Code snippets (key functions)
   - Insights and recommendations

4. **Export to PDF**

### OR Use LaTeX for Professional Output

Template structure:
```
1. Title Page
2. Executive Summary (1 page)
3. Problem Statement & Approach (2 pages)
4. Datasets Used (1 page)
5. Methodology (3-4 pages)
6. Data Analysis & Visualizations (8-10 pages)
7. Code Appendix (embedded key functions)
8. Insights & Recommendations (2-3 pages)
9. References
```

---

## Key Files Reference

### Source Code
- `src/data_loader.py` - Data loading utilities
- `src/preprocessing.py` - Data cleaning and feature engineering
- `src/visualization.py` - Reusable visualization functions
- `src/temporal_analysis.py` - Time series analysis
- `src/spatial_analysis.py` - Geographic analysis
- `src/anomaly_detector.py` - Anomaly detection methods
- `src/forecasting.py` - Predictive models

### Notebooks
- Complete analysis workflow with explanations
- Ready to embed in PDF submission

### Documentation
- `README.md` - Project overview
- `SUBMISSION_GUIDE.md` - Comprehensive submission details
- `QUICKSTART.md` - This file

---

## Timeline Recommendation

**Day 1-2**: Setup + Data Loading + Preprocessing
- Run `01_data_loading_preprocessing.ipynb`
- Validate data quality
- Generate processed datasets

**Day 3-4**: EDA + Analysis
- Run `02_exploratory_data_analysis.ipynb`
- Run temporal and spatial analysis
- Generate visualizations

**Day 5**: Advanced Analysis
- Anomaly detection
- Predictive modeling
- Model evaluation

**Day 6**: Insights + Documentation
- Compile findings
- Write recommendations
- Prepare code documentation

**Day 7**: Submission Preparation
- Create PDF report
- Review all content
- Final checks and submission

---

## Getting Help

Check:
1. Function docstrings: `help(function_name)`
2. Inline comments in code
3. Error messages (usually descriptive)
4. `SUBMISSION_GUIDE.md` for detailed methodology

---

## Next Steps After This Guide

1. ✅ Run `python run_analysis.py` to test setup
2. ✅ Open Jupyter and run first notebook
3. ✅ Review generated outputs in `outputs/` folder
4. ✅ Proceed through remaining notebooks
5. ✅ Compile final PDF submission

---

**Good luck with your submission! 🎯**
