# AnalystLab Africa – Week 2: Data Preprocessing & Feature Engineering

## Project Overview
Complete preprocessing pipeline for the ShopSmart retail dataset, turning raw multi-table data into a machine-learning-ready customer feature matrix.

## Contents
- `Week2_Data_Preprocessing_Feature_Engineering.ipynb` – Full Jupyter notebook covering all assignment parts
- `data/` – Original ShopSmart CSV files (Customers, Orders, Products, Stores, Returns, …)
- After running the notebook you will also get:
  - `ShopSmart_ML_Ready_Customers.csv` – Final ML-ready dataset
  - `preprocessing_visualizations.png` – 6 required charts

## How to Run
1. Open the notebook in Jupyter / VS Code / Colab / Cursor.
2. Make sure the relative path `DATA_DIR = '../data/'` points to the data folder (or change it).
3. Run all cells top-to-bottom.
4. The processed CSV and visualization image are written to the same folder as the notebook.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Assignment Coverage
| Part | Content |
|------|---------|
| Part 1 | Data Inspection (load, head, shape, dtypes, missing, duplicates, describe) |
| Part 2 | Data Cleaning (dates, types, standardization) |
| Part 3 | Feature Engineering (14 RFM + behavioural features) |
| Part 4 | Encoding (Ordinal + One-Hot) & Scaling (Robust + Standard) |
| Part 5 | Outlier detection (IQR, Z-score, winsorize) + Feature Selection |
| Part 6 | Final ML-ready CSV |

≥ 6 visualizations with titles and labels are included.

## Business Questions Answered
See the final markdown section of the notebook for detailed answers to all 7 business questions.

## Author
Junior Machine Learning Engineer – AnalystLab Africa Internship
