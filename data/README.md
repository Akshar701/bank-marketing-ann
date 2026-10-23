# Data Directory

This directory stores the official UCI Bank Marketing dataset.

## Dataset Provenance
- **Source:** [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Reference File:** `bank-full.csv` (semicolon `;` delimited)
- **Observations:** 45,211 rows
- **Input Features:** 16 attributes (7 numerical, 9 categorical)
- **Target Variable:** `y` ("yes" or "no", indicating whether the customer subscribed to a term deposit)
- **Class Imbalance:** ~11.7% "yes" (5,289) vs ~88.3% "no" (39,922)

## Leakage Prevention
Preprocessing pipelines (StandardScaler, OneHotEncoder) are fitted strictly on the 70% training split. The validation (15%) and test (15%) splits remain untouched during fitting.
