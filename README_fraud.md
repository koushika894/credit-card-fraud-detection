# Credit Card Fraud Detection using Machine Learning

## Project Overview

Machine learning solution for detecting fraudulent credit card transactions in a highly imbalanced dataset (fraud ≈ 0.17%). Uses SMOTE for class balancing and evaluates models with recall-focused metrics.

## Key Features

- Handles severe class imbalance using SMOTE
- Compares Logistic Regression vs Random Forest
- Hyperparameter tuning with RandomizedSearchCV
- Comprehensive evaluation metrics (Precision, Recall, F1, ROC-AUC)
- Cross-validation analysis
- Feature importance analysis

## Dataset

- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (Time, Amount, V1-V28 PCA components)
- **Target**: Class (0 = Legitimate, 1 = Fraud)
- **Class Distribution**: 99.83% legitimate vs 0.17% fraud

## Model Performance

Comprehensive evaluation comparing baseline and SMOTE-enhanced models:

- **Baseline Models**: Initial Logistic Regression and Random Forest on imbalanced data
- **SMOTE Models**: Models retrained on balanced data with synthetic oversampling
- **Tuned Model**: Random Forest with optimized hyperparameters (RandomizedSearchCV)

**Key Evaluation Metrics**:
- Accuracy, Precision, Recall (most critical for fraud detection)
- F1-Score, ROC-AUC
- Confusion matrices and cross-validation analysis

**Expected Improvement**: SMOTE significantly improves recall (fraud detection rate) while maintaining acceptable precision.

## Technical Stack

- **Python 3.8+**
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn
- **ML Algorithms**: Logistic Regression, Random Forest
- **Techniques**: SMOTE, RandomizedSearchCV, K-Fold Cross-Validation

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.9.0
scipy>=1.7.0
jupyter>=1.0.0
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd credit-card-fraud-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook kaggle.ipynb
```

### Running on Google Colab

1. Upload `kaggle.ipynb` to Google Colab
2. Upload your Kaggle API credentials (`kaggle.json`)
3. Run all cells sequentially

## Project Structure

```
credit-card-fraud-detection/
├── README.md              # Project documentation
├── kaggle.ipynb           # Main Jupyter notebook
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore rules
```

## Workflow

1. **Setup & Data Loading** - Kaggle API integration and dataset download
2. **Exploratory Data Analysis** - Dataset inspection and class distribution analysis
3. **Data Preprocessing** - Train-test split with stratification
4. **Baseline Models** - Initial Logistic Regression and Random Forest training
5. **SMOTE Application** - Address class imbalance
6. **Hyperparameter Tuning** - RandomizedSearchCV optimization
7. **Model Retraining** - Train on balanced data
8. **Cross-Validation** - 5-fold validation for stability
9. **Visualization** - ROC curves, performance comparison, feature importance
10. **Conclusions** - Key findings and practical implications

## Key Insights

- **Class Imbalance**: Accuracy alone is misleading; recall is critical for fraud detection
- **SMOTE Impact**: Significantly improves fraud detection rate
- **Feature Importance**: Certain PCA components are strong fraud indicators
- **Model Selection**: Random Forest outperforms Logistic Regression after tuning

## Practical Notes

- Trade-off between false positives (annoying customers) and false negatives (missed fraud)
- Model requires periodic retraining as transaction patterns evolve
- Can be deployed as real-time API for production systems
