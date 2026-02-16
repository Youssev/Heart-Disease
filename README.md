# Heart-Disease
Heart Disease Prediction
Machine learning project predicting cardiovascular risk using the UCI Heart Disease dataset. Applies Logistic Regression with comprehensive EDA, model evaluation, and feature importance analysis using Python data science stack.

## Features


• Data Loading & Preprocessing: Pandas/NumPy for CSV handling, missing value analysis, LabelEncoder for target


• Exploratory Data Analysis: Seaborn correlation heatmaps, count plots, pairplots


• Interactive Visualizations: Plotly scatter matrix, confusion matrix heatmap, ROC curve, feature importance


• Model Training: Logistic Regression with train/test split (80/20), accuracy scoring


• Model Evaluation: ROC-AUC, confusion matrix, permutation importance ranking


**Tech Stack**

 ```
NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, Plotly
Dataset: Heart_Disease_Prediction.csv (303 samples, 14 features)
 ```

**Key Results**

```
• Logistic Regression Accuracy: ~85% on test set
• ROC-AUC: Strong discrimination performance
• Top Features: [Chest Pain Type, Exercise ST Depression, Thalassemia]
```

## Visualizations Generated 

• Target distribution countplot

• Numeric features correlation heatmap

• Pairwise scatter matrix (Plotly)

• ROC curve analysis

• Confusion matrix heatmap

• Horizontal permutation feature importance

• Cholesterol by Sex barplot

## Dataset
[Heart Disease Dataset ](https://www.kaggle.com/datasets/data855/heart-disease)

```
df = pd.read_csv("Heart_Disease_Prediction.csv")  
```
## Notebook Structure

1. Data Loading & Inspection
2. Target Encoding & EDA  
3. Correlation Analysis & Visualizations
4. Train/Test Split
5. Logistic Regression Training
6. Model Evaluation (ROC, Confusion Matrix)
7. Feature Importance Analysis
8. Demographic Visualizations
