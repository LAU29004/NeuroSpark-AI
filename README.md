# NeuroSpark-AI 🧠

A machine learning project for detecting and classifying neurological conditions using advanced AI models. NeuroSpark-AI combines deep learning and ensemble methods to predict brain stroke risk and epileptic seizure patterns from medical data.

## 📋 Project Overview

NeuroSpark-AI consists of two main models:

### 1. **Brain Stroke Detection Model**
Predicts stroke risk using patient medical data with class imbalance handling and multiple classification algorithms.

**Key Features:**
- Dataset: Stroke dataset with comprehensive patient health metrics
- Data Preprocessing: Label encoding for categorical variables
- Class Imbalance Handling: SMOTE (Synthetic Minority Over-sampling Technique)
- Models: Decision Tree, XGBoost
- Evaluation: Confusion matrices, accuracy metrics, classification reports

### 2. **Epileptic Seizure Recognition Model**
Multi-class classification of epileptic brain activity patterns using both machine learning and deep learning approaches.

**Key Features:**
- Dataset: IEEE epilepsy dataset with 5 classes (Eyes Open, Eyes Closed, Healthy Brain Region, Tumor Region, Seizure)
- Data Preprocessing: StandardScaler normalization
- Class Imbalance Handling: SMOTE for balanced training
- Models: 
  - Machine Learning: XGBoost, KNN (K-Nearest Neighbors), Random Forest
  - Deep Learning: Neural networks with Conv1D layers
- Multi-set Validation: Train/Validation/Test splits with stratification

## 📂 Repository Structure

```
NeuroSpark-AI/
├── Brain_stroke_Final.ipynb          # Brain stroke detection model
├── IEEE_Epilepsy_Final.ipynb         # Epilepsy seizure recognition model
├── Stroke Dataset.csv                # Stroke patient data
├── Epileptic Seizure Recognition.csv # EEG signal data for seizure detection
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required libraries (see below)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LAU29004/NeuroSpark-AI.git
cd NeuroSpark-AI
```

2. Install required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow
```

### Dependencies

| Library | Purpose |
|---------|---------|
| **numpy** | Numerical computing |
| **pandas** | Data manipulation and analysis |
| **matplotlib** | Data visualization |
| **seaborn** | Statistical data visualization |
| **scikit-learn** | Machine learning algorithms |
| **imbalanced-learn** | SMOTE for handling imbalanced data |
| **xgboost** | Gradient boosting classifier |
| **tensorflow/keras** | Deep learning models |

## 📊 Datasets

### Stroke Dataset
- **Filename**: `Stroke Dataset.csv`
- **Description**: Contains patient medical information for stroke risk prediction
- **Features**: Demographics, medical history, and health metrics
- **Target**: Stroke (binary classification)

### Epileptic Seizure Recognition Dataset
- **Filename**: `Epileptic Seizure Recognition.csv`
- **Description**: EEG signal data for seizure detection and classification
- **Features**: 178 features (EEG measurements)
- **Classes**: 5 categories (Eyes Open, Eyes Closed, Healthy Region, Tumor Region, Seizure)
- **Target**: Class label (1-5, adjusted to 0-based indexing)

## 🔧 Model Architecture & Techniques

### Common Preprocessing Steps

1. **Data Cleaning**: Column name standardization
2. **Categorical Encoding**: LabelEncoder for categorical variables
3. **Feature Scaling**: StandardScaler for normalization (Epilepsy model)
4. **Class Imbalance Handling**: SMOTE with minority sampling strategy

### Data Splitting

- **Stroke Model**: Train/Test split with appropriate proportions
- **Epilepsy Model**: 
  - 60% Training
  - 20% Validation  
  - 20% Testing
  - Stratified splits to preserve class distribution

### Classifiers Used

#### Machine Learning Models:
- **Decision Tree Classifier** (Stroke)
- **XGBoost** (Both models)
- **K-Nearest Neighbors (KNN)** (Epilepsy)
- **Random Forest** (Epilepsy)

#### Deep Learning:
- **Convolutional Neural Networks with Conv1D layers** (Epilepsy)
- Input → Conv1D layers → Dense layers → Output

## 📈 Model Performance Evaluation

### Metrics Used:
- **Accuracy**: Overall correctness of predictions
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**: Precision, Recall, F1-Score
- **Validation & Test Set Performance**: Multi-level evaluation

### Visualizations:
- Confusion matrices with heatmaps
- Class distribution plots
- Training/Validation performance curves
- Feature importance analysis

## 💡 Key Techniques

### SMOTE (Synthetic Minority Over-sampling)
Addresses class imbalance by generating synthetic samples for minority classes, preventing model bias toward majority classes.

### Stratified Train-Test Split
Ensures class distribution is maintained across train, validation, and test sets for unbiased evaluation.

### Ensemble Methods
XGBoost combines weak learners to create a robust predictive model with better generalization.

## 📓 Usage

### Running the Brain Stroke Model
1. Open `Brain_stroke_Final.ipynb` in Jupyter Notebook
2. Run cells sequentially to:
   - Load and explore the stroke dataset
   - Preprocess and encode features
   - Apply SMOTE for class balancing
   - Train Decision Tree and XGBoost models
   - Evaluate and compare model performance

### Running the Epilepsy Model
1. Open `IEEE_Epilepsy_Final.ipynb` in Jupyter Notebook
2. Run cells sequentially to:
   - Load EEG signal data
   - Perform exploratory data analysis
   - Standardize features
   - Apply SMOTE for balanced sampling
   - Train XGBoost, KNN, Random Forest, and Neural Network models
   - Generate confusion matrices and performance metrics
   - Compare model accuracies

## 🎯 Results & Insights

The models demonstrate:
- Effective handling of imbalanced medical datasets
- Strong predictive performance using ensemble methods
- Comparative analysis of multiple algorithms
- Deep learning viability for EEG signal classification

For detailed results, confusion matrices, and performance comparisons, see the respective notebook outputs.

## 📚 Learning Resources

- [Imbalanced Learn Documentation](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

## 🔍 Next Steps & Improvements

- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Feature Engineering**: Domain-specific feature creation for medical data
- **Cross-Validation**: K-fold cross-validation for robust evaluation
- **Model Interpretability**: SHAP values for feature importance analysis
- **Real-World Deployment**: REST API for clinical integration
- **Additional Datasets**: Validation on other stroke/seizure datasets

## ⚠️ Disclaimer

These models are developed for educational and research purposes. They should not be used for actual medical diagnosis without proper validation by healthcare professionals and regulatory approval.
