
# Credit Card Fraud Detection

## Introduction

This project implements a **Credit Card Fraud Detection** system using an Artificial Neural Network (ANN). It involves training a deep learning model to classify fraudulent and non-fraudulent transactions based on a dataset of historical credit card transactions. The system is designed to help financial institutions and businesses detect fraudulent activities and prevent financial losses.

The project consists of:
1. **A Jupyter Notebook (`credit_card_fraud_detection_final.ipynb`)** for training and evaluating the model.
2. **A Streamlit web application (`app.py`)** for real-time fraud detection using the trained model.

### **Version Information**  

#### **Software Versions Used:**
- **Python**: 3.8+
- **TensorFlow**: 2.x
- **Scikit-Learn**: 1.x
- **Pandas**: 1.x
- **NumPy**: 1.x
- **Streamlit**: Latest stable version  

#### **Project Versions:**
- **Version 1.0**: Initial implementation with ANN model and Streamlit web app.
- **Version 1.1**: Improved feature engineering and model optimization.
- **Version 1.2**: UI enhancements and real-time fraud detection refinements.


## Features

- **Data Preprocessing**: Includes feature engineering, normalization, and handling of imbalanced data.
- **Artificial Neural Network (ANN)**: Trained using TensorFlow/Keras to classify transactions as fraudulent or non-fraudulent.
- **Web Interface**: Built using Streamlit to allow users to upload transaction data for real-time fraud predictions.
- **Downloadable Results**: Users can download the predictions in CSV format.


## Installation and Setup

To run the project locally, follow these steps:

### Prerequisites

Ensure to have Python 3.8+ installed. Install the required dependencies:

pip install -r requirements.txt

### Running the Web App

1. Train the model using the Jupyter Notebook (`credit_card_fraud_detection_final.ipynb`).
2. Save the trained model as `credit_card_fraud_detection_model.h5`.
3. Run the Streamlit app:


streamlit run app.py

4. Upload a CSV file containing transaction data.
5. View and download fraud detection results.


## Usage

### 1. Training the Model
The `credit_card_fraud_detection_final.ipynb` file:
- Loads and preprocesses the dataset.
- Trains an ANN model using TensorFlow/Keras.
- Evaluates model performance (accuracy, precision, recall, F1-score).
- Saves the trained model.

### 2. Fraud Detection via Web App
The `app.py` file:
- Loads the trained model.
- Accepts CSV transaction data for real-time fraud prediction.
- Displays and allows users to download prediction results.

**Expected CSV Format:**
| Time | Amount | V1 | V2 | ... | V28 |
|------|--------|----|----|-----|-----|
| 10   | 100.5  | 1.2 | -0.5 | ... | 2.1 |

The CSV file contains all necessary columns.


## Model Performance

- **Test Accuracy:** ~93%
- **Precision (Fraud):** 83%
- **Recall (Fraud):** 59%
- **F1-Score (Fraud):** 0.69

The model is effective at identifying fraudulent transactions but may require further improvements to increase recall.

### **Conclusion**  

This project successfully implements a **Credit Card Fraud Detection System** using an **Artificial Neural Network (ANN)**. The model achieves **high accuracy (~93%)** and **good precision (83%)** in identifying fraudulent transactions. However, the recall (59%) suggests that some fraudulent cases are missed, which could be improved through techniques like **class balancing, hyperparameter tuning, or using ensemble learning methods**. The integration of a **Streamlit web application** allows for **real-time fraud detection**, making this project practical for deployment in financial institutions.

Future enhancements could include:
- **Improving Recall** using advanced class balancing techniques.
- **Exploring Other Models** like ensemble learning, XGBoost, or hybrid approaches.
- **Enhancing Feature Engineering** by incorporating more behavioral data.

This project serves as a **robust foundation** for real-world fraud detection and can be further optimized to **minimize financial losses and protect customers from fraudulent transactions**.


