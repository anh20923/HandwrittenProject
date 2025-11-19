Part 1. Project Overview:
This project implements a handwritten digit recognition system using the MNIST dataset.
Two machine learning models were developed:
1. Logistic Regression (Baseline Model)
2. Convolutional Neural Network (CNN)
A Streamlit-based GUI was created to allow users to upload handwritten digit images (0–9) and receive prediction results along with confidence scores.

Part 2. Repository Structure:
source_code folder:
│
├── app.py
├── utils.py
├── train_model.py
├── logistic_baseline.py
├── check_mnist.py
│
├── cnn_model.h5
│
├── cnn_metrics.txt
├── logistic_metrics.txt
├── cnn_classification_report.txt
├── logistic_classification_report.txt
│
├── confusion_matrix.png
├── logistic_confusion_matrix.png
├── loss_curve.png
├── metrics.png
│
└── README.md


Part 3. 
Before running the GUI, install required packages: 
pip install -r requirements.txt

Run the following command: 
streamlit run app.py
