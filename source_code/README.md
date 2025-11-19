## Part 1. Project Overview: 
This project implements a handwritten digit recognition system using the MNIST dataset.
Two machine learning models were developed:
1. Logistic Regression (Baseline Model)
2. Convolutional Neural Network (CNN)
A Streamlit-based GUI was created to allow users to upload handwritten digit images (0–9) and receive prediction results along with confidence scores.


## Part 2. Repository Structure
### source_code folder
```
source_code/
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
```

### GUI_executable folder
```
GUI_executable/
│
├── app.py                   
├── utils.py
├── cnn_model.h5             
│
├── requirements.txt         
│
└── MNIST_23_0.png (sample image)
```
        


## Part 3. Running the GUI
To run the Streamlit GUI correctly, follow the steps below:
### 1. Navigate to the GUI folder
``` cd GUI_executable ```

### 2. Install required packages
```pip install -r requirements.txt```

### 3. Run the Streamlit app
```streamlit run app.py```


## Part 4. Training the Models
### Run those command below in VS terminal: 
### 1. Navigate to the source_code folder
``` cd source_code ```
### 2. Train the CNN model:
``` python train_model.py ```

This will generate the following files:
cnn_model.h5
confusion_matrix.png
metrics.png
loss_curve.png
cnn_metrics.txt
cnn_classification_report.txt

### 3. Train Logistic Regression baseline:
Run this in VS terminal: 
``` python logistic_baseline.py ```

This will generate:
logistic_confusion_matrix.png
logistic_metrics.txt
logistic_classification_report.txt



## Part 5. File Descriptions
### Python Scripts 
```
File	                        Description
app.py	                        Streamlit GUI interface
utils.py	                    Functions for preprocessing user-uploaded images
train_model.py	                CNN model definition, training, evaluation
logistic_baseline.py	        Logistic regression baseline model
check_mnist.py	                Quick check for MNIST validation
```

### Model Files 
```
File	                                        Description
cnn_model.h5	                                Trained CNN model used by GUI
cnn_metrics.txt, logistic_metrics.txt	        Model performance summaries
```

### Plots 
```
Image	                        Description
confusion_matrix.png	        CNN confusion matrix
logistic_confusion_matrix.png	Baseline confusion matrix
loss_curve.png	                Training/validation loss
metrics.png	                    Training/validation accuracy
```
