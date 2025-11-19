import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.datasets import mnist

# 1. Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Preprocessing for Logistic Regression
# Flatten 28x28 → 784 features
X_train_flat = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
X_test_flat = X_test.reshape(len(X_test), -1).astype("float32") / 255.0

# 3. Train Logistic Regression model
print("Training Logistic Regression... (may take 10–20 seconds)")
model = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=200,
    n_jobs=-1
)

model.fit(X_train_flat, y_train)

# 4. Predict
y_pred = model.predict(X_test_flat)

# 5. Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Test Accuracy: {acc*100:.2f}%")

# 6. Save accuracy result
with open("logistic_metrics.txt", "w") as f:
    f.write(f"Logistic Regression Accuracy: {acc:.4f}\n")

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix – Logistic Regression")
plt.savefig("logistic_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

# 8. Classification Report
report = classification_report(y_test, y_pred, digits=4)
print(report)

with open("logistic_classification_report.txt", "w") as f:
    f.write(report)

print("Saved logistic_metrics.txt, logistic_confusion_matrix.png, logistic_classification_report.txt")
