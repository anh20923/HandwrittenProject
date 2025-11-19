import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Preprocessing
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 3. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks để tối ưu training
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
]

# 5. Train model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=12,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print("Final CNN Test Accuracy:", round(test_acc*100, 2), "%")

# 7. Save model
model.save("cnn_model.h5")
print("Saved cnn_model.h5")

# 8. Plot accuracy
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("metrics.png", dpi=200)
plt.close()

# 9. Plot loss
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=200)
plt.close()

# 10. Confusion matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
plt.title("Confusion Matrix – CNN")
plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

# 11. Classification report
report = classification_report(y_test, y_pred, digits=4)
print(report)

with open("cnn_classification_report.txt", "w") as f:
    f.write(report)

with open("cnn_metrics.txt", "w") as f:
    f.write(f"CNN Accuracy: {test_acc:.4f}\n")
    f.write(f"CNN Loss: {test_loss:.4f}\n")

print("Saved cnn_classification_report.txt and cnn_metrics.txt")
