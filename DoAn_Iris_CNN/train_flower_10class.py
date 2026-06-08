import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. DUONG DAN DATASET
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "Training Data")
VALID_DIR = os.path.join(DATASET_DIR, "Validation Data")
TEST_DIR = os.path.join(DATASET_DIR, "Testing Data")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# 2. LOAD DATASET
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Danh sach class:", class_names)
print("So luong class:", num_classes)

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)

# Tang toc load data, khong lam thay doi du lieu
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 3. XAY DUNG MODEL CNN THUAN
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # Chuan hoa pixel ve khoang [0, 1]
    layers.Rescaling(1. / 255.0),

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Fully Connected
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

# =========================
# 4. COMPILE MODEL
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 5. TRAIN MODEL
# =========================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# Luu model
model.save("flower_model_cnn_thuan.keras")

# =========================
# 6. DANH GIA TREN TEST SET
# =========================
test_loss, test_acc = model.evaluate(test_ds)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# =========================
# 7. VE BIEU DO ACCURACY
# =========================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_chart.png")
plt.show()

# =========================
# 8. VE BIEU DO LOSS
# =========================
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_chart.png")
plt.show()

# =========================
# 9. CONFUSION MATRIX + CLASSIFICATION REPORT
# =========================
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()

print("Da train xong va luu model:")
print("- flower_model_cnn_thuan.keras")
print("- class_names.json")
print("- accuracy_chart.png")
print("- loss_chart.png")
print("- confusion_matrix.png")