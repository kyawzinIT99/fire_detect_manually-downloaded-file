import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

# 1️⃣ Load model
model_path = "/Users/berry/Desktop/firedetect/best_fire_model.h5"
model = tf.keras.models.load_model(model_path)

# 2️⃣ Prepare validation data
val_dir = "/Users/berry/Desktop/firedetect/fire_dataset/fire_dataset/val"
val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# 3️⃣ Predict all validation images
preds = model.predict(val_gen)   # <<< this was missing
y_pred = (preds > 0.5).astype(int).flatten()
y_true = val_gen.classes

# 4️⃣ Evaluate
labels = [0, 1]  # 0 = no_fire, 1 = fire
target_names = list(val_gen.class_indices.keys())

print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))