import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# -----------------------------
# Paths
# -----------------------------
base_path = "/Users/berry/Desktop/firedetect/fire_dataset/fire_dataset"
train_dir = os.path.join(base_path, "train")
val_dir   = os.path.join(base_path, "val")

# -----------------------------
# ImageDataGenerators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary'
)

# -----------------------------
# Build CNN model
# -----------------------------
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

base_model.trainable = True
for layer in base_model.layers[:-50]:  # freeze the bottom layers
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# -----------------------------
# Train model
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_fire_model.h5", save_best_only=True)
]
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)
# -----------------------------
# Save model
# -----------------------------
model.save("fire_detection_cnn.h5")
print("✅ Model saved as fire_detection_cnn.h5")