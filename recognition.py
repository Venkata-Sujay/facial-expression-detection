#23BTRCL190 G.V.SUJAY
#23BTRCL192 I.DHANUSH
#23BTRCL193 I.L.SASIDHAR REDDY
#23BTRCL205 K.HIMESH
#23BTRCL252 K.MAHITHA
#23BTRCL186 G.KEERTHANA


import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP
# ==========================================
IMAGE_FOLDER_PATH = 'Cropped_Data'  # <--- UPDATE THIS

filenames = os.listdir(IMAGE_FOLDER_PATH)
categories = []

for filename in filenames:
    name_upper = filename.upper()
    if 'HA' in name_upper: categories.append('Happy')
    elif 'SA' in name_upper: categories.append('Sad')
    elif 'AN' in name_upper: categories.append('Anger')
    elif 'SU' in name_upper: categories.append('Surprise')
    elif 'FE' in name_upper: categories.append('Fear')
    elif 'DI' in name_upper: categories.append('Disgust')
    elif 'NE' in name_upper: categories.append('Neutral')
    else: categories.append('Skip')

df = pd.DataFrame({'filename': filenames, 'category': categories})
df = df[df['category'] != 'Skip']

# Stratify ensures we have equal emotions in train and validation
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['category'])

# ==========================================
# 2. GENERATORS (MobileNet Spec)
# ==========================================
IMG_SIZE = 224 
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Specific for MobileNetV2
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    IMAGE_FOLDER_PATH, 
    x_col='filename',
    y_col='category',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

validation_generator = val_datagen.flow_from_dataframe(
    validate_df, 
    IMAGE_FOLDER_PATH, 
    x_col='filename',
    y_col='category',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=BATCH_SIZE
)

# ==========================================
# 3. MODEL: MobileNetV2
# ==========================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Prevents overfitting
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# 4. CALLBACKS (The Safety Net)
# ==========================================
# 1. Save the best model ONLY (if val_accuracy improves)
checkpoint = ModelCheckpoint('best_emotion_model.h5', 
                             monitor='val_accuracy', 
                             save_best_only=True, 
                             mode='max', 
                             verbose=1)

# 2. Stop if no improvement for 5 epochs
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=5, 
                           restore_best_weights=True)

# 3. Reduce learning rate if stuck
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=3, 
                              min_lr=0.00001)

# ==========================================
# 5. TRAINING (CORRECTED)
# ==========================================
print(f"Training steps: {len(train_generator)}")
print(f"Validation steps: {len(validation_generator)}")

history = model.fit(
    train_generator,
    epochs=20,
    # Use len() to get the exact number of batches available
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ==========================================
# 6. RESULTS
# ==========================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# ==========================================
# 7. FINE-TUNING (The "Boost" Step)
# ==========================================
print("\nStarting Fine-Tuning to boost accuracy...")

# Unfreeze the base model
base_model.trainable = True

# MobileNetV2 has ~155 layers. We want to train only the top 50.
# We freeze the bottom 100 layers to keep the basic shapes (lines, circles) stable.
for layer in base_model.layers[:100]:
    layer.trainable = False

# CRITICAL: We must recompile with a VERY LOW learning rate.
# If the rate is too high, we will destroy the progress we made.
model.compile(optimizer=Adam(learning_rate=1e-5),  # 0.00001 (Very slow & precise)
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train for 20 more epochs
history_fine = model.fit(
    train_generator,
    epochs=20, # 20 more epochs
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    steps_per_epoch=len(train_generator),
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ==========================================
# 8. FINAL PLOT (Combined History)
# ==========================================
# Combine the history from the first run and this run to see the full picture
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.plot([19, 19], plt.ylim(), label='Start Fine Tuning') # Mark where we switched
plt.legend(loc='lower right')
plt.title('Total Accuracy History')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.plot([19, 19], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Total Loss History')
plt.show()

# Save the final polished model
model.save('emotion_model_final.h5')
print("Final model saved as 'emotion_model_final.h5'")