# Cell 1: Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Download Data and Set Variables
# Download the dataset
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip

# Unzip the dataset
import os
import zipfile

local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# Set directory paths
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Set image dimensions and batch size
IMG_HEIGHT = 150
IMG_WIDTH = 150
batch_size = 32

# Cell 3: Create Image Data Generators
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

validation_data_gen = validation_image_generator.flow_from_directory(
    directory=validation_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

test_data_gen = test_image_generator.flow_from_directory(
    directory=test_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    shuffle=False
)

# Cell 4: Plot Images (Provided Function)
def plotImages(images_arr, probabilities=None):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(5*len(images_arr), 5))
    if len(images_arr) == 1:
        axes = [axes]
    for img, ax, prob in zip(images_arr, axes, probabilities or [None]*len(images_arr)):
        ax.imshow(img)
        if prob is not None:
            ax.set_title(f"{'Dog' if prob >= 0.5 else 'Cat'}: {abs(prob-0.5)*2*100:.1f}%")
        ax.axis('off')
    plt.show()

# Display 5 random training images
sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

# Cell 5: Add Data Augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data_gen = train_image_generator.flow_from_directory(
    directory=train_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

# Cell 6: Visualize Augmented Images
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Cell 7: Create and Compile the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Cell 8: Train the Model
epochs = 20
steps_per_epoch = train_data_gen.samples // batch_size
validation_steps = validation_data_gen.samples // batch_size

history = model.fit(
    x=train_data_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_data_gen,
    validation_steps=validation_steps
)

# Cell 9: Visualize Accuracy and Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Cell 10: Predict on Test Images
probabilities = model.predict(test_data_gen).flatten()
probabilities = [float(prob) for prob in probabilities]

test_images = []
for i in range(test_data_gen.samples):
    img, _ = next(test_data_gen)
    test_images.append(img[0])

plotImages(test_images[:50], probabilities[:50])

# Cell 11: Evaluate Test Accuracy
test_loss, test_accuracy = model.evaluate(test_data_gen)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Check if challenge is passed
if test_accuracy >= 0.63:
    print("Congratulations! You passed the challenge with >= 63% accuracy!")
else:
    print("Test accuracy is below 63%. Try increasing epochs, adjusting the model, or adding more data augmentation.")