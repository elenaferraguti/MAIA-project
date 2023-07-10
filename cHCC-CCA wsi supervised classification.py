#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:09:59 2023

@author: Elena
"""

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'c:\path\to\openslide-win64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import openslide
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


slide = openslide.OpenSlide('/home/Elena/Desktop/MAIA project/WSI images/18AG 01666.4  MALDI.svs')

#slide_name = slide.slide_id
dimensions = slide.dimensions
level_count = slide.level_count
level_dimensions = slide.level_dimensions

level = 1  # Choose the desired level (0 is the highest resolution level)
image = slide.read_region((0, 0), level, slide.level_dimensions[level])
rotated_image = image.rotate(180)

#reduced_resolution = rotated_image.resize((image.width // 100, image.height // 100), resample=Image.BICUBIC)
Image.MAX_IMAGE_PIXELS = None
roi = (5000, 3000, 19100, 14100)
cropped_image = rotated_image.crop(roi)


plt.imshow(cropped_image)
plt.grid(True)
plt.show()

#%%
from pyimzml.ImzMLParser import ImzMLParser

hcc_cca_18ag = ImzMLParser('/home/Elena/Desktop/MAIA project/Testing data/First mixed maldi + annotation/18ag 01666-20012022.imzML', include_spectra_metadata = None )
coord_mix_18 = enumerate(hcc_cca_18ag.coordinates)

(mz, spectrum0) = hcc_cca_18ag.getspectrum(0)
(mz, spectrum1) = hcc_cca_18ag.getspectrum(1)
spectra_mix_18 = []
XCoord_mix_18 = []
YCoord_mix_18 = []
for i, (x,y,z) in coord_mix_18:
        (mz, spectrum) = hcc_cca_18ag.getspectrum(i)
        spectra_mix_18.append(spectrum)
        XCoord_mix_18.append(x)
        YCoord_mix_18.append(y)
spectra_mix_18 = np.asarray(spectra_mix_18)

pred18 = np.load('/home/Elena/Desktop/MAIA project/Predictions MALDI with wrong mz/18ag.npy')
plt.imshow(pred18)
plt.grid(True)
plt.show()
#%%

import openslide
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your SVS dataset
# dataset_path = '/home/Elena/Desktop/MAIA project/WSI images/18AG 01666.4  MALDI.svs'

# Open an SVS file using openslide
def open_svs_file(file_path):
    slide = openslide.open_slide(file_path)
    return slide

# Extract patches from the SVS file
def extract_patches(width, height, patch_size, stride):
    patches = []
    for i in range(0, width-patch_size+1, stride):
        for j in range(0, height-patch_size+1, stride):
            patch = slide.read_region((i, j), 0, (patch_size, patch_size))
            patch = patch.convert("RGB")
            patch = np.array(patch)
            patches.append(patch)
    return patches

# Load and preprocess the dataset
def load_dataset(dataset_path, patch_size, stride):
    slide_files = ['slide1.svs', 'slide2.svs', 'slide3.svs']  # Replace with your SVS slide filenames

    X = []
    y = []

    for slide_file in slide_files:
        slide_path = dataset_path + '/' + slide_file
        slide = open_svs_file(slide_path)
        patches = extract_patches(slide, patch_size, stride)
        #label = get_label(slide_file)  # Implement your label extraction logic
        X.extend(patches)
        #y.extend([label] * len(patches))

    X = np.array(X)
    y = np.array(y)

    return X, y

# Split the dataset into training and testing sets
def split_dataset(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Create the classifier model
def create_classifier_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    classifier_model = Model(inputs=base_model.input, outputs=predictions)
    return classifier_model

# Compile the classifier model
def compile_classifier_model(classifier_model, learning_rate):
    optimizer = Adam(learning_rate=learning_rate)
    classifier_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the classifier model
def train_classifier_model(classifier_model, X_train, y_train, X_val, y_val, batch_size, num_epochs):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    history = classifier_model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=num_epochs,
        validation_data=(X_val, y_val)
    )

    return history

#%%

width, height = cropped_image.size
patch_size = 100

patches = []

# Iterate over the image and extract patches
for y in range(0, height, patch_size):
    for x in range(0, width, patch_size):
        # Extract patch at the specified coordinates
        y_pred = int(y/100)
        x_pred = int(x/100)
        if pred18[y_pred,x_pred] == (0 | 1):
            patch = cropped_image.crop((x, y, x + patch_size, y + patch_size))
            patch = patch.convert("RGB")
            patch = np.array(patch)
            patches.append(patch)

y=[]

for i in range(0, pred18.shape[0]):
    for j in range(0, pred18.shape[1]):
        if pred18[i,j] == (0 | 1):
            y.append(pred18[i,j])
            
X = np.array(patches)
y = np.array(y)

#%%

input_shape = (100,100,3)
X_train, X_test, y_train, y_test = split_dataset(X, y, 1000)
classifier = create_classifier_model(input_shape, 2)
compile_classifier_model(classifier, 0.001)

history = train_classifier_model(classifier, X_train, y_train, X_test, y_test, batch_size=16, num_epochs=100)

#%%

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#%%

slide_new = openslide.OpenSlide('/home/Elena/Desktop/MAIA project/WSI images/20AG02608-06.svs')

dimensions = slide_new.dimensions
level_count = slide_new.level_count
level_dimensions = slide_new.level_dimensions

level = 1  # Choose the desired level (0 is the highest resolution level)
image_new = slide_new.read_region((0, 0), level, slide_new.level_dimensions[level])
rotated_image_new = image_new.rotate(180)

#reduced_resolution = rotated_image.resize((image.width // 100, image.height // 100), resample=Image.BICUBIC)
Image.MAX_IMAGE_PIXELS = None
roi = (0, 0, 8000, 11000)
cropped_image_new = rotated_image_new.crop(roi)

plt.imshow(cropped_image_new)
plt.grid(True)
plt.show()

#%%

width, height = cropped_image_new.size
patch_size = 100

patches = []

# Iterate over the image and extract patches
for y in range(0, height, patch_size):
    for x in range(0, width, patch_size):
        # Extract patch at the specified coordinates
            patch = cropped_image_new.crop((x, y, x + patch_size, y + patch_size))
            patch = patch.convert("RGB")
            patch = np.array(patch)
            patches.append(patch)
            
X_test = np.array(patches)

#%%

pred = classifier(X_test)

