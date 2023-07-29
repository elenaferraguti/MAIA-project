#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:41:14 2023

@author: Elena
"""
# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/CODES/openslide-win64-20230414/bin'

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

#%%
path = 'C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/WSI/21AG06292-25.svs'
slide = openslide.OpenSlide(path)

#slide_name = slide.slide_id
dimensions = slide.dimensions
level_count = slide.level_count
level_dimensions = slide.level_dimensions

level = 1  # Choose the desired level (0 is the highest resolution level)
image = slide.read_region((0, 0), level, slide.level_dimensions[level])
rotated_image = image.rotate(180)

#reduced_resolution = rotated_image.resize((image.width // 100, image.height // 100), resample=Image.BICUBIC)
Image.MAX_IMAGE_PIXELS = None
if path == 'C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/WSI/21AG06292-25.svs':
    roi = (6000, 1000, 14000, 6000)
    cropped_image = image.crop(roi)
elif path == 'C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/WSI/20AG02608-06.svs':
    roi = (500, 0, 8000, 12000)
    cropped_image = rotated_image.crop(roi)
elif path == 'C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/WSI/18AG 01666.4  MALDI.svs':
    roi = (2000, 1500, 19100, 14100)
    cropped_image = rotated_image.crop(roi)
elif path == 'C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/WSI/16AG06295.10 MALDI.svs' and level == 1:
    roi = (0, 8000, 28000, 20000)
    cropped_image = rotated_image.crop(roi)
elif path == 'C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/WSI/16AG06295.10 MALDI.svs' and level == 2:
    roi = (0, 2000, 7000, 5000)
    cropped_image = rotated_image.crop(roi)

plt.imshow(cropped_image)
plt.grid(True)
plt.show()


#%%

img = cropped_image.convert("RGB")
img_array = np.array(img)
img_data = img_array.reshape(-1,3)

#%% K-means clustering

# from sklearn.cluster import KMeans

# num_clusters = 3  # Number of clusters to create
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# labels = kmeans.fit_predict(img_data)

# #%%
# #width, height = cropped_image.size

# clustered_image = labels.reshape(cropped_image.size[1], cropped_image.size[0])
# plt.imshow(clustered_image)
# plt.title('Clustering Result')
# plt.axis('off')
# plt.colorbar()
# plt.show()


#%% Gaussian Mixture Model

from sklearn.mixture import GaussianMixture

num_clusters = 4  # Number of clusters to create
gmm = GaussianMixture(n_components=num_clusters, random_state=0)
gmm.fit(img_data)
labels = gmm.predict(img_data)

width, height = cropped_image.size

custom_colors = ['1', '2', '3', '4']

# Visualize the clustering results
clustered_image = labels.reshape(height, width)
plt.imshow(clustered_image, cmap=plt.cm.get_cmap('viridis', num_clusters))
plt.title('Clustering Result')
plt.colorbar(ticks=np.arange(num_clusters))
plt.clim(0, num_clusters - 1)
plt.axis('off')

colorbar = plt.gcf().axes[-1]
colorbar.set_yticklabels(custom_colors)

plt.show()
#%%

from pyimzml.ImzMLParser import ImzMLParser

hcc_cca_18ag = ImzMLParser('C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/Test/First mixed maldi + annotation/18ag 01666-20012022.imzML', include_spectra_metadata = None )
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

pred18 = np.load('C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/Predictions MALDI/18ag.npy')
plt.imshow(pred18)
plt.grid(True)
plt.show()

#%%
from skimage.segmentation import slic
from skimage.io import imread,imsave
import numpy as np

num_segments = 10
compactness = 10

segments = slic(img_array, n_segments=num_segments, compactness=compactness)

segmentation_result = np.zeros_like(img_array)
for segment_id in np.unique(segments):
    mask = segments == segment_id
    segmentation_result[mask] = np.mean(img_array[mask], axis=0)
    
#%%

segmentation_result_image = Image.fromarray(segmentation_result.astype(np.uint8))
plt.imshow(segmentation_result_image)
plt.show()




