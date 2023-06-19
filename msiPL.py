#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:14:03 2023

@author: Elena
"""
import numpy as np
import pandas as pd
from keras.layers import Lambda, Input, Dense, ReLU, BatchNormalization
from keras.models import Model
from keras.losses import  categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K

df_pure = pd.read_csv('/home/Elena/Desktop/MAIA project/Dataframe/training_preproc_dataframe.csv', index_col=[0])
X = df_pure.iloc[:, 0:-1].values
y = df_pure.iloc[:, -1].values

nSpecFeatures = X.shape[1]
input_shape = X.shape
intermediate_dim = 100
latent_dim = 30


def sampling(args):
    """
    Reparameterization trick by sampling from a continuous function (Gaussian with an auxiliary variable ~N(0,1)).
    [see Our methods and for more details see arXiv:1312.6114]
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim)) # random_normal (mean=0 and std=1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
input_shape = (nSpecFeatures, )
inputs = Input(shape=input_shape, name='encoder_input')
h = Dense(intermediate_dim)(inputs)
h = BatchNormalization()(h)
h = ReLU()(h)
z_mean = Dense(latent_dim, name = 'z_mean')(h)
z_mean = BatchNormalization()(z_mean)
z_log_var = Dense(latent_dim, name = 'z_log_var')(h)
z_log_var = BatchNormalization()(z_log_var)

# Reparametrization Tric:
z = Lambda(sampling, output_shape = (latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name = 'encoder')
print("==== Encoder Architecture...")
encoder.summary()
plot_model(encoder, to_file='VAE_BN_encoder.png', show_shapes=True)
        
# =========== 2. Encoder Model================
latent_inputs = Input(shape = (latent_dim,), name='Latent_Space')
hdec = Dense(intermediate_dim)(latent_inputs)
hdec = BatchNormalization()(hdec) 
hdec = ReLU()(hdec)
outputs = Dense(nSpecFeatures, activation = 'sigmoid')(hdec)
decoder = Model(latent_inputs, outputs, name = 'decoder') 
print("==== Decoder Architecture...") 
decoder.summary()       
plot_model(decoder, to_file='VAE_BN__decoder.png', show_shapes=True)
        
#=========== VAE_BN: Encoder_Decoder ================
outputs = decoder(encoder(inputs)[2])
VAE_BN_model = Model(inputs, outputs, name='VAE_BN')
        
# ====== Cost Function (Variational Lower Bound)  ==============
"KL-div (regularizes encoder) and reconstruction loss (of the decoder): see equation(3) in our paper"
# 1. KL-Divergence:
kl_Loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_Loss = K.sum(kl_Loss, axis=-1)
kl_Loss = kl_Loss * (-0.0005)
# 2. Reconstruction Loss
reconstruction_loss = categorical_crossentropy(inputs,outputs) # Use sigmoid at output layer
#reconstruction_loss = reconstruction_loss * nSpecFeatures
        
# ========== Compile VAE_BN model ===========
model_Loss = K.mean(reconstruction_loss + kl_Loss)
VAE_BN_model.add_loss(model_Loss)
VAE_BN_model.compile(optimizer='adam')

#%%
from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

cc70 = ImzMLParser('/home/Elena/Desktop/MAIA project/Training data/tr_20170713_e_70_1.imzML', include_spectra_metadata = None )
cc72 = ImzMLParser('/home/Elena/Desktop/MAIA project/Training data/tr20170802_e72_1.imzML', include_spectra_metadata = None)
chc93 = ImzMLParser('/home/Elena/Desktop/MAIA project/Training data/tr_20170823_e_93_1.imzML', include_spectra_metadata = None)

coord70 = enumerate(cc70.coordinates)
coord72 = enumerate(cc72.coordinates)
coord93 = enumerate(chc93.coordinates)

(mz, spectrum0) = cc70.getspectrum(0)
(mz, spectrum1) = cc70.getspectrum(1)
Spec_Data70 = []
XCoord70 = []
YCoord70 = []
for i, (x,y,z) in coord70:
        (mz, spectrum) = cc70.getspectrum(i)
        Spec_Data70.append(spectrum)
        XCoord70.append(x)
        YCoord70.append(y)
        # if i%1000 == 0:
        #     print(i)
Spec_Data70 = np.asarray(Spec_Data70)
 
mz_int = np.arange(start=620, stop=3201)

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

peaks = []
for i in range(620, 3201):
    ind = find_indices(mz, lambda e: (e >= i) & (e < (i+1)))
    peak_list = []
    for j in range(0, Spec_Data70.shape[0]):
        peak = max(Spec_Data70[j,ind])
        peak_list.append(peak)
    peaks.append(peak_list)

MSI_train = np.asarray(peaks) 
nSpecFeatures = len(mz)
if MSI_train.shape[1] != nSpecFeatures:
    MSI_train = np.transpose(MSI_train)
xLocation = np.array(XCoord70)
yLocation = np.array(YCoord70)
col = max(np.unique(XCoord70))
row = max(np.unique(YCoord70))
im = np.zeros((col,row))
mzId = np.argmin(np.abs(mz_int[:] - 620))
for i in range(len(xLocation)):
    im[ (xLocation[i]-1), (yLocation[i]-1)] = MSI_train[i,mzId] #image index starts at 0 not 1
plt.imshow(im);plt.colorbar()

#%%

(mz, spectrum0) = cc72.getspectrum(0)
(mz, spectrum1) = cc72.getspectrum(1)
Spec_Data72 = []
XCoord72 = []
YCoord72 = []
for i, (x,y,z) in coord72:
        (mz, spectrum) = cc72.getspectrum(i)
        Spec_Data72.append(spectrum)
        XCoord72.append(x)
        YCoord72.append(y)
        # if i%1000 == 0:
        #     print(i)
Spec_Data72 = np.asarray(Spec_Data72)

(mz, spectrum0) = chc93.getspectrum(0)
(mz, spectrum1) = chc93.getspectrum(1)
Spec_Data93 = []
XCoord93 = []
YCoord93 = []
for i, (x,y,z) in coord93:
        (mz, spectrum) = chc93.getspectrum(i)
        Spec_Data93.append(spectrum)
        XCoord93.append(x)
        YCoord93.append(y)
        # if i%1000 == 0:
        #     print(i)
Spec_Data93 = np.asarray(Spec_Data93)

peak_72 = []
for i in range(620, 3201):
    ind = find_indices(mz, lambda e: (e >= i) & (e < (i+1)))
    peak_list = []
    for j in range(0, Spec_Data72.shape[0]):
        peak = max(Spec_Data72[j,ind])
        peak_list.append(peak)
    peak_72.append(peak_list)
    
peak_93 = []
for i in range(620, 3201):
    ind = find_indices(mz, lambda e: (e >= i) & (e < (i+1)))
    peak_list = []
    for j in range(0, Spec_Data93.shape[0]):
        peak = max(Spec_Data93[j,ind])
        peak_list.append(peak)
    peak_93.append(peak_list)
    
peak_72 = np.asarray(peak_72)
peak_93 = np.asarray(peak_93)

    
MSI_train = np.vstack((MSI_train, peak_72.T, peak_93.T))


#%%

import time

# ============= Model Training =================
""" The training processes involves: 
	epochs: 100 iterations
	batch_size: a randomly-shuffled subset of 128 spectra is loaded at a time into the RAM 
	This phase will run faster if a GPU is utilized
 """
try:
    start_time = time.time()
    history = VAE_BN_model.fit(X, epochs=100, batch_size=128, shuffle="batch")   
    plt.plot(history.history['loss'])
    plt.ylabel('loss'); plt.xlabel('epoch')
    print("--- %s seconds ---" % (time.time() - start_time))
    VAE_BN_model.save_weights('TrainedModel_msiPL_train70.h5')
except MemoryError as error:
    import psutil
    Memory_Information = psutil.virtual_memory()
    print('>>> There is a memory issue: and here are a few suggestions:')
    print('>>>>>> 1- Make sure that you are using  python 64-bit.')
    print('>>>>>> 2- use a lower value for the batch_size (default is 128).')
    print('**** Here is some information about your memory (MB):', Memory_Information)
    
#%%

# ============= Model Predictions ===============
encoded_imgs = encoder.predict(MSI_train) # Learned non-linear manifold
decoded_imgs = VAE_BN_model.predict(MSI_train) # Reconstructed Data
dec_TIC = np.sum(decoded_imgs, axis=-1)

#%%

from sklearn.metrics import mean_squared_error

# ======= Calculate mse between orig & rec. data =====
""" The mean squared error (mse): 
	the mse is used to evaluate the quality of the reconstructed data"""
mse = mean_squared_error(MSI_train,decoded_imgs)
meanSpec_Rec = np.mean(decoded_imgs,axis=0) 
print('mean squared error(mse)  = ', mse)
meanSpec_Orig = np.mean(MSI_train,axis=0) # TIC-norm original MSI Data
N_DecImg = decoded_imgs/dec_TIC[:,None]  # TIC-norm reconstructed MSI  Data
meanSpec_RecTIC = np.mean(N_DecImg,axis=0)
plt.plot(mz_int, meanSpec_Orig); plt.plot(mz_int, meanSpec_RecTIC,color = [1.0, 0.5, 0.25]); 
plt.title('TIC-norm distribution of average spectrum: Original and Predicted')
