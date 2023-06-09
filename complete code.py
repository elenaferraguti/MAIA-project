#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:49:52 2023

@author: Elena
"""

#%% Libraries

import pandas as pd
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage
import matplotlib.pyplot as plt

#%% Loading and reading of MALDI data: TMAs of pure tumour samples

cc70 = ImzMLParser('/home/Elena/Desktop/MAIA project/Training data/tr_20170713_e_70_1.imzML', include_spectra_metadata = None )
cc72 = ImzMLParser('/home/Elena/Desktop/MAIA project/Training data/tr20170802_e72_1.imzML')
chc93 = ImzMLParser('/home/Elena/Desktop/MAIA project/Training data/tr_20170823_e_93_1.imzML')

#%% Coordinates of spectra

coord70 = enumerate(cc70.coordinates)
coord72 = enumerate(cc72.coordinates)
coord93 = enumerate(chc93.coordinates)

#%% Spectra extraction as arrays

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
Spec_Data70 = np.asarray(Spec_Data70)

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
Spec_Data93 = np.asarray(Spec_Data93)

#%% Peak picking for each m/z value

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def peak_mz(spectra_matrix):
    peaks = []
    for i in range(620, 3201):
        ind = find_indices(mz, lambda e: (e >= i) & (e < (i+1)))
        peak_list = []
        for j in range(0, spectra_matrix.shape[0]):
            peak = max(spectra_matrix[j,ind])
            peak_list.append(peak)
        peaks.append(peak_list)
    return np.asarray(peaks)

peak_70 = peak_mz(Spec_Data70)
peak_72 = peak_mz(Spec_Data72)
peak_93 = peak_mz(Spec_Data93)

mz_int = np.arange(start=620, stop=3201)

#%% Baseline removal

from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def baseline_removal(spectra_matrix):
    peak_baseline = []
    for i in range(0, spectra_matrix.shape[1]):
        baseline = baseline_als(spectra_matrix[:, i], 10e2, 0.001)
        peak_baseline.append(spectra_matrix[:, i] - baseline)
    peak_baseline = np.asarray(peak_baseline)
    return peak_baseline

peak_baseline_70 = baseline_removal(peak_70)
peak_baseline_72 = baseline_removal(peak_72)
peak_baseline_93 = baseline_removal(peak_93)


#%% Spectra normalization

def compute_mean_intensity_spectra(spectra):
     intensities = []
     for i in range(0, spectra.shape[0]):
         intensities.append(spectra[i,:])
     mean_intensity = np.mean(intensities)
     return mean_intensity, intensities

def TIC_normalize_spectrum(spectrum, method, mean_intensity):
    if method == 'mean':
        intensities = spectrum
        mean_instance_intensity = np.mean(intensities)
        scaling = mean_instance_intensity / mean_intensity
        return spectrum * scaling
    elif method == 'sum':
        intensities = spectrum
        scaling = 1.0 / np.sum(intensities)
        return spectrum * scaling
    else:
        raise RuntimeError(
                f'Unexpected normalization method "{method}"')


def normalization(spectra_matrix, method_norm, method_tic):
    if method_norm == 'TIC':
        mean_intensity_spectra, intensities = compute_mean_intensity_spectra(spectra_matrix)
        normalized_spectra = []
        for i in range(0, spectra_matrix.shape[0]):
            norm = TIC_normalize_spectrum(spectra_matrix[i,:], method_tic , mean_intensity_spectra)
            normalized_spectra.append(norm)
        normalized_spectra = np.asarray(normalized_spectra)
        
    elif method_norm == 'max peak':
        normalized_spectra = []
        for i in range(0, spectra_matrix.shape[0]):
            maxpeak = max(spectra_matrix[i,:])
            normalized_spectra.append(spectra_matrix[i,:]/maxpeak)
        normalized_spectra = np.asarray(normalized_spectra)
        
    return normalized_spectra
        
norm_70 = normalization(peak_baseline_70, 'max peak', None)
norm_72 = normalization(peak_baseline_72, 'max peak', None)
norm_93 = normalization(peak_baseline_93, 'max peak', None)


#%% Dataset

mz_string = []
for ii in range(0, len(mz_int)):
    mz_string.append(str(mz_int[ii]))
    
dataset = np.concatenate((norm_70, norm_72, norm_93), axis=0)
df_pure = pd.DataFrame(dataset, columns = mz_string)
labels = []
for ii in range(dataset.shape[0]):
    if ii < (norm_70.shape[0]+norm_72.shape[0]):
        labels.append(0)
    else:
        labels.append(1)
df_pure = df_pure.assign(Labels = labels)

#%% Train-test splitting (70%-30%)

from sklearn.model_selection import train_test_split

y = df_pure.iloc[:,-1]
X = df_pure.iloc[:,0:-1]
X_train, X_test, y_train, y_test = train_test_split(df_pure.iloc[:,0:-1], 
                                                    df_pure.iloc[:,-1], 
                                                    test_size = 0.30, #by default is 75%-25%
                                                    #shuffle is set True by default,
                                                    stratify = df_pure.iloc[:,-1],
                                                    random_state= 123)

#%% KNN

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

knn40 = KNeighborsClassifier(n_neighbors=40) 

knn40.fit(X_train, y_train)
y_pred = knn40.predict(X_test)
y_pred_train = knn40.predict(X_train)
scores = cross_val_score(knn40, X, y, cv=5)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn import metrics

print('***RESULTS ON TRAIN SET***')
print("precision: ", metrics.precision_score(y_train, y_pred_train)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_train, y_pred_train)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_train, y_pred_train)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_train, y_pred_train)) # (tp+tn)/m

print('***RESULTS ON TEST SET***')
print("precision: ", metrics.precision_score(y_test, y_pred)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_test, y_pred)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_test, y_pred)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_test, y_pred)) # (tp+tn)/m 

#%% NAIVE BAYES

from sklearn.naive_bayes import GaussianNB 

NaiveB = GaussianNB()

NaiveB.fit(X_train, y_train)
y_pred = NaiveB.predict(X_test)
y_pred_train = NaiveB.predict(X_train)
scores.append(cross_val_score(NaiveB, X, y, cv=5))

print('***RESULTS ON TRAIN SET***')
print("precision: ", metrics.precision_score(y_train, y_pred_train)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_train, y_pred_train)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_train, y_pred_train)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_train, y_pred_train)) # (tp+tn)/m

print('***RESULTS ON TEST SET***')
print("precision: ", metrics.precision_score(y_test, y_pred)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_test, y_pred)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_test, y_pred)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_test, y_pred)) # (tp+tn)/m
    
#%% DECISION TREE

from sklearn.tree import DecisionTreeClassifier

DecTree = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=2)

DecTree.fit(X_train, y_train)
y_pred_train = DecTree.predict(X_train)
y_pred = DecTree.predict(X_test)
scores.append(cross_val_score(DecTree, X, y, cv=5))


print('***RESULTS ON TRAIN SET***')
print("precision: ", metrics.precision_score(y_train, y_pred_train)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_train, y_pred_train)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_train, y_pred_train)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_train, y_pred_train)) # (tp+tn)/m

print('***RESULTS ON TEST SET***')
print("precision: ", metrics.precision_score(y_test, y_pred)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_test, y_pred)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_test, y_pred)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_test, y_pred)) # (tp+tn)/m

#%%
from sklearn import tree
r = tree.export_text(DecTree,feature_names=X_test.columns.tolist())
print(r)

tree.plot_tree(DecTree, fontsize=8)

#%% RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=6, min_samples_split=4, min_samples_leaf=2)

rand_forest.fit(X_train, y_train)
y_pred_train = rand_forest.predict(X_train)
y_pred = rand_forest.predict(X_test)
scores.append(cross_val_score(rand_forest, X, y, cv=5))

print('***RESULTS ON TRAIN SET***')
print("precision: ", metrics.precision_score(y_train, y_pred_train)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_train, y_pred_train)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_train, y_pred_train)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_train, y_pred_train)) # (tp+tn)/m

print('***RESULTS ON TEST SET***')
print("precision: ", metrics.precision_score(y_test, y_pred)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_test, y_pred)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_test, y_pred)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_test, y_pred)) # (tp+tn)/m


#%% LOGISTIC REGRESSION

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train) 
X_scaled = scaler.transform(X_train)

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(C=10, solver='lbfgs', max_iter=300000)

LogReg.fit(X_train, y_train)
y_pred_train = LogReg.predict(X_train)
y_pred = LogReg.predict(X_test)
scores.append(cross_val_score(LogReg, X, y, cv=5))

print('***RESULTS ON TRAIN SET***')
print("precision: ", metrics.precision_score(y_train, y_pred_train)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_train, y_pred_train)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_train, y_pred_train)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_train, y_pred_train)) # (tp+tn)/m

print('***RESULTS ON TEST SET***')
print("precision: ", metrics.precision_score(y_test, y_pred)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_test, y_pred)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_test, y_pred)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_test, y_pred)) # (tp+tn)/m


#%% SUPPORT VECTOR MACHINE

from sklearn.svm import SVC

svm = SVC(kernel='linear',C=1)

svm.fit(X_train, y_train)
y_pred_train = svm.predict(X_train)
y_pred = svm.predict(X_test)
scores.append(cross_val_score(svm, X, y, cv=5))

print('***RESULTS ON TRAIN SET***')
print("precision: ", metrics.precision_score(y_train, y_pred_train)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_train, y_pred_train)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_train, y_pred_train)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_train, y_pred_train)) # (tp+tn)/m

print('***RESULTS ON TEST SET***')
print("precision: ", metrics.precision_score(y_test, y_pred)) # tp / (tp + fp)
print("recall: ", metrics.recall_score(y_test, y_pred)) # tp / (tp + fn)
print("f1_score: ", metrics.f1_score(y_test, y_pred)) #F1 = 2 * (precision * recall) / (precision + recall)
print("accuracy: ", metrics.accuracy_score(y_test, y_pred)) # (tp+tn)/m

#%% Loading, reading and preprocessing of MALDI data: tissue sample of combined tumours

hcc_cca_18ag = ImzMLParser('/home/Elena/Desktop/MAIA project/Testing data/18ag 01666-20012022.imzML', include_spectra_metadata = None )
hcc_cca_20ag = ImzMLParser('/home/Elena/Desktop/MAIA project/Testing data/20ag02608-09122021.imzML', include_spectra_metadata = None )
coord_mix_18 = enumerate(hcc_cca_18ag.coordinates)
coord_mix_20 = enumerate(hcc_cca_20ag.coordinates)

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

(mz, spectrum0) = hcc_cca_20ag.getspectrum(0)
(mz, spectrum1) = hcc_cca_20ag.getspectrum(1)
spectra_mix_20 = []
XCoord_mix_20 = []
YCoord_mix_20 = []
for i, (x,y,z) in coord_mix_20:
        (mz, spectrum) = hcc_cca_20ag.getspectrum(i)
        spectra_mix_20.append(spectrum)
        XCoord_mix_20.append(x)
        YCoord_mix_20.append(y)
spectra_mix_20 = np.asarray(spectra_mix_20)

peak_mix_18 = peak_mz(spectra_mix_18)
peak_baseline_mix_18 = baseline_removal(peak_mix_18)
norm_mix_18 = normalization(peak_baseline_mix_18, 'max peak', None)

peak_mix_20 = peak_mz(spectra_mix_20)
peak_baseline_mix_20 = baseline_removal(peak_mix_20)
norm_mix_20 = normalization(peak_baseline_mix_20, 'max peak', None)

#%% Dataframe cHCC-CCA

df_mix_18 = pd.DataFrame(norm_mix_18, columns = mz_string)
df_mix_20 = pd.DataFrame(norm_mix_20, columns = mz_string)

#%% Predictions on MALDI data of cHCC-CCA with the trained algorithms

y_knn_18 = knn40.predict(df_mix_18)
y_naiveb_18 = NaiveB.predict(df_mix_18)
y_dectree_18 = DecTree.predict(df_mix_18)
y_randfor_18 = rand_forest.predict(df_mix_18)
y_logreg_18 = LogReg.predict(df_mix_18)
y_svm_18 = svm.predict(df_mix_18)

y_knn_20 = knn40.predict(df_mix_20)
y_naiveb_20 = NaiveB.predict(df_mix_20)
y_dectree_20 = DecTree.predict(df_mix_20)
y_randfor_20 = rand_forest.predict(df_mix_20)
y_logreg_20 = LogReg.predict(df_mix_20)
y_svm_20 = svm.predict(df_mix_20)

#%% Libraries for autoencoder

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot

#%% Autoencoder architecture

y = df_pure.iloc[:,-1]
X = df_pure.iloc[:,0:-1].values

# define encoder
inputs = Input(shape=(X.shape[1],))
# encoder level 1
e = Dense(X.shape[1]*2)(inputs)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(X.shape[1])(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
# n_bottleneck = round(float(X.shape[1]) / 2.0)
n_bottleneck = 30
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(X.shape[1])(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(X.shape[1]*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(X.shape[1], activation='linear')(d)
# define autoencoder model
model = Model(inputs=inputs, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')
# plot the autoencoder
plot_model(model, 'autoencoder_compress.png', show_shapes=True)

#%% Fit of the autoencoder

history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,y_test))
# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#%% Save the encoder

# define an encoder model (without the decoder)
encoder = Model(inputs=inputs, outputs=bottleneck)
plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
# save the encoder to file
encoder.save('encoder.h5')

#%% Evaluation of encoder performance with predictions

from tensorflow import keras
from sklearn.metrics import accuracy_score

encoder = keras.models.load_model('/home/Elena/Desktop/MAIA project/encoder.h5')
# encode the train data
encoder.compile()
#encoder.fit(X_train, y_train, epochs=200, batch_size=16, verbose=2, validation_data=(X_test,y_test))
X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)
# define the model
model = DecisionTreeClassifier()
# fit the model on the training set
model.fit(X_train_encode, y_train)
# make predictions on the test set
yhat = model.predict(X_test_encode)
# calculate classification accuracy
acc = accuracy_score(y_test, yhat)
print(acc)

#%% Features reduction for cHCC-CA MALDI

X_mix18 = df_mix_18.iloc[:,:].values
X_mix20 = df_mix_20.iloc[:,:].values

X_encode18 = encoder.predict(X_mix18)
yhat_18 = model.predict(X_encode18)

X_encode20 = encoder.predict(X_mix20)
yhat_20 = model.predict(X_encode20)


