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

df_pure = pd.read_csv('C:/Users/Elena/OneDrive/Desktop/HealthTech/Tesi/Dati/training_preproc_dataframe_norm_peak_max.csv', index_col=[0])


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

knn40 = KNeighborsClassifier(n_neighbors=30) 

knn40.fit(X_train, y_train)
y_pred = knn40.predict(X_test)
y_pred_train = knn40.predict(X_train)
scores = cross_val_score(knn40, X, y, cv=3)
scores = scores.tolist()

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
scores.append(cross_val_score(NaiveB, X, y, cv=3))

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

DecTree = DecisionTreeClassifier(criterion='entropy', max_depth=50, min_samples_split=5, min_samples_leaf=2)

DecTree.fit(X_train, y_train)
y_pred_train = DecTree.predict(X_train)
y_pred = DecTree.predict(X_test)
scores.append(cross_val_score(DecTree, X, y, cv=3))


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

rand_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, min_samples_split=10, min_samples_leaf=10)

rand_forest.fit(X_train, y_train)
y_pred_train = rand_forest.predict(X_train)
y_pred = rand_forest.predict(X_test)
scores.append(cross_val_score(rand_forest, X, y, cv=3))

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

LogReg = LogisticRegression(C=100, solver='liblinear', max_iter=10000, penalty='l1')

LogReg.fit(X_train, y_train)
y_pred_train = LogReg.predict(X_train)
y_pred = LogReg.predict(X_test)
scores.append(cross_val_score(LogReg, X, y, cv=3))

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

svm = SVC(kernel='poly',C=100, gamma=1, degree=2)

svm.fit(X_train, y_train)
y_pred_train = svm.predict(X_train)
y_pred = svm.predict(X_test)
scores.append(cross_val_score(svm, X, y, cv=3))

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

data = [np.array(scores[0:4]), scores[5], scores[6], scores[7], scores[8], scores[9]]
import matplotlib.pyplot as plt

fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)

ax.set_xticklabels(['KNN', 'NaiveBayes',
                    'DecTree', 'RandFor',
                    'LogReg', 'SVM'])
plt.xticks(fontsize=12)
 
# show plot
plt.show()

#%% Loading, reading and preprocessing of MALDI data: tissue sample of combined tumours

hcc_cca = ImzMLParser('/home/Elena/Desktop/MAIA project/Testing data/18ag 01666-20012022.imzML', include_spectra_metadata = None )
coord_mix = enumerate(hcc_cca.coordinates)


(mz, spectrum0) = hcc_cca.getspectrum(0)
(mz, spectrum1) = hcc_cca.getspectrum(1)
spectra_mix = []
XCoord_mix = []
YCoord_mix = []
for i, (x,y,z) in coord_mix:
        (mz, spectrum) = hcc_cca.getspectrum(i)
        spectra_mix.append(spectrum)
        XCoord_mix.append(x)
        YCoord_mix.append(y)
spectra_mix = np.asarray(spectra_mix)

peak_mix = peak_mz(spectra_mix)
peak_baseline_mix = baseline_removal(peak_mix)
norm_mix = normalization(peak_baseline_mix, 'max peak', None)


#%% Dataframe cHCC-CCA

df_mix = pd.DataFrame(norm_mix, columns = mz_string)

#%% Predictions on MALDI data of cHCC-CCA with the trained algorithms

y_knn_18 = knn40.predict(df_mix)
y_naiveb_18 = NaiveB.predict(df_mix)
y_dectree_18 = DecTree.predict(df_mix)
y_randfor_18 = rand_forest.predict(df_mix)
y_logreg_18 = LogReg.predict(df_mix)
y_svm_18 = svm.predict(df_mix)

#%% Predictions plot
image_mixed = getionimage(hcc_cca, 1200, tol=0.1, z=1)
dimensions = image_mixed.shape
import numpy as np
predicted = 2*np.ones(dimensions)
for i in range(0, len(y_naiveb_18)):
    indx = XCoord_mix[i]-1
    indy = YCoord_mix[i]-1
    
    predicted[indy, indx] = y_naiveb_18[i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 4))
plt.imshow(predicted)


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

X_mix = df_mix.iloc[:,:].values

X_encode = encoder.predict(X_mix)
yhat_mix = model.predict(X_encode)


#%% scatterplot 2D
import seaborn as sns
import random

index = random.sample(range(0, X_test_encode.shape[0]), 100)
X_test_encode_abs = abs(X_test_encode)

p1=sns.scatterplot(x=X_test_encode_abs[:,3], y=X_test_encode_abs[:,6],
              alpha=.3, 
              legend=False,
              data=X_test_encode_abs[index,:]);

# add annotations one by one with a loop
for line in index:
    if line < 44335:
      p1.text(X_test_encode_abs[line,3], X_test_encode_abs[line,6], line, horizontalalignment='left', size='medium', color='m')
    else:
      p1.text(X_test_encode_abs[line,3], X_test_encode_abs[line,6], line, horizontalalignment='left', size='medium', color='c')
      
#%% scatterplot 3D

import matplotlib.pyplot as plt
import numpy as np

Xax = X_test_encode_abs[:,10]
Yax = X_test_encode_abs[:,14]
Zax = X_test_encode_abs[:,21]
cdict = {0:'m',1:'c'}
label = {0:'cca',1:'hcc'}
y = y_test

fig = plt.figure(figsize=(14,9))
ax = fig.add_subplot(111, 
                     projection='3d')
 
for l in np.unique(y):
 ix=np.where(y==l)
 ax.scatter(Xax[ix], 
            Yax[ix], 
            Zax[ix], 
            c=cdict[l], 
            s=60,
           label=label[l])

ax.set_xlabel("feature 3", 
              fontsize=12)
ax.set_ylabel("feature 6", 
              fontsize=12)
ax.set_zlabel("feature 10", 
              fontsize=12)
 
ax.view_init(60, 60)
ax.legend()
plt.title("3D PCA plot")
plt.show()


