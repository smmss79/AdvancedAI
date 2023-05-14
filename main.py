# In the name of GOD

# In this project, we attempted to classify EEG signals for MI based BCI 
# 2023/1/18
# by S.M.M.S.S

#  ****************************************   steps of work   *****************************************

# Data Loading -> 1. load data from file  2. seperation according to tasks 

# preprocessing ->  1. Artifact reduction  2. Noise reduction  3. remove power line noise  4. bandpass frequency filter 

# feature extraction: 
## -- define window for feature extraction
## a.time domain -> (mean-range-max-min-std-var-kutosis)
## b.frequency domain -> (PSD)
## c.spatial domain
## d.Wavelet based features
## e.Entropy features
## f.fractal features
## g.complexity features
## h.Connectivity features
## i.Weight vector of input to hideen layer of an ANN 

# feature selection (optional):
## a.metaheuristic
## b. traditional feature selection methods

# classification:
## a. SVM + (kernel trick)
## b. MLP
## c. LSTM

# Evaluation & Results -> 1. evaluation metrics   2. representation 3. validation 


# ************************************ Data Loading *************************************

## !!! TODO: read signal for all (sample set-> 20) subjects and records 
## !! TODO: spli data into 10 train 3 test 7 validation
## ! TODO: feature extraction in window

from EEGHelper.EDFFile import get_records_directory, load_records_to_list,seperate_by_labels
from EEGHelper.SignalRepresent import plot_one_subject,draw_annotations





## ****************  initializiation
channel_name = ['FC5.', 'FC3.', 'FC1.', 'FCZ.', 'FC2.', 'FC4.', 'FC6.', 'C5..', 'C3..', 'C1..', 'CZ..', 'C2..', 'C4..', 'C6..', 'CP5.', 'CP3.', 'CP1.', 'CPZ.', 'CP2.', 'CP4.', 'CP6.', 'FP1.', 'FPZ.', 'FP2.', 'AF7.', 'AF3.', 'AFZ.', 'AF4.', 'AF8.', 'F7..', 'F5..', 'F3..', 'F1..', 'FZ..', 'F2..', 'F4..', 'F6..', 'F8..', 'FT7.', 'FT8.', 'T7..', 'T8..', 'T9..', 'T10.', 'TP7.', 'TP8.', 'P7..', 'P5..', 'P3..', 'P1..', 'PZ..', 'P2..', 'P4..', 'P6..', 'P8..', 'PO7.', 'PO3.', 'POZ.', 'PO4.', 'PO8.', 'O1..', 'OZ..', 'O2..', 'IZ..']
sample_rate = 160
subject_count = 109
sample_subject_list = [0]
record_count = 14
channel_count = len(channel_name)
record_file_directory = get_records_directory('Dataset\RECORDS',subject_count,record_count)


## ************** Load data from dataset to a list object 

records_data = load_records_to_list(sample_subject_list, record_count, record_file_directory, channel_name)

## print(len(records_data[0][0][0]))

## data representation
## plot_one_subject(figsize_tuple= (50,3), fig_dpi=100, isPlotNotShow=1,records_data=records_data,ch_start=0,ch_stop=3,sampling_frequency= sample_rate)

## ********** Labelling data 

labelled_signal = seperate_by_labels(records_data, sample_rate)



## ***************************************** window and feature creation ***************************

from EEGHelper.SignalProcess import extract_features, extract_features_labelled
import time

## *************  initializiation

# inits =[16,32,50,64,100,128,150,160,200,256,300,512]
# res_x,res_y = [], []

# for win in inits:
window_size = 256
# task_count = 3
task_list = [0,1,2,6,10] 
ch_list = list(range(14,21))+list(range(44,64))

# ch_list = range(64)



## ************ feature creation 



## print(records_data[0][0])
## print(sliding_window(records_data[0][0][0], window_size))



# a= time.time_ns()

# feature_list = extract_features(records_data,task_count,window_size,ch_list)

# print(time.time_ns()-a)

## print(len(feature_list[0]))


a= time.time_ns()

features,labels = extract_features_labelled(labelled_signal,task_list,window_size,ch_list)

print("feature_extraction_time: ",time.time_ns()-a)

# import numpy as np
# print(np.shape(features))

## *************************************** classification *******************************

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from EEGHelper.classification import compare_classifiers,voting_classification
import numpy as np
import matplotlib.pyplot as plt

## ! TODO start labelling features and classify with MLP classififer



for temp_count in range(10):

    print("\nTrial:", temp_count)

    plt.subplot(5,2,temp_count+1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size=0.2 ,random_state=np.random.RandomState())
    voting_classification(X_train, X_test, y_train, y_test)

    compare_classifiers(X_train, X_test, y_train, y_test)

plt.show()



# clf = MLPClassifier(hidden_layer_sizes=(10,8,5,3),random_state=1, max_iter=len(X_train)*50)
# clf.fit(X_train,y_train)


## res_x.append(win)
## res_y.append(clf.score(X_test,y_test)*100)

# import matplotlib.pyplot as plt
## plt.scatter(res_x, res_y)
## plt.show()


# print(X_test[0])
# print(y_test[0])
# print(clf.predict([X_test[0]]))
# print(clf.score(X_test,y_test)*100)




# TODO: use classification.py to compare classifiers







## *************************************** save model *******************************


# import joblib
# # save the model to disk
# filename = 'finalized_model.sav'
# joblib.dump(results, filename)
 
# # some time later...
 
# # load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)