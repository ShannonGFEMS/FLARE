#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:56:00 2019

@author: shannon
"""
#source venv/bin/activate
#Load the necessary python libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import sklearn as sklearn
import imblearn  as imblearn
from imblearn import under_sampling
from collections import Counter
plt.style.use('ggplot')

#Load the dataset
df = pd.read_csv('VIS_syntheticdata.csv', low_memory=False)

#Print the first 5 rows of the dataframe.
#print(df.head())

#Observe the shape of the dataframe.
#print(df.shape)

#Drop rows with null values in columns we want to use for the model. Alternatively, set them to 0.
df = df.dropna(subset=['namsxkd','lhdn','tsld','ld11','nganh_kd','fake_label2'])

classCounts = df['fake_label2'].value_counts()
classDict = {}

for k, v in classCounts.items():
    if v >= 500:
        classDict[k] = True
    else:
        classDict[k] = False

boolList = []
for item in df['fake_label2']:
    boolList.append(classDict[item])
    
df = df[boolList]

#It might be wise to check the size of df again here. Call line 21 again.

df[['namsxkd','lhdn','tsld','ld11','nganh_kd', 'fake_label2']] = df[['namsxkd','lhdn','tsld','ld11','nganh_kd','fake_label2']].apply(pd.to_numeric, errors='coerce')

#Downsample the majority class.
df_majority = df[df.fake_label2 == 0]
df_minority = df[df.fake_label2 == 1]

df_majority_downsampled = df_majority.sample(n=10000, random_state=42)

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

#Create numpy arrays for features and target
plotdata = df[['namsxkd','lhdn','tsld','ld11','nganh_kd']]
X = df_downsampled[['namsxkd','lhdn','tsld','ld11','nganh_kd']].values
y = df_downsampled['fake_label2'].values

'''
Downsampling only, feature importances and f-score
nganh_kd    0.787256
tsld        0.061934
ld11        0.061624
namsxkd     0.055209
lhdn        0.033977
dtype: float64
[[3702  299]
 [  93 1262]]
              precision    recall  f1-score   support

         0.0       0.98      0.93      0.95      4001
         1.0       0.81      0.93      0.87      1355

   micro avg       0.93      0.93      0.93      5356
   macro avg       0.89      0.93      0.91      5356
weighted avg       0.93      0.93      0.93      5356

'''
'''
# ClusterCentroids, this one doesn't work that well
X = df[['namsxkd','lhdn','tsld','ld11','nganh_kd']].values
y = df['fake_label2'].values
sampler = under_sampling.ClusterCentroids(ratio={0: 100, 1: 1})
X_rs, y_rs = sampler.fit_sample(X, y)
print('Cluster centriods undersampling {}'.format(Counter(y_rs)))
'''
'''
#Cluster Centroids only, with feature importances, f-score
Cluster centriods undersampling Counter({0.0: 100, 1.0: 1})
nganh_kd    0.416561
namsxkd     0.187365
ld11        0.166510
tsld        0.164446
lhdn        0.065118
dtype: float64
[[140820    244]
 [  1320     35]]
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99    141064
         1.0       0.13      0.03      0.04      1355

   micro avg       0.99      0.99      0.99    142419
   macro avg       0.56      0.51      0.52    142419
weighted avg       0.98      0.99      0.99    142419

'''

'''
#plot resampling
def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)
    plt.savefig('inputplot.png')
'''
'''
#SMOTE
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
X = df[['namsxkd','lhdn','tsld','ld11','nganh_kd']].values
y = df['fake_label2'].values
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit(X, y)
'''


#X = df[['namsxkd','lhdn','tsld','ld11','nganh_kd']].values
#y = df['fake_label2'].values

#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

#Rescaling the features
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#Import the classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)  

#pickle the model
import pickle
pickle.dump((classifier, scaler), open('rfendstate.p','wb'))

#Make the prediction
y_pred = classifier.predict(X_test)  

#Plot the results using t_SNE
from sklearn.manifold import TSNE

np.random.seed(42)
rndperm = np.random.permutation(plotdata.shape[0])

N = 20000

df_merge = pd.DataFrame(data=X_train, columns=['namsxkd','lhdn','tsld','ld11','nganh_kd'])

df_merge = df_merge[:N]
#df_plotset = plotdata.loc[rndperm[:N],:].copy()
#data_plotset = df_plotset.values
#print(data_plotset)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
tsne_results = tsne.fit_transform(df_merge.values)
df_merge["label"] = y_train[:N]
df_merge['tsne-2d-one'] = tsne_results[:,0]
df_merge['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.set_style("white")
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("dark", 2),
    data=df_merge,
    #legend='full',
    alpha=0.3
)
legend = plt.legend()
legend.get_texts()[1].set_text('Low Risk')
legend.get_texts()[2].set_text('High Risk')
sns.despine()
plt.savefig("RFPlot2.png")
#plt.show()

#Print the output if you want. Otherwise write it to file.
#print(y_pred)

feature_imp = pd.Series(classifier.feature_importances_,index=df[['namsxkd','lhdn','tsld','ld11','nganh_kd']].columns).sort_values(ascending=False)
print(feature_imp)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  