import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans

train = pd.read_csv('Dataset/xAPI-Edu-Data.csv')
le = LabelEncoder()
train['gender'] = pd.Series(le.fit_transform(train['gender']))
train['NationalITy'] = pd.Series(le.fit_transform(train['NationalITy']))
train['PlaceofBirth'] = pd.Series(le.fit_transform(train['PlaceofBirth']))
train['StageID'] = pd.Series(le.fit_transform(train['StageID']))
train['GradeID'] = pd.Series(le.fit_transform(train['GradeID']))
train['SectionID'] = pd.Series(le.fit_transform(train['SectionID']))
train['Topic'] = pd.Series(le.fit_transform(train['Topic']))
train['Semester'] = pd.Series(le.fit_transform(train['Semester']))
train['Relation'] = pd.Series(le.fit_transform(train['Relation']))
train['ParentAnsweringSurvey'] = pd.Series(le.fit_transform(train['ParentAnsweringSurvey']))
train['ParentschoolSatisfaction'] = pd.Series(le.fit_transform(train['ParentschoolSatisfaction']))
train['StudentAbsenceDays'] = pd.Series(le.fit_transform(train['StudentAbsenceDays']))
train['Class'] = pd.Series(le.fit_transform(train['Class']))

X = train.values[:, 0:17] 
Y = train.values[:, 16]
X = normalize(X)
pca = PCA(n_components = 6)
X = pca.fit_transform(X)

n_pcs= pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = train.columns
print(initial_feature_names)
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print(most_important_names)
'''
selected = pd.DataFrame(pca.components_,columns=train.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6'])
print(selected)
print(X)

df_corr = train.corr()
df_corr[['Semester']].plot(kind='bar')
plt.show()
'''
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

print(labels)
clusters = unique(labels)
for cluster in clusters:
    row_ix = where(labels == cluster)
    temp = np.asarray(row_ix)
    
    plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.show()
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
subclusters = []
noise_cluster = []
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
index = 0
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    if index == 0:
        for i in range(len(X)):
            
            if class_member_mask[i] == True:
                subclusters.append(X[i])
            if class_member_mask[i] == False:
                noise_cluster.append(X[i])    
    index = 1            
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
subclusters = np.asarray(subclusters)
print(subclusters.shape)


kmeans_result = []
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(subclusters)
labels = kmeans.labels_
print(labels)
clusters = unique(labels)
for cluster in clusters:
    class_member_mask = (labels == cluster)
    kmeans_result.append(class_member_mask.nonzero()[0])
    row_ix = where(labels == cluster)
    temp = np.asarray(row_ix)
    
    plt.scatter(subclusters[row_ix, 0], subclusters[row_ix, 1])
# show the plot
plt.show()


