from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans


main = tkinter.Tk()
main.title("An Unsupervised Ensemble Clustering Approach for the Analysis of Student Behavioral Patterns") 
main.geometry("1300x1200")

global dataset_file
global X, Y
global train
dbscan_main = []
dbscan_noise = []
kmeans_main = []

def loadData():
    global dataset_file
    global train
    dataset_file = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=dataset_file)
    text.delete('1.0', END)
    text.insert(END,dataset_file+" student behaviour dataset loaded\n\n")
    train = pd.read_csv(dataset_file)
    text.insert(END,str(train.head()))
    
def featuresExtraction():
    global train
    global X, Y
    text.delete('1.0', END)
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
    text.insert(END,str(train.head()))

    X = train.values[:, 0:17] 
    Y = train.values[:, 16]
    X = normalize(X)
    pca = PCA(n_components = 6)
    X = pca.fit_transform(X)
    n_pcs= pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = train.columns
    text.insert(END,"\n\nAll Features Available in Dataset : "+str(initial_feature_names)+"\n\n")
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    text.insert(END,"Selected Important Features using PCA : "+str(most_important_names)+"\n\n")

    df_corr = train.corr()
    df_corr[['Semester']].plot(kind='bar')
    plt.show()

def runDBSCAN():
    global X, Y
    global dbscan_main
    global dbscan_noise
    dbscan_main.clear()
    dbscan_noise.clear()
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    index = 0
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        if index == 0:
            for i in range(len(X)):
                if class_member_mask[i] == True:
                    dbscan_main.append(X[i])
                if class_member_mask[i] == False:
                    dbscan_noise.append(i)    
        index = 1            
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
    dbscan_main = np.asarray(dbscan_main)
    dbscan_noise = np.asarray(dbscan_noise)
    text.delete('1.0', END)
    text.insert(END,"Total records found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total Mainstream student found by DBSCAN : "+str(len(dbscan_main))+"\n")
    text.insert(END,"Total Anomalous student found by DBSCAN  : "+str(len(dbscan_noise))+"\n\n")
    text.insert(END,"Below are the Anomalous Students ID : \n\n")
    text.insert(END,str(dbscan_noise))    
    plt.title('Estimated number of clusters: %d'+str(len(unique_labels)))
    plt.show()
    

def runKMEANS():
    text.delete('1.0', END)
    global X, Y
    global train
    global kmeans_main
    kmeans_main.clear()
    global dbscan_main
    dbscan_main = np.asarray(dbscan_main)
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(dbscan_main)
    labels = kmeans.labels_
    clusters = unique(labels)
    for cluster in clusters:
        class_member_mask = (labels == cluster)
        kmeans_main.append(class_member_mask.nonzero()[0])
        row_ix = where(labels == cluster)
        plt.scatter(dbscan_main[row_ix, 0], dbscan_main[row_ix, 1])
    plt.show()
    
    
def visualizeClusters():
    global kmeans_main
    print(kmeans_main)
    height = []
    bars = []
    for i in range(len(kmeans_main)):
        height.append(len(kmeans_main[i]))
        bars.append("Cluster "+str(i+1))
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()    
    
def anomalousStudent():
    global kmeans_main
    text.delete('1.0', END)
    text.insert(END,"Students found Anomalous after applying KMEANS\n\n")
    for i in range(len(kmeans_main)):
        if len(kmeans_main[i]) < 30:
            text.insert(END,"Anomalous Students ID : "+str(kmeans_main[i])+"\n") 
    
font = ('times', 16, 'bold')
title = Label(main, text='An Unsupervised Ensemble Clustering Approach for the Analysis of Student Behavioral Patterns')
title.config(bg='skyblue', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Data Collection/Upload Dataset", command=loadData)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='skyblue', fg='black')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

fsButton = Button(main, text="Features Extraction", command=featuresExtraction)
fsButton.place(x=50,y=150)
fsButton.config(font=font1) 

dbscanButton = Button(main, text="Run Initial DBSCAN Clustering", command=runDBSCAN)
dbscanButton.place(x=330,y=150)
dbscanButton.config(font=font1) 

kmeansButton = Button(main, text="Run KMeans Clustering", command=runKMEANS)
kmeansButton.place(x=630,y=150)
kmeansButton.config(font=font1)

visualButton = Button(main, text="Visualize Clusters", command=visualizeClusters)
visualButton.place(x=50,y=200)
visualButton.config(font=font1) 

stdButton = Button(main, text="Display Anomalous Student ID's", command=anomalousStudent)
stdButton.place(x=330,y=200)
stdButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='skyblue')
main.mainloop()
