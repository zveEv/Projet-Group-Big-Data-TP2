# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:38:53 2019

@author: 
"""



##############################################################################
#
#    ALGORITHMIE DU BIG DATA
#
##############################################################################


#
# QUESTION 0 - IMPORTATION DES PACKAGES ET LIBRAIRIES UTILISEES PAR LA SUITE
# 
! pip install "dask[complete]" 

import sys
import numpy as np # gestion des matrices
import pandas as pd # gestion et manipulation des dataframes
import os # reading the input files we have access to
import matplotlib.pyplot as plt #pour faire de visualisation de données
import seaborn as sns #realisation des pairplots
from sklearn import preprocessing #standarization des valeurs
from sklearn.cluster import KMeans 
from sklearn import linear_model #estimation du modele
from sklearn.metrics import mean_squared_error #prediction du modele
from dask import dataframe as dd
from dask-ml.cluster import KMeans
#Pour le grands volums des données "Big Data"
import dask 
from dask_glm.estimators import LinearRegression

import dask.dataframes as dd #pour importer un fichier big data, ca veut dire avec taille en Gigabytes.

#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 


### Q1.1 - Indiquer le dossier et le fichier cible


dossier_train='C:/Users/lilian/Documents/M2 FOAD/Big Data/Projet Groupe/train/train.csv'
dossier_trainEch='C:/Users/lilian/Documents/M2 FOAD/Big Data/Projet Groupe/train_echantillon.csv'




### Q1.2 - Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)

trainEch_df =  pd.read_csv(dossier_trainEch)
trainEch_df.dtypes

test_trainEch_df =  pd.read_csv(dossier_trainEch, nrows=10000)
test_trainEch_df.dtypes
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

train_df = dd.read_csv(dossier_train)
train_df.dtypes


test_train_df =  dd.read_csv(dossier_train).head(n=20000)
test_train_df.dtypes








#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 


### Q2.1 - Nettoyer et préparer les données


# Enlever les valeurs incorrectes ou manquantes (si pertinent)


# ---------- Utiliser une librairie usuelle

#nombre de valeurs manquants par variable
print(trainEch_df.isnull().sum())
trainEch_df_clean = trainEch_df.dropna(how = 'any', axis = 'rows')

print(test_trainEch_df.isnull().sum())
test_trainEch_df_clean = test_trainEch_df.dropna(how = 'any', axis = 'rows')
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)!!!!!
print(test_train_df.isnull().sum())
test_train_df_clean = test_train_df.dropna(how = 'any', axis = 'rows')

#je ne peux pas trouver comment supprimer valeurs manquants avec dask.

# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie


# ---------- Utiliser une librairie usuelle
test_trainEch_df_clean=test_trainEch_df_clean.drop(['key','pickup_datetime','passenger_count'], axis=1)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
#le meme que librairie usuelle
test_train_df_clean=test_train_df_clean.drop(['key','pickup_datetime','passenger_count'], axis=1)


# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle

test_trainEch_df_clean.describe()


#1eme solution
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

for var in list(test_trainEch_df_clean.columns.values):
    test_trainEch_df_clean= test_trainEch_df_clean[~test_trainEch_df_clean[var].isin(list(outliers_iqr(test_trainEch_df_clean[var])))]


#2eme facon
    
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

for var in list(test_trainEch_df_clean.columns.values):
    test_trainEch_df_clean=remove_outlier(test_trainEch_df_clean,var)
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

test_train_df_clean.describe()

#1er solution
for var in list(test_train_df_clean.columns.values):
    test_train_df_clean= test_train_df_clean[~test_train_df_clean[var].isin(list(outliers_iqr(test_train_df_clean[var])))]

#2eme solution
    
for var in list(test_train_df_clean.columns.values):
    test_train_df_clean=remove_outlier(test_train_df_clean,var)    
    
# ---------- Utiliser une librairie usuelle

sns.pairplot(test_trainEch_df_clean)
variables=list(test_trainEch_df_clean.columns.values)
for var in variables:
        test_trainEch_df_clean[var].plot.hist()
        plt.title(var)
        plt.show()

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
sns.pairplot(test_train_df_clean)
for var in variables:
        test_train_df_clean[var].plot.hist()
        plt.title(var)
        plt.show()

# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")

inputvar=["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]

# ---------- Utiliser une librairie usuelle


X,y = test_trainEch_df_clean[inputvar],test_trainEch_df_clean["fare_amount"]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


X_big,y_big = test_train_df_clean[inputvar],test_train_df_clean["fare_amount"]


# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle

X_scaled=preprocessing.scale(X)
y_scaled=preprocessing.scale(y)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


X_big_scaled=preprocessing.scale(X_big)
y_big_scaled=preprocessing.scale(y_big)







#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

Kmeans_model=KMeans(n_clusters=4, random_state=1).fit(X_scaled)
labels=Kmeans_model.labels_
inertia=Kmeans_model.inertia_
print("Nombre des clusters : " + str(4) + "- inertie : " + str(inertia))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

kmeans_dask = KMeans(n_clusters= 4)
kmeans_dask.fit_transform(X_big_scaled)
cluster=kmeans_dask.labels



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle

for k in range (1,10):
  Kmeans_model=KMeans(n_clusters=k, random_state=1).fit(X_scaled)
  labels=Kmeans_model.labels_
  inertia=Kmeans_model.inertia_
  print("Nombre des clusters : " + str(k) + "- inertie : " + str(inertia))

Nombre des clusters : 1- inertie : 33384.0
Nombre des clusters : 2- inertie : 19497.435282192197
Nombre des clusters : 3- inertie : 15990.850390080006
Nombre des clusters : 4- inertie : 13494.044120871778
Nombre des clusters : 5- inertie : 12177.51614215438
Nombre des clusters : 6- inertie : 11152.728887386244
Nombre des clusters : 7- inertie : 10244.800056180686
Nombre des clusters : 8- inertie : 9558.27609763317
Nombre des clusters : 9- inertie : 8996.701282078819





### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



Selon la sortie que l'on obtien avec la reponse à la question Q 3.1, 
le nombre optimal des clusters est 4, puisque à partir de ce nombre l'inertie(variance intracluster) decroit de moins en moins.


### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 



En regardant la reponse à la 
Les clusters obtenus comportent des caracteristiques homogenes basés sur la localisation.
Les 4 variables contribuent de la meme facon.
Il serait interessant de calculer les distances en calculant la difference entre latitide dropoff et pickup. Pareil pour le longitude.



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle




data_cluster=test_trainEch_df_clean[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
data_cluster['cluster'] = labels
sample_index=np.random.randint(0, len(X_scaled), 1000)

sns.pairplot(data_cluster.loc[sample_index, :], hue= "cluster")
plt.show()






