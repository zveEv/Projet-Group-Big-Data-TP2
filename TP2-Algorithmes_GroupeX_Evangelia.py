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
! pip install "dask[complete]" #installation complete du package dask

import sys # getion de matrices
import numpy as np # gestion des matrices
import pandas as pd # gestion et manipulation des dataframes
import os # reading the input files we have access to
import matplotlib.pyplot as plt #pour faire de visualisation de données
import seaborn as sns #realisation des pairplots
from sklearn import preprocessing #standarization des valeurs
from sklearn.cluster import KMeans #librairie du clustering non-bigdata
from sklearn import linear_model #estimation du modele
from sklearn.metrics import mean_squared_error #prediction du modele
from dask import dataframe as dd
from dask-ml.cluster import KMeans#!!incorrect
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

trainEch_df = pd.read_csv(dossier_trainEch)
trainEch_df.dtypes

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

train_df = dd.read_csv(dossier_train)
train_df.dtypes


#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 


### Q2.1 - Nettoyer et préparer les données


# Enlever les valeurs incorrectes ou manquantes (si pertinent)


# ---------- Utiliser une librairie usuelle

#nombre de valeurs manquants par variable
print(trainEch_df.isnull().sum())
trainEch_df_clean = trainEch_df.dropna(how = 'any', axis = 'rows')

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)!!!!!
print(train_df.isnull().sum())
train_df_clean = train_df.dropna(how = 'any', axis = 'rows')

#je ne peux pas trouver comment supprimer valeurs manquants avec dask.

# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie


# ---------- Utiliser une librairie usuelle
trainEch_df_clean=trainEch_df_clean.drop(['key','pickup_datetime','passenger_count'], axis=1)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
#le meme que librairie usuelle
train_df_clean=train_df_clean.drop(['key','pickup_datetime','passenger_count'], axis=1)


# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle

trainEch_df_clean.describe()


#1eme solution
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

for var in list(trainEch_df_clean.columns.values):
    trainEch_df_clean= test_trainEch_df_clean[~trainEch_df_clean[var].isin(list(outliers_iqr(trainEch_df_clean[var])))]


#2eme facon
    
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

for var in list(trainEch_df_clean.columns.values):
    trainEch_df_clean=remove_outlier(test_trainEch_df_clean,var)
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

train_df_clean.describe()

#1er solution
for var in list(train_df_clean.columns.values):
    train_df_clean= test_train_df_clean[~train_df_clean[var].isin(list(outliers_iqr(train_df_clean[var])))]

#2eme solution
    
for var in list(train_df_clean.columns.values):
    train_df_clean=remove_outlier(train_df_clean,var)    
    
# ---------- Utiliser une librairie usuelle

sns.pairplot(trainEch_df_clean)
variables=list(trainEch_df_clean.columns.values)
for var in variables:
        trainEch_df_clean[var].plot.hist()
        plt.title(var)
        plt.show()

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
sns.pairplot(train_df_clean)
for var in variables:
        train_df_clean[var].plot.hist()
        plt.title(var)
        plt.show()

# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")

inputvar=["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]

# ---------- Utiliser une librairie usuelle


X,y = trainEch_df_clean[inputvar],trainEch_df_clean["fare_amount"]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


X_big,y_big = train_df_clean[inputvar], train_df_clean["fare_amount"]


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



En regardant la reponse à la Question No5. Tous les clusters dependent de la localisation.
Les clusters obtenus comportent des caracteristiques homogenes basés sur la localisation.
Les 4 variables contribuent de la meme facon sur le clustering et le criteres de clustering sont les memes pour tout les groups.
Il serait interessant de calculer les distances en calculant la difference entre latitide dropoff et pickup. Pareil pour le longitude.



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle




data_cluster=trainEch_df_clean[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
data_cluster['cluster'] = labels
sample_index=np.random.randint(0, len(X_scaled), 1000)

sns.pairplot(data_cluster.loc[sample_index, :], hue= "cluster")
plt.show()







#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 

# ---------- Utiliser une librairie usuelle


CODE




### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       


REPONSE ECRITE (3 lignes maximum)




### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


CODE




### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


REPONSE ECRITE (3 lignes maximum)











#
# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Calculer le RMSE et le R² sur le jeu de test.



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Quelle est la qualité de la prédiction sur le jeu de test ?


REPONSE ECRITE (3 lignes maximum)








#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 



### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE




### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Quelle est la qualité de la prédiction sur le jeu de test ?


REPONSE ECRITE (3 lignes maximum)







#
# QUESTION 7 - RESEAU DE NEURONES (QUESTION BONUS)
# 



### Q7.1 - Mener une régression de la sortie "fare_amount" en fonction de l'entrée (mise à l'échelle), 
###       sur tout le jeu de données, avec un réseau à 2 couches cachées de 10 neurones chacune



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



### Q7.2 - Prédire le prix de la course en fonction de nouvelles entrées avec le réseau de neurones entraîné


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression avec réseau de neurones sur l'échantillon d'apprentissage et en testant plusieurs 
# nombre de couches et de neurones par couche sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Calculer le RMSE et le R² de la meilleure prédiction sur le jeu de test.


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Quelle est la qualité de la prédiction sur le jeu de test ? Comment se compare-t-elle à la régression linéaire?


REPONSE ECRITE (3 lignes maximum)


