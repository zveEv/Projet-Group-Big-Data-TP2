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
from sklearn import linear_model #estimation du modele
from sklearn.metrics import mean_squared_error #prediction du modele
from dask import dataframe as dd
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
test_train_clean = test_train_df.dropna(how = 'any', axis = 'rows')

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

#le z-value est -20 pour les variables de longitude et 20 pour les variables de latitude
# ---------- Utiliser une librairie usuelle

test_trainEch_df_clean.describe()

#1er solution
test_trainEch_df_clean = test_trainEch_df_clean[(test_trainEch_df_clean.dropoff_longitude < -20) & (test_trainEch_df_clean.pickup_longitude < -20) 
& (test_trainEch_df_clean.dropoff_latitude > 20) & (test_trainEch_df_clean.pickup_latitude > 20)]

#2eme solution
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

for var in list(test_trainEch_df_clean.columns.values):
    test_trainEch_df_clean= test_trainEch_df_clean.loc[~test_trainEch_df_clean[var].isin(list(outliers_iqr(test_trainEch_df_clean[var])))]
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

test_train_df_clean.describe()

#2eme solution
for var in list(test_train_df_clean.columns.values):
    test_train_df_clean= test_train_df_clean.loc[~test_train_df_clean[var].isin(list(outliers_iqr(test_train_df_clean[var])))]


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

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle

CODE





### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



REPONSE ECRITE (3 lignes maximum)





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle


CODE









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


