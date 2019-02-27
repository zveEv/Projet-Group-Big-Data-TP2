# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:38:53 2019

@author: Freddy Donald FOGUE NOUMSSI
"""

##############################################################################
#
#    ALGORITHMIE DU BIG DATA
#
##############################################################################


#
# QUESTION 0 - IMPORTATION DES PACKAGES ET LIBRAIRIES UTILISEES PAR LA SUITE
# 
#Installation des packages potentiellement inexistants 

! pip install "dask[complete]" #Pour la gestion des grands volumes de données 
! pip install  sys
! pip install  numpy #gestion des matrices
! pip install  pandas #gestion et manipulation des dataframes
! pip install  os #reading the input files we have access to
! pip install  matplotlib.pyplot #pour faire de visualisation de données
! pip install  seaborn #realisation des pairplots
! pip install  sklearn #Pour la Préparation et la modélisation
! pip install  sklearn.metrics #Pour le diagnostique du modèle
! pip install  "dask[dataframe]" #Pour la gestion des grandes bases de données

#Importation des packages installés 

import sys
import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
import sklearn.metrics


#Utilisation du package dask pour les grandes bases de données, en particulier pour notre base
import dask
import dask.dataframe as dd  

#Importation des librairies 

from sklearn import preprocessing #Pour la standardisation des variables
from sklearn import linear_model #estimation du modèle linéaire
from sklearn.metrics import mean_squared_error #Calcul de l'érreur quadratique moyenne de prédiction
from sklearncluster import kMeans #Réaliser un cluster avec sklearn

#Librairies du package dask
from dask_glm.estimators import LinearRegression #Regression linéaire sur un big data
from dask_ml.clusters import kMeans #réaliser un kmeans sur les big data

#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 
### Q1.1 - Indiquer le dossier et le fichier cible

fich_complet= "E:/Mon Master 2/Année 2018-2019/Trimestre 2/Big Data/TP2/train.csv"
fich_echant= "E:/Mon Master 2/Année 2018-2019/Trimestre 2/Big Data/TP2/train_echantillon.csv"

### Q1.2 - Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)

df_ech=pd.read_csv(fich_echant)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

df_fich_compl=dd.read_csv(fich_complet)

#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 
### Q2.1 - Nettoyer et préparer les données
# Enlever les valeurs incorrectes ou manquantes (si pertinent) 

# ---------- Utiliser une librairie usuelle

df_ech=df_ech.dropna(how='all')  #Suppression des lignes qui dont toutes les valeurs sont manquantes

total = print(df_ech.isnull().sum())#Pour chaque colonnes de la table, calcul du nombre de valeurs manquantes et les classes par ordre décroissant
pourcentage = (df_ech.isnull().sum()/df_ech.isnull().count()) #Pour chaque colonne calcul du pourcentage et les classes par ordre décroissant
Valeur_manq = pd.concat([total, pourcentage], axis=1, keys=['Total', 'Pourcentage']) #Concatene les deux précédentes tables

var_val_manq=Valeur_manq[Valeur_manq.pourcentage >= 0.4].index #Les variables ayant plus de 40 pourcent de valeurs manquantes

df_ech=df_ech.drop(var_val_manq, axis=1) #Suppression de ces variables
             
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

df_ech=df_ech.dropna(how='all')  #Suppression des lignes qui dont toutes les valeurs sont manquantes

total = print(df_ech.isnull().sum()) #Pour chaque colonnes de la table, calcul du nombre de valeurs manquantes et les classes par ordre décroissant
pourcentage = (df_ech.isnull().sum()/df_ech.isnull().count()) #Pour chaque colonne calcul du pourcentage et les classes par ordre décroissant
Valeur_manq = dd.concat([total, pourcentage], axis=1, keys=['Total','Pourcentage']

var_val_manq=Valeur_manq[Valeur_manq.pourcentage >= 0.4].index #Les variables ayant plus de 40 pourcent de valeurs manquantes

df_ech=df_ech.drop(var_val_manq, axis=1) #Suppression de ces variables

# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie

# ---------- Utiliser une librairie usuelle

df_ech=df_ech["fare_amount","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_atitude"]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

df_ech=df_ech["fare_amount","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_atitude"])

# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes

# ---------- Utiliser une librairie usuelle

df_ech.describe()

#Calcul des valeurs aberrantes et suppression


def extract_ab(): #Fonction qui supprime les valeurs aberrantes
    p_quart=np.percentile(df_ech,25) #Calcul du premier quartile
    t_quart=np.percentile(df_ech,75) #Calcul du troisième quartile
    for i in 0:len(df_ech.columns):
        df_ech = df_ech[~(df_ech[:,df_ech.columns[i]] > t_quart[t_quart.columns[i]])] #Supprime toutes les lignes dont les valeurs sont supérieur au troisième quartile
        df_ech = df_ech[~(df_ech[:,df_ech.columns[i]] < p_quart[p_quart.columns[i]])] #Supprime toutes les lignes dont les valeurs sont inférieures au premier quartile
     return(df_ech)

df_ech=extract_ab()
     
    
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

df_ech.describe()


def extract_ab(): #Fonction qui supprime les valeurs aberrantes
    p_quart=np.percentile(df_ech,25) #Calcul du premier quartile
    t_quart=np.percentile(df_ech,75) #Calcul du troisième quartile
    for i in 0:len(df_ech.columns):
        df_ech = df_ech[~(df_ech[:,df_ech.columns[i]] > t_quart[t_quart.columns[i]])] #Supprime toutes les lignes dont les valeurs sont supérieur au troisième quartile
        df_ech = df_ech[~(df_ech[:,df_ech.columns[i]] < p_quart[p_quart.columns[i]])] #Supprime toutes les lignes dont les valeurs sont inférieures au premier quartile
     return(df_ech)

df_ech=extract_ab()

# Visualiser les distributions des variables d'entrée et de sortie (histogramme, pairplot)

# ---------- Utiliser une librairie usuelle

def histogram_trac():
    for col in df_ech.columns:
        df_ech.hist(col)
              
# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")

# ---------- Utiliser une librairie usuelle

var_entree=df_ech["fare_amount"]
var_sortie=df_ech.drop(["fare_amount"], axis=1)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

var_entree=df_ech["fare_amount"]
var_sortie=df_ech.drop(["fare_amount"], axis=1)

# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)
# ---------- Utiliser une librairie usuelle

var_entree_stand=preprocessing.scale(var_entree)
var_sortie_stand=preprocessing.scale(var_sortie)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


var_entree_stand=preprocessing.scale(var_entree)
var_sortie_stand=preprocessing.scale(var_sortie)


#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 

### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

kmeans_model=kMeans(n_clusters=3, random_state=1).fit(var_entree_stand)
labels=kmeans_model.labels_
inertia=kmeans_model.inertia

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

kmeans_dask=kMeans(n_clusters=4)
kmeans_dask.fit_transform(var_entree_stand)
cluster=kmeans_dask.labels

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


