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
! pip install dask-ml
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
import dask.array as da 

#Importation des librairies 

from sklearn import preprocessing #Pour la standardisation des variables
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model #estimation du modèle linéaire
from sklearn.metrics import mean_squared_error, r2_score #Calcul de l'érreur quadratique moyenne de prédiction
from sklearn.cluster import KMeans #Réaliser un cluster avec sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

#Librairies du package dask
from dask_glm.estimators import LinearRegression #Regression linéaire sur un big data
from dask_ml.clusters import kMeans #réaliser un kmeans sur les big data
from dask_ml.decomposition import PCA
from dask_ml.linear_model import LogisticRegression

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



En regardant la reponse à la Question No5. Tous les clusters dependent de la localisation.
Les clusters obtenus comportent des caracteristiques homogenes basés sur la localisation.
Les 4 variables contribuent de la meme facon sur le clustering et le criteres de clustering sont les memes pour tout les groups.
Il serait interessant de calculer les distances en calculant la difference entre latitide dropoff et pickup. Pareil pour le longitude.



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle




data_cluster=test_trainEch_df_clean[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
data_cluster['cluster'] = labels
sample_index=np.random.randint(0, len(X_scaled), 1000)

sns.pairplot(data_cluster.loc[sample_index, :], hue= "cluster")
plt.show()







#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

pca=PCA(n_components=4)
pca_resultat=pca.fit_transform(X_scaled)



# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

dX=da.from_array(X_scaled, chunks=X_scaled.shape)
pca=PCA(n_components=4)
pca.fit(dX)


print(pca.explained_variance_ratio_)
print(pca.singular_values_)


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 

# ---------- Utiliser une librairie usuelle

plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       

On doit garder 2. Car a partir de 2eme composante la variance diminue avec un rythme plus lent.
autrement dit avec les deux premiers compostants on peut expliquer 80% 



### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle

X_scaled=StandardScaler().fit_transform(X)


x_vector=pca.components_[0]
y_vector=pca.components_[1]

xs=pca.transform(X_scaled)[:,0]
ys=pca.transform(X_scaled)[:,1]

points_plot_index=np.random.randint(0, len(xs), 1000)

for i in range(len(x_vector)):
    plt.arrow(0, 0, x_vector[i]*max(xs), y_vector[i]*max(ys), color='r', width=0.0005, head_width=0.0025)
    plt.text(x_vector[i]*max(xs)*1.2, y_vector[i]*max(ys)*1.2, list(test_train_df_clean[inputvar].columns.values)[i], color='r')
for i in points_plot_index:
    plt.plot(xs[i],ys[i],'bo')
    plt.text(xs[i]*1.2, ys[i]*1.2, list(test_train_df_clean[inputvar].index)[i], color='b')
    

plt.show()

### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


les variables dropoff_longitude et dropoff_latitude sont liés entre elles et tres liés par rapport à la premiere 
composante principale.

les variables pickup longitude et pickup latitude sont en relation negative avec la composante numero deux. Alignés sur la composante numero 2.

#####INUTILE
y_binaire=np.zeros(len(y))
y_binaire[y > y.median()]=1

idx_train=np.random.rand(len(y_binaire)) < 0.8
Xtrain=X[idx_train]
Xtrain_scaler=StandardScaler().fit(Xtrain)
Xtrain=Xtrain_scaler.transform(X[~idx_train])

y_binaire=np.zeros(len(y))


y_binaire=np.zeros(len(y))
y_binaire[np.array(y > y.median())]=1
ytrain, ytest= y_binaire[idx_train], y_binaire[~idx_train]

log_reg_preacp=LogisticRegression()#modelisation

log_reg_preacp.fit(Xtrain, ytrain)




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

y.plot.hist()
y.mean
y.median

### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil

X,y = test_trainEch_df_clean[inputvar],test_trainEch_df_clean["fare_amount"]
X_scaled=StandardScaler().fit_transform(X)



# ---------- Utiliser une librairie usuelle


y_binaire=np.zeros(len(y))
y_binaire[y > y.median()]=1



# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

y_binaire=np.zeros(len(y))
y_binaire[np.array(y > y.median())]=1


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle


log_reg=LogisticRegression()#modelisation
log_reg.fit(X_scaled, y)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
#https://dask-ml.readthedocs.io/en/latest/
log_reg=LogisticRegression()
log_reg.fit(X_scaled, y)



### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



...


### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


#Xtrain, Xtest= X_scaled[idx_train], X_scaled[~idx_train]
#ytrain, ytest= y_binaire[idx_train], y_binaire[~idx_train]



idx_train=np.random.rand(len(y_binaire)) < 0.6
Xtrain=X[idx_train]
Xtrain_scaler=StandardScaler().fit(Xtrain)
Xtrain=Xtrain_scaler.transform(X[~idx_train])

Xtrain, Xvalidtest= X_scaled[idx_train], X_scaled[~idx_train]
ytrain, yvalidtest= y_binaire[idx_train], y_binaire[~idx_train]

idx_valid=np.random.rand(len(y_binaire)) < 0.2
XValid=X[idx_valid]
XValid_scaler=StandardScaler().fit(XValid)
XValid=XValid_scaler.transform(X[~idx_valid])

XValid, Xtest=X_scaled[idx_valid], X_scaled[~idx_valid]
yvalid, ytest= y_binaire[idx_valid], y_binaire[~idx_valid]

# ---------- Utiliser une librairie usuelle

regr=linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

lr=LinearRegression()
lr.fit(Xtrain, ytrain)


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle


log_reg=LogisticRegression()#modelisation
log_reg.fit(Xtrain, ytrain)

prediction_logreg=log_reg.predict(XValid)
pred_proba_logreg=log_reg.predict_proba(XValid)
...

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

log_reg=LogisticRegression()#modelisation
log_reg.fit(Xtrain, ytrain)

prediction_biglogreg=log_reg.predict(XValid)
log_reg.score(XValid, yvalid)#Returns the mean accuracy on the given test data and labels.
....
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


