# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:06:35 2019

@author: lilian
"""

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
import dask.array as da 

#Importation des librairies 

from sklearn import preprocessing #Pour la standardisation des variables
from sklearn import linear_model #estimation du modèle linéaire
from sklearn.metrics import mean_squared_error, r2_score #Calcul de l'érreur quadratique moyenne de prédiction
from sklearn.cluster import KMeans #Réaliser un cluster avec sklearn
from sklearn.decomposition import PCA

#Librairies du package dask
from dask_glm.estimators import LinearRegression #Regression linéaire sur un big data
from dask_ml.clusters import kMeans #réaliser un kmeans sur les big data
from dask_ml import preprocessing
from dask_ml import dask_ml.linear_model.LogisticRegression

#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 
### Q1.1 - Indiquer le dossier et le fichier cible

fich_complet= "C:/Users/lilian/Documents/M2 FOAD/Big Data/Projet Groupe/train/train.csv"
fich_echant= "C:/Users/lilian/Documents/M2 FOAD/Big Data/Projet Groupe/train_echantillon.csv"

### Q1.2 - Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)

df_ech=pd.read_csv(fich_echant, nrows=10000)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

df_fich_compl=dd.read_csv(fich_complet).head(n=20000)

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

var_val_manq=Valeur_manq[Valeur_manq.Pourcentage >= 0.4].index #Les variables ayant plus de 40 pourcent de valeurs manquantes

df_ech=df_ech.drop(var_val_manq, axis=1) #Suppression de ces variables
             
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

df_ech=df_ech.dropna(how='all')  #Suppression des lignes qui dont toutes les valeurs sont manquantes

total = print(df_ech.isnull().sum()) #Pour chaque colonnes de la table, calcul du nombre de valeurs manquantes et les classes par ordre décroissant
pourcentage = (df_ech.isnull().sum()/df_ech.isnull().count()) #Pour chaque colonne calcul du pourcentage et les classes par ordre décroissant
Valeur_manq = dd.concat([total, pourcentage], axis=1, keys=['Total','Pourcentage'])

var_val_manq=Valeur_manq[Valeur_manq.pourcentage >= 0.4].index #Les variables ayant plus de 40 pourcent de valeurs manquantes

df_ech=df_ech.drop(var_val_manq, axis=1) #Suppression de ces variables


# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie

# ---------- Utiliser une librairie usuelle

df_ech=df_ech.drop(['key','pickup_datetime','passenger_count'], axis=1)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

df_ech=df_ech.drop(['key','pickup_datetime','passenger_count'], axis=1)

# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes

# ---------- Utiliser une librairie usuelle

df_ech.describe()

#Calcul des valeurs aberrantes et suppression


ef outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

for var in list(df_ech.columns.values):
    df_ech= df_ech[~df_ech[var].isin(list(outliers_iqr(df_ech[var])))]

    
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

df_ech.describe()

#Calcul des valeurs aberrantes et suppression

for var in list(df_ech.columns.values):
    df_ech= df_ech[~df_ech[var].isin(list(outliers_iqr(df_ech[var])))]


# Visualiser les distributions des variables d'entrée et de sortie (histogramme, pairplot)

# ---------- Utiliser une librairie usuelle

sns.pairplot(df_ech)
variables=list(df_ech.columns.values)
for var in variables:
        df_ech[var].plot.hist()
        plt.title(var)
        plt.show()
              
# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")
        
inputvar=["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]

# ---------- Utiliser une librairie usuelle

X,Y = df_ech[inputvar],df_ech["fare_amount"]


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

X,Y = df_ech[inputvar],df_ech["fare_amount"]


# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)
# ---------- Utiliser une librairie usuelle


var_entree_stand=preprocessing.scale(var_entree)
var_sortie_stand=preprocessing.scale(var_sortie)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)



var_entree_stand=dask_ml.preprocessing.scale(X)
var_sortie_stand=dask_ml.preprocessing.scale(Y)


#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 

### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

for k in range (1, 10): 
    kmeans_model=KMeans(n_clusters=k, random_state=1).fit(var_entree_stand)
    labels=kmeans_model.labels_
    inertia=kmeans_model.inertia_
    print("Nombre des clusters : " + str(k) + "- inertie : " + str(inertia))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

for k in range (1, 10):
    kmeans_dask = KMeans(n_clusters= k)
    kmeans_dask.fit_transform(var_entree_stand)
    cluster=kmeans_dask.labels
    inertia=kmeans_dask.inertia_
    print("Nombre des clusters : " + str(k) + "- inertie : " + str(inertia))
    

### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters

# ---------- Utiliser une librairie usuelle

def trace_inertie (n): 
    t=[]
    num=[]
    for k in range (1,20):
        kmeans_model=KMeans(n_clusters=k, random_state=1).fit(var_entree_stand) #Réalise un k-means
        num.append(k) #Création de la liste des nombres 
        t.append(kmeans_model.inertia_) #Création de la liste d'inertie
    inertie=np.array(t) #vecteur d'inertie
    num=np.array(num) #Vecteur de nombres
    plt.plot(num, inertie) #Tracé de la courbe
    plt.show() #Présentation de la courbe
    
 

### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



REPONSE ECRITE (3 lignes maximum)


Selon la sortie que l''on obtient avec la reponse à la question Q 3.1, 
le nombre optimal des clusters est 4, puisque à partir de ce nombre l''inertie(variance intracluster) decroit de moins en moins.


### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 

En regardant la reponse à la Question No5. Tous les clusters dependent de la localisation.
Les clusters obtenus comportent des caracteristiques homogenes basés sur la localisation.
Les 4 variables contribuent de la meme facon sur le clustering et le criteres de clustering sont les memes pour tout les groups.
Il serait interessant de calculer les distances en calculant la difference entre latitide dropoff et pickup. Pareil pour le longitude.



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle


data_cluster=test_trainEch_df_clean[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']] #Sélection de la table avec les variables de localisation
data_cluster['cluster'] = labels
sample_index=np.random.randint(0, len(X_scaled), 1000) #Création d'un jeu de données de taille 1000 et de taille X_scaled

sns.pairplot(data_cluster.loc[sample_index, :], hue= "cluster") #Tracé de la courbe
plt.show() #Présentation de la courbe









#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

pca=PCA(n_components=6) #On va se limiter aux 6 premières composantes principales
pca_result=pca.fit_transform(var_entree_stand)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

dX=da.from_array(var_entree_stand, chunks=var_entree_stand.shape)
pca=PCA(n_components=6)
pca_result=pca.fit(dX)


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 
# ---------- Utiliser une librairie usuelle


diag_var=pca.explained_variance_ratio_
plt.bar(range(6), diag_var, width=0.8, color='blue') #Tracé du graphique à barre
plt.show()


### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       

On doit garder 2. Car a partir de 2eme composante la variance diminue avec un rythme plus lent.
autrement dit avec les deux premiers compostants on peut expliquer 80% 



### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle





x_vector=pca.components_[0]
y_vector=pca.components_[1]

xs=pca.transform(var_entree_stand)[:,0]
ys=pca.transform(var_entree_stand)[:,1]

points_plot_index=np.random.randint(0, len(xs), 1000)

for i in range(len(x_vector)):
    plt.arrow(0, 0, x_vector[i]*max(xs), y_vector[i]*max(ys), color='r', width=0.0005, head_width=0.0025)
    plt.text(x_vector[i]*max(xs)*1.2, y_vector[i]*max(ys)*1.2, list(df_ech[inputvar].columns.values)[i], color='r')
for i in points_plot_index:
    plt.plot(xs[i],ys[i],'bo')
    plt.text(xs[i]*1.2, ys[i]*1.2, list(df_ech[inputvar].index)[i], color='b')
    

plt.show()



### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


les variables dropoff_longitude et dropoff_latitude sont liés entre elles et tres liés par rapport à la premiere 
composante principale.

les variables pickup longitude et pickup latitude sont en relation negative avec la composante numero deux. Alignés sur la composante numero 2.





# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle

regr=linear_model.LinearRegression()
regr.fit(var_entree_stand, var_sortie_stand)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

lr=linearRegression()
lr.fit(var_entree_stand, var_sortie_stand)


### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

import random  #Librairie pour génerer un échantillon aléatoire

def generate_dataset(n, t):  #Fonction qui génère un vecteur aléatoire de taille n avec une probabilité de sélection de t.
    return [1 if random.random() < t else 0 for i in range(0,n)]

def custom_split_train_test(ens, p):  #Fonction qui repartie la population en deux.
    choice = generate_dataset(len(ens), p)
    train = [x for x, c in zip(ens, choice) if c == 1] #Sélection d'un échantillon de taille représentant p pourcent de la population totale
    autre = [x for x, c in zip(ens, choice) if c == 0] #Sélection d'un échantillon de taille représentant 1-p pourcent de la population totale
    return train, autre #Restition des deux tables

train, autre = custom_split_train_test(ens, 0.6)  #Obtient la base d'apprentissage et une autre base
test, validation= custom_split_train_test(autre, 0.5)  #Obtient les base test et de validation


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

import dask.array

from dask.array import random  #Librairie pour génerer un nombre aléatoire

def generate_dataset(n, t):  #Fonction qui génère un vecteur aléatoire de taille n avec une probabilité de sélection de t.
    return [1 if dask.array.random() < t else 0 for i in range(0,n)]

def custom_split_train_test(ens, p):  #Fonction qui repartie la population en deux.
    choice = generate_dataset(len(ens), p)
    train = [x for x, c in zip(ens, choice) if c == 1] #Sélection d'un échantillon de taille représentant p pourcent de la population totale
    autre = [x for x, c in zip(ens, choice) if c == 0] #Sélection d'un échantillon de taille représentant 1-p pourcent de la population totale
    return train, autre #Restition des deux tables

train, autre = custom_split_train_test(ens, 0.6)  #Obtient la base d'apprentissage et une autre base
test, validation= custom_split_train_test(autre, 0.5)


# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

X_train_scaled=preprocessing.scale(train[inputvar]) #Standardisation des données d'entree de la base d'apprentissage
Y_train_scaled=preprocessing.scale(train['fare_amount'])  #Standardisation des données de sortie de la base d'apprentissage


regr.fit(X_train_scaled, Y_train_scaled)


X_valid_scaled=preprocessing.scale(validation[inputvar]) #Standardisation des données d'entree de la base de validation
Y_valid_scaled=preprocessing.scale(validation['fare_amount'])  #Standardisation des données de sortie de la base de validation


from sklear.metrics import confusion_matrix
 
prediction_valid=reg.predict(X_valid_scaled)
confusion_mat=confusion_matrix(Y_valid_scaled, prediction_valid)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

X_train_scaled=dask_ml.preprocessing.scale(train[inputvar]) #Standardisation des données d'entree de la base d'apprentissage
Y_train_scaled=dask_ml.preprocessing.scale(train['fare_amount'])  #Standardisation des données de sortie de la base d'apprentissage


lr.fit(X_train_scaled, Y_train_scaled)

prediction_valid=lr.predict(X_valid_scaled)
confusion_mat=confusion_matrix(Y_valid_scaled, prediction_valid)

# Calculer le RMSE et le R² sur le jeu de test.


# ---------- Utiliser une librairie usuelle

prediction_lm=regr.predict(X_train_scaled)

print ('Erreur (RMS) : %.2f' %mean_squared_error(Y_train_scaled, prediction_lm))
print ('Variance de score : %.2f' %r2_score(Y_train_scaled, prediction_lm))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

prediction_biglm=lr.predict(X_train_scaled)
print ('Erreur (RMS) : %.2f' %mean_squared_error(Y_train_scaled, prediction_biglm))
print ('Variance de score : %.2f' %r2_score(Y_train_scaled, prediction_biglm))

# Quelle est la qualité de la prédiction sur le jeu de test ?


REPONSE ECRITE (3 lignes maximum)








#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 



### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil


# ---------- Utiliser une librairie usuelle

Y_binaire=np.zeros(len(Y))
Y_binaire[Y > Y.median()]=1

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

Y_binaire=dask.array.zeros(len(Y))
Y_binaire[dask.array(Y > Y.median())]=1


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

log_reg=LogisticRegression()
log_reg.fit(var_entree_stand, Y_binaire)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


log_reg=LogisticRegression()
log_reg.fit(var_entree_stand, Y_binaire)




### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

import random  #Librairie pour génerer un nombre aléatoire

def generate_dataset(n, t):  #Fonction qui génère un vecteur aléatoire de taille n avec une probabilité de sélection de t.
    return [1 if random.random() < t else 0 for i in range(0,n)]

def custom_split_train_test(ens, p):  #Fonction qui repartie la population en deux.
    choice = generate_dataset(len(ens), p)
    train = [x for x, c in zip(ens, choice) if c == 1] #Sélection d'un échantillon de taille représentant p pourcent de la population totale
    autre = [x for x, c in zip(ens, choice) if c == 0] #Sélection d'un échantillon de taille représentant 1-p pourcent de la population totale
    return train, autre #Restition des deux tables

train, autre = custom_split_train_test(df_ech, 0.6)  #Obtient la base d'apprentissage et une autre base
test, validation= custom_split_train_test(autre, 0.5)  #Obtient les base test et de validation



# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

import dask.array

from dask.array import random  #Librairie pour génerer un nombre aléatoire

def generate_dataset(n, t):  #Fonction qui génère un vecteur aléatoire de taille n avec une probabilité de sélection de t.
    return [1 if dask.array.random() < t else 0 for i in range(0,n)]

def custom_split_train_test(ens, p):  #Fonction qui repartie la population en deux.
    choice = generate_dataset(len(ens), p)
    train = [x for x, c in zip(ens, choice) if c == 1] #Sélection d'un échantillon de taille représentant p pourcent de la population totale
    autre = [x for x, c in zip(ens, choice) if c == 0] #Sélection d'un échantillon de taille représentant 1-p pourcent de la population totale
    return train, autre #Restition des deux tables

train, autre = custom_split_train_test(df_ech, 0.6)  #Obtient la base d'apprentissage et une autre base
test, validation= custom_split_train_test(autre, 0.5)  #Obtient les base test et de validation


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

X_train_scaled=preprocessing.scale(train[inputvar]) #Standardisation des données d'entree de la base d'apprentissage
Y_train=train['fare_amount']
Y_train_binaire[np.zeros(len(Y_train)) > np.zeros(len(Y_train)).median()]=1


X_valid_scaled=preprocessing.scale(validation[inputvar]) #Standardisation des données d'entree de la base de validation
Y_valid=validate['fare_amount']
Y_valid_binaire[np.zeros(len(Y_valid)) > np.zeros(len(Y_valid)).median()]=1

log_reg=LogisticRegression()#modelisation
log_reg.fit(X_train_scaled, Y_train_binaire)

prediction_logreg=log_reg.predict(X_valid_scaled)
pred_proba_logreg=log_reg.predict_proba(Y_valid_binaire)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

X_train_scaled=dask_ml.preprocessing.scale(train[inputvar]) #Standardisation des données d'entree de la base d'apprentissage
X_valid_scaled=dask_ml.preprocessing.scale(validation[inputvar]) #Standardisation des données d'entree de la base de validation



blog_reg=dask_ml.linear_model.LogisticRegression()
blog_reg.fit(X_train_scaled, Y_train_binaire)

prediction_biglogreg=blog_reg.predict(X_valid_scaled)
pred_proba_logreg=blog_reg.predict_proba(Y_valid_binaire)



# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.



# ---------- Utiliser une librairie usuelle

from sklearn.metrics import accuracy_score
accuracy_score(Y_train_binaire, prediction_logreg) 

fig, ax = plt.subplots(figsize=(8, 8))
fpr, tpr, _ = roc_curve(Y_train_binaire, prediction_logreg)
ax.plot(fpr, tpr, lw=3,
label='ROC Curve (ares = {:.2f})'.format(auc(fpr, tpr)))
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set(
xlim=(0, 1),
ylim=(0, 1),
title="ROC Curve",
xlabel="False Positive Rate",
ylabel="True Positive Rate",
)
ax.legend();

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

from dask_ml.metrics import accuracy_score 
accuracy_score(Y_train_scaled, prediction_logreg) 


from sklearn.metrics import accuracy_score
accuracy_score(Y_train_binaire, prediction_logreg) 

fig, ax = plt.subplots(figsize=(8, 8))
fpr, tpr, _ = roc_curve(Y_train_binaire, prediction_logreg)
ax.plot(fpr, tpr, lw=3,
label='ROC Curve (ares = {:.2f})'.format(auc(fpr, tpr)))
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set(
xlim=(0, 1),
ylim=(0, 1),
title="ROC Curve",
xlabel="False Positive Rate",
ylabel="True Positive Rate",
)
ax.legend();


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