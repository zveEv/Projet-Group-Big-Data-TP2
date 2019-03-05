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


CODE
library(data.table)
library(bigmemory)
library(biganalytics)
library(tidyverse)
library(dplyr)


#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 


### Q1.1 - Indiquer le dossier et le fichier cible



dossier <- "C:/Users/HP ProBooK/Desktop/M2_StatEco_UTC/AN_2/Big_Data/TP/TP2/data/"
data <- list.files("C:/Users/HP ProBooK/Desktop/M2_StatEco_UTC/AN_2/Big_Data/TP/TP2/data/") 
chemin_db <- paste0(dossier, data)
chemin_db

### Q1.2 - Importer les jeux de donn?es complets et ?chantillonn?es
###        Prediction du prix du taxi ? New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier ?chantillonn?e)



train_echant <- fread(chemin_db[2], sep = ",")
head(train_echant)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version compl?te du fichier)


# train <- read.big.matrix(chemin_db[1], sep = ",", header = T, backingfile = "train.bin", descriptorfile = "train.desc")



#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 


### Q2.1 - Nettoyer et pr�parer les donn�es


# Enlever les valeurs incorrectes ou manquantes (si pertinent)


# ---------- Utiliser une librairie usuelle

CODE # Donnees manquantes:

apply(train_echant, 2, function(x){sum(is.na(x))})

### En consideranat la taille du jeu de donnee et du peu de donnees maquantes, on peut peut les supprimer:
train_echant <- na.omit(train_echant)
 
CODE # valeurs incorrectes:
### la variable "fare_amount" ne peut pas �tre negative et nous interessons a la variable "passenger_count" superieure a zero:
train_echant <- train_echant %>% filter(fare_amount > 0, passenger_count > 0)




# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



# Ne garder que les variables de g�olocalisation (pour le jeu de donn�es en ent�e) et
# la variable "fare_amount" pour la sortie


# ---------- Utiliser une librairie usuelle



train_echant_geo <- train_echant %>% select(fare_amount, pickup_longitude:dropoff_latitude)
                                         


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)



# Obtenir les caracteristiques statistiques de base des variables d'entree et de sortie
# (par exemple, min, moyenne, mediane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle



###les caracteristiques statistiques de base des variables d'entree et de sortie
CODE
summary(train_echant_geo)

### Valeurs aberantes (outliers):
###Pour la var "fare_amount" et de geolocalisation: nous supprimons les valeurs < 1% et > 99% de percentiles
CODE

train_echant_geo <- train_echant_geo %>% filter(fare_amount > 0.01, fare_amount < quantile(fare_amount, 0.99))

train_echant_geo <- train_echant_geo %>% filter(pickup_longitude > quantile(pickup_longitude, 0.01), 
                                        pickup_longitude < quantile(pickup_longitude, 0.99))

train_echant_geo <- train_echant_geo %>% filter(pickup_latitude > quantile(pickup_latitude, 0.01), 
                                        pickup_latitude < quantile(pickup_latitude, 0.99))

train_echant_geo <- train_echant_geo %>% filter(dropoff_longitude  > quantile(dropoff_longitude, 0.01), 
                                        dropoff_longitude  < quantile(dropoff_longitude, 0.99))

train_echant_geo <- train_echant_geo %>% filter(dropoff_latitude > quantile(dropoff_latitude, 0.01), 
                                        dropoff_latitude < quantile(dropoff_latitude, 0.99))




# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Visualiser les distributions des variables d'entree et de sortie (histogramme, pairplot)


# ---------- Utiliser une librairie usuelle


#### Histogramme
hist(train_echant_geo[, "fare_amount"], main = "fare_amount")
for(var in names(train_echant_geo)) {hist(train_echant_geo[, var], main = var)}

### Pairplot
pairs(train_echant_geo)



# Separer la variable a predire ("fare_amount") des autres variables d'entree
# Creer un objet avec variables d'entree et un objet avec valeurs de sortie (i.e. "fare_amount")



# ---------- Utiliser une librairie usuelle

### objet avec variables d'entree
x_input <- train_echant_geo %>% select(pickup_longitude:dropoff_latitude)

###objet avec valeurs de sortie
y_output <- train_echant_geo %>% select(fare_amount)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

code


# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle

x_input <- scale(x_input)
y_output <- scale(y_output)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


CODE







#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Realiser un clustering k-means sur les donnees d'entree standardisees


# ---------- Utiliser une librairie usuelle


for (k in 1:10) {
  kmeans_clusters <- kmeans(x_input, centers = k, iter.max = 200, algorithm = "Lloyd")
  print(paste0(k, "kmeans_clusters: ", kmeans_clusters$tot.withinss))
}
str(kmeans_clusters)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R� en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle




### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



REPONSE ECRITE (3 lignes maximum)





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs differentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle

index_plot <- sample(nrow(x_input), 1000)
pairs(x_input[index_plot,], col = kmeans_clusters$cluster[index_plot], pch = 19)









#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de donnees standardis?


# ---------- Utiliser une librairie usuelle

acp_transform <- prcomp(x_input, center = T, scale. = T)
print(acp_transform)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q4.2 - Reliser le diagnostic de variance avec un graphique ? barre (barchart)

summary(acp_transform)
barplot(c(1.3123^2, 0.998^2, 0.9054^2, 0.6772^2))

# ---------- Utiliser une librairie usuelle


CODE




### Q4.3 - Combien de composantes doit-on garder? Pourquoi?

 ltrois car les composantes 

REPONSE ECRITE (3 lignes maximum)




### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

biplot(acp_transform)

# ---------- Utiliser une librairie usuelle


CODE




### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


REPONSE ECRITE (3 lignes maximum)











#
# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une regression lineaire de la sortie "fare_amount" 
###        en fonction de l'entree (mise a l'echelle), sur tout le jeu de donnees


# ---------- Utiliser une librairie usuelle

reg <- lm(train_echant_geo$fare_amount~x_input)
summary(reg)

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q5.2 - Que pouvez-vous dire des resultats du modele? Quelles variables sont significatives?

#Tous les variables sont significatives, mais le modele n'est pas bien specifier: 12%

#REPONSE ECRITE (3 lignes maximum)



### Q5.3 - Predire le prix de la course en fonction de nouvelles entres avec une regression lineaire


predict(reg, as.data.frame(x_input))

# Diviser le jeu de donnees initial en echantillons d'apprentissage (60% des donnees), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

train_60 <-

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Realiser la regression lineaire sur l'echantillon d'apprentissage, tester plusieurs valeurs
# de regularisation (hyperparametre de la regression lineaire) et la qualite de prediction sur l'echantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Calculer le RMSE et le R� sur le jeu de test.



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


