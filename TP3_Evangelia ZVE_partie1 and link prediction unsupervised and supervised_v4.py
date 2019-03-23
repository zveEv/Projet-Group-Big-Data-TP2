# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 21:31:03 2019

@author: lilian
"""
#pip install libraries

! pip install pandas
! pip install networkx[all]
! pip install seaborn
! pip install python-louvain
! pip install igraph
! pip install itertools
! pip install maya.cmds
! pip install random

#import the useful packages
import networkx as nx
from community import community_louvain
from networkx.algorithms import community
from networkx.algorithms.community import k_clique_communities
import pandas as pd
import random as rd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

pathData = "C:/Users/lilian/Documents/M2 FOAD/webmining/TP3 à rendre/lesmis.gml"
g = nx.read_gml(pathData)

""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""  
    
    print("Vertices of graph:")
    print(g.nodes())

    print("Edges of graph:")
    print(g.edges())

    
# =============================================================================
# #Community detection

#In this section, we will focus on community detection algorithm. For this, have a look to the networkx package documentation and apply the following community detection algorithms:

#Kernighan–Lin bipartition algorithm
#Percolation method
#Fluid communities algorithm
#Girvan-Newman method
#When the number of communities to detect has to be specified as a parameter, you will use the coverage metric to select the appropriate number (ranging from 2 to 5).

#Finally, for each community algorithm, you will add an attribute to each node of the graph. The value of the attribute will be the identifier of the community tne node belongs to (ranging from 0 to nbCommunity -1).
# =============================================================================

#parts = community_louvain.best_partition(g)
#values = [parts.get(node) for node in g.nodes()]
#node list
print(g.nodes)   
nodes=['Myriel', 'Napoleon', 'MlleBaptistine','MmeMagloire', 'CountessDeLo', 'Geborand', 'Champtercier', 'Cravatte', 'Count', 'OldMan', 'Labarre', 'Valjean', 'Marguerite', 'MmeDeR', 'Isabeau', 'Gervais', 'Tholomyes', 'Listolier', 'Fameuil', 'Blacheville', 'Favourite', 'Dahlia', 'Zephine', 'Fantine', 'MmeThenardier', 'Thenardier', 'Cosette', 'Javert', 'Fauchelevent', 'Bamatabois', 'Perpetue', 'Simplice', 'Scaufflaire', 'Woman1', 'Judge', 'Champmathieu', 'Brevet', 'Chenildieu', 'Cochepaille', 'Pontmercy', 'Boulatruelle', 'Eponine', 'Anzelma', 'Woman2', 'MotherInnocent', 'Gribier', 'Jondrette', 'MmeBurgon', 'Gavroche', 'Gillenormand', 'Magnon', 'MlleGillenormand', 'MmePontmercy', 'MlleVaubois', 'LtGillenormand', 'Marius', 'BaronessT', 'Mabeuf', 'Enjolras', 'Combeferre', 'Prouvaire', 'Feuilly', 'Courfeyrac', 'Bahorel', 'Bossuet', 'Joly', 'Grantaire', 'MotherPlutarch', 'Gueulemer', 'Babet', 'Claquesous', 'Montparnasse', 'Toussaint', 'Child1', 'Child2', 'Brujon', 'MmeHucheloup']    
#identification du nombre ideal des communautés : 5 communautés identifiés
import community
communities = community.best_partition(g)
for value in sorted(communities.values()):
    print (value)

#1.Kernighan–Lin bipartition algorithm
bisectionID=nx.community.kernighan_lin_bisection(g)
bisectionID=[list(x) for x in bisectionID]
keys=[]
values=[]
for item in bisectionID[0]:
        keys.append(item)
        values.append(0)
 
for j in bisectionID[1]:
        keys.append(j)
        values.append(1)
        
 #nodes dans aucune communauté       
 for j in nodes:
      if j not in keys:
        keys.append(j)
        values.append(-1)

BisectionCommunityID = dict(zip(keys, values))

#2.Percolation method
percolationID=list(k_clique_communities(g, 5))
percolationID=[list(x) for x in percolationID]
keys=[]
values=[]
    for item in percolationID[0]:
        keys.append(item)
        values.append(0)
 
    for j in percolationID[1]:
        keys.append(j)
        values.append(1)
        
     for j in percolationID[2]:
        keys.append(j)
        values.append(2)
        
    for j in percolationID[3]:
        keys.append(j)
        values.append(3)
    
    for j in percolationID[4]:
        keys.append(j)
        values.append(4)
#nodes qui apparteient à aucune communauté       
    for j in g.nodes:
      if j not in keys:
        keys.append(j)
        values.append(-1)
         
percolationCommunityID = dict(zip(keys, values))

#3.Fluid communities algorithm
fluidcID=list(nx.community.asyn_fluidc(g, 5))
fluidcID=[list(x) for x in fluidcID]
keys=[]
values=[]
   for item in fluidcID[0]:
        keys.append(item)
        values.append(0)
 
   for j in fluidcID[1]:
        keys.append(j)
        values.append(1)
        
   for j in fluidcID[2]:
        keys.append(j)
        values.append(2)
        
   for j in fluidcID[3]:
        keys.append(j)
        values.append(3)
    
   for j in fluidcID[4]:
        keys.append(j)
        values.append(4)
       
   for j in nodes:
      if j not in keys:
        keys.append(j)
        values.append(-1) 
       
FluidcCommunityID = dict(zip(keys, values))


#4.Girvan-Newman method   
import itertools
k = 5
comp = nx.community.girvan_newman(g)
for communities in itertools.islice(comp, k):
    print(tuple(sorted(c) for c in communities))
        
GirvanID=list(nx.community.asyn_fluidc(g, 5))
GirvanID=[list(x) for x in GirvanID]
keys=[]
values=[]
   for item in GirvanID[0]:
        keys.append(item)
        values.append(0)
 
   for j in GirvanID[1]:
        keys.append(j)
        values.append(1)
        
   for j in GirvanID[2]:
        keys.append(j)
        values.append(2)
        
   for j in GirvanID[3]:
        keys.append(j)
        values.append(3)
    
   for j in GirvanID[4]:
        keys.append(j)
        values.append(4)
        
    for j in nodes:
      if j not in keys:
        keys.append(j)
        values.append(-1) 
       
GirvanCommunityID = dict(zip(keys, values))

#for each community algorithm, add an attribute to each node of the graph
nx.classes.function.set_node_attributes(g, BisectionCommunityID, name='bisectionID')
nx.classes.function.set_node_attributes(g, percolationCommunityID, name='percolationID')
nx.classes.function.set_node_attributes(g, FluidcCommunityID, name='fluidcID')
nx.classes.function.set_node_attributes(g, GirvanCommunityID, name='girvanID')

#filter out nodes with missing communities
k=0
v=0

filter1={}
for k,v in percolationCommunityID.items():
   if v!=-1:
      filter1[k]=v

#options
options = {
    'node_color' : list(filter1.values()), # a list that contains the community id for the nodes we want to plot
    'node_size' : 10000, 
    'cmap' : plt.get_cmap("jet"),
    'node_shape' : 'o',
    'with_labels' : True, 
    "width" : 0.1, 
    "font_size" : 15,
    "nodelist" : list(filter1.keys()), # A list that contains the labels of the nodes we want to plot
    "alpha" : 0.8   
}

plt.figure(figsize=(30,30))
nx.draw(g,**options)


#Link prediction
##Unsupervised

###build a Panda Series from the edges of the graph
g2=g.copy()

base=pd.Series(data=list(g2.edges.keys()))

#select a sample of size 50 from this series
data=base.sample(n=50)
datalist=list(data)
d=pd.DataFrame(data, columns=['edge'])
#edges in the sample have to be removed from a copied version of g
g2.remove_edges_from(datalist)
print(g2.edges())


#Repeat this process with the following link prediction metrics :

#Resource allocation index
rai=list(nx.resource_allocation_index(g2, ebunch=None))
rai=dict([((a,b), c) for a, b, c in rai])
for k, v in rai.items():
    rai[k] = float(v)    

dataframeDICT=pd.DataFrame(list(rai.items()),columns=['Edg2e','resource allocation index'])
    
#nx.set_edg2e_attributes(g2, rai, 'resource allocation index')
#dataf=nx.to_pandas_edg2elist(g2, source='source', targ2et='targ2et',nodelist=None, dtype=None, order=None)

top50datafrai=dataframeDICT.nlargest(50, 'resource allocation index')

dtop50datafrai=top50datafrai.isin(d)
print (dtop50datafrai[dtop50datafrai['Edg2e'] == True].count())


#Jaccard coefficient
jc=list(nx.jaccard_coefficient(g2, ebunch=None))
jc=dict([((a,b), c) for a, b, c in jc])
for k, v in jc.items():
    jc[k] = float(v)
    
   
dataframeDICT2=pd.DataFrame(list(jc.items()), columns=['Edg2e', 'Jaccard coefficient'])
    
#nx.set_edg2e_attributes(g2, jc, 'Jaccard coefficient')
#dataf=nx.to_pandas_edg2elist(g2, source='source', targ2et='targ2et',nodelist=None, dtype=None, order=None)
top50datafjc=dataframeDICT2.nlargest(50, 'Jaccard coefficient')

dtop50datafjc=top50datafjc.isin(d)
print (dtop50datafjc[dtop50datafjc['Edg2e'] == True].count())

#Adamic-Adar index
aai=list(nx.adamic_adar_index(g2, ebunch=None))
aai=dict([((a,b), c) for a, b, c in aai])
for k, v in aai.items():
    aai[k] = float(v)

dataframeDICT3=pd.DataFrame(list(aai.items()), columns=['Edg2e', 'Adamic-Adar index'])

#nx.set_edg2e_attributes(g2, aai, 'Adamic-Adar index')
#dataf=nx.to_pandas_edg2elist(g2, source='source', targ2et='targ2et',nodelist=None, dtype=None, order=None)
top50datafaai=dataframeDICT3.nlargest(50, 'Adamic-Adar index')

dtop50datafaai=top50datafaai.isin(d)
print (top50datafaai[top50datafaai['Edg2e'] == True].count())

#Preferential attachment
pa=list(nx.preferential_attachment(g2, ebunch=None))
pa=dict([((a,b), c) for a, b, c in pa])
for k, v in pa.items():
    pa[k] = float(v)

dataframeDICT4=pd.DataFrame(list(pa.items()), columns=['Edg2e', 'Preferential attachment'])
#nx.set_edg2e_attributes(g2, pa, 'Preferential attachment')
#dataf=nx.to_pandas_edg2elist(g2, source='source', targ2et='targ2et',nodelist=None, dtype=None, order=None)
top50datafpa=dataframeDICT4.nlargest(50, 'Preferential attachment')

dtop50datafpa=top50datafpa.isin(d)
print (top50datafpa[top50datafpa['Edg2e'] == True].count())


##Supervised

####1/ Set a variable sizeTestSet to 50, a variable sizeTrainingPositiveSet to the number of edges in g minus the size of the test set and, a variable sizeTrainingSet to 2 times the size of the positive training set.

sizeTestSet = 50
sizeTrainingPositiveSet = len(g.edges()) - sizeTestSet
sizeTrainingSet = 2 * sizeTrainingPositiveSet

####2/ We will build the positive training set and the test set. To do so, first copy the graph g into g_training. Second, generate a sample of size sizeTestSet, denoted by sampleTest, from the series of edges of g_training. 
#This sample will be your test set (we will apply our model on it and hope the existence of a link will be predicted). Then, remove from g_training the edges in sampleTest. Finally, convert the remaining edges as a series.

#copy the graph g into g_training
g_training=g.copy()
#generate a sample of size sizeTestSet, denoted by sampleTest, from the series of edges of g_training
from random import sample
sampleTest=sample(g_training.edges(), sizeTestSet)
#Graph for the test sample
g_test=nx.Graph()  
g_test.add_edges_from(sampleTest)

#remove from g_training the edges in sampleTest
g_training.remove_edges_from(sampleTest)
print(g_training.edges())

#Finally, convert the remaining edges as a series.
samplePositiveTraining=pd.Series(data=g_training.edges())

#3/ To balance the training set, we will randomly pick pairs of unconnected vertices (negative class). 
#The number of pairs should be equal to the number of considered connections (positive class) in the training set. Find a way to generate this negative training set and name it sampleNegativeTraining.
import numpy as np
non_edges = list(nx.non_edges(g_training))

sample_num = len(g_training.edges())
sample = sample(non_edges, sample_num)
sampleNegativeTraining=pd.Series(data=sample)

#add new edges in the training graph based on the negative sample
g_training.add_edges_from(sampleNegativeTraining)



#5/ Use the following code (and modify it if necessary) to create 2 empty data frames (one for the training set and the other for the test set).
import numpy as np
sampleTraining = pd.concat([samplePositiveTraining,sampleNegativeTraining,],ignore_index=True)
dfTraining_1 = pd.DataFrame((list(sampleTraining)), columns=["target","source"])
dfTraining_2 = pd.DataFrame(np.zeros((sizeTrainingSet, 11)), columns=features)
dfTraining=pd.concat([dfTraining_1,dfTraining_2],axis=1)

dftest_1 = pd.DataFrame((list(sampleTest)), columns=["target","source"])
dfTest_2 = pd.DataFrame(np.zeros((sizeTestSet, 11)), columns=features)
dfTest=pd.concat([dftest_1,dfTest_2],axis=1)

#4/ It is now time to calculate the features for each member of the training and test sets. The features list is presented below:

#size of the shortest path
#number of shortest paths
#for each community algorithm, does the vertices associated to a connection belongs to the same community (except -1) : 1 or 0
#for each link prediction algorithm, the strength of the connection
#The feature list is:

  
features = [
    "lShortestPath",
    "nbShortestPath",
    "bipartition",
    "percolation",
    "fluid",
    "girvan",
    "resource",
    "jaccard",
    "adamic",
    "preferential",
    "class",    
   ]
  
   

#1.size of the shortest path

#Calculate shortest paths and store them to training dataframe.

len_list = [] #placeholders

for row in dfTraining.itertuples():
    so, tar = row[1], row[2]
    length=nx.shortest_path_length(g, source=so, target=tar)
    len_list.append(length)

#Add these lists as new columns in the DF
dfTraining['lShortestPath'] = len_list                    

#Calculate shortest paths and store them to test dataframe.

len_list = [] #placeholders

for row in dfTest.itertuples():
    so, tar = row[1], row[2]
    length=nx.shortest_path_length(g, source=so, target=tar)
    len_list.append(length)

#Add these lists as new columns in the DF
dfTest['lShortestPath'] = len_list  


#2.number of shortest paths

n_spaths={}
def num_spaths(G):
    n_spaths = dict.fromkeys(G, 0.0)
    spaths = dict(nx.all_pairs_shortest_path(G))

    for source in G:
        for path in spaths[source].values():
            for node in path[1:]: # ignore firs element (source == node)
                n_spaths[node] += 1 # this path passes through `node`

return n_spaths

num_spaths(g)




#6/ Write a function calculateFeatures with the following specifications:
#bdeoin d'undictionnaire avec 11 colonnes qui contiendrent les valeurs des 11 variables par couple à mettre à jour

def calculateFeatures(sample, df, training, positive, feature_values):
    
     if training == True:
           
        dfTraining_1_update = pd.DataFrame((list(sample)), columns=["target","source"])
        dfTraining_2_update = pd.DataFrame(feature, columns=features)
        df_update=pd.concat([dfTraining_1,dfTraining_2],axis=1)
    
    elif training == False:
                
        dftest_1 = pd.DataFrame((list(sampleTest)), columns=["target","source"])
        dfTest_2 = pd.DataFrame(np.zeros((sizeTestSet, 11)), columns=features)
        df_update=pd.concat([dfTest_1,dfTest_2],axis=1)

    
    
    df.pd.update(df_update)   

  
return(df)






    

