# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:44:41 2019

@author: arian
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

#==============================================================================
# Fonctions données
#==============================================================================

with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1')
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées 

# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return

def groupByLabel(y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y == i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

#==============================================================================
# 1. Discrétisation
#==============================================================================

def discretise(X,nbEtats):
    """
    """
    intervalle = 360 / nbEtats
    res = np.empty(X.shape, dtype = X.dtype)
    for i in range(X.size): 
        res[i] = np.floor(X[i]/intervalle)
    return res

#==============================================================================
# 2. Regrouper les indices des signaux par classe
#==============================================================================

#rien a faire

#==============================================================================
# 3. Apprendre les modèles CM
#==============================================================================

def learnMarkovModel(Xc, nbEtats):
    A = np.zeros((nbEtats, nbEtats))
    Pi = np.zeros(nbEtats)
    
    for x in Xc:
        for i in range(x.size):
            if i == 0:
                Pi[int(x[i])] += 1
            else:
                ai = int(x[i - 1])
                aj = int(x[i])
                A[ai, aj] += 1
    
    A = A / np.maximum(A.sum(1).reshape(nbEtats, 1), 1) # normalisation
    Pi = Pi / Pi.sum()
    
    return Pi, A

#==============================================================================
# 4. Stocker les modèles dans une liste 
#==============================================================================

#rien a faire

#==============================================================================
# 1. (log)Probabilité d'une séquence dans un modèle
#==============================================================================

def probaSequence(s, Pi, A):
    proba = 0
    for i in range(s.size):
        if i == 0:
            proba += np.log(Pi[int(s[0])])
        else:
            ai = int(s[i - 1])
            aj = int(s[i])
            proba += np.log(A[ai, aj])
    return proba

def probasSequence(s, model):
    probas = np.zeros(len(model))
    for i, (Pi, A) in enumerate(model):
        probas[i] = probaSequence(s, Pi, A)
    return probas

#==============================================================================
# Biais d'évaluation
#==============================================================================

def evalModel(X, Y, nbEtats, p, learn = learnMarkovModel):
    itrain, itest = separeTrainTest(Y, p)
    ia = []
    for i in itrain:
        ia += i.tolist()    
    it = []
    for i in itest:
        it += i.tolist()

    Xd = discretise(X, nbEtats)  # application de la discrétisation
    model = []
    for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
        model.append(learn(Xd[itrain[cl]], nbEtats))
    
    proba = np.array([probasSequence(Xd[i], model) for i in range(Xd.size)])
    pred = proba.argmax(1) # max ligne par ligne
    
    Ynum = np.zeros(Y.shape)
    for num, char in enumerate(np.unique(Y)):
        Ynum[Y == char] = num
    
    tauxTrain = np.where(pred[ia] != Ynum[ia], 0.,1.).mean()
    tauxTest = np.where(pred[it] != Ynum[it], 0.,1.).mean()
    
    return tauxTrain, tauxTest

#==============================================================================
# Lutter contre le sur-apprentissage
#==============================================================================

def learnMarkovModelAmeliore(Xc, nbEtats):
    A = np.ones((nbEtats, nbEtats))
    Pi = np.ones(nbEtats)
    
    for x in Xc:
        for i in range(x.size):
            if i == 0:
                Pi[int(x[i])] += 1
            else:
                ai = int(x[i - 1])
                aj = int(x[i])
                A[ai, aj] += 1
    
    A = A / np.maximum(A.sum(1).reshape(nbEtats, 1), 1) # normalisation
    Pi = Pi / Pi.sum()
    
    return Pi, A

#==============================================================================
# Tests
#==============================================================================

if __name__ == "__main__": 

    # 1. Discretisation
    nbEtats = 3
    res = discretise(X, nbEtats)
    print(res[0])
    print(np.allclose(res[0], [0., 2., 2., 2., 2., 2., 2., 2., 1., 1., 1., 1., 
                      1., 1., 2., 2., 2., 2., 0., 0., 0., 0., 0.]))
    
    # 3. Apprendre les modèles CM
    index = groupByLabel(Y) # groupement des signaux par classe
    Xa = res[index[0]]
    Pi, A = learnMarkovModel(Xa, nbEtats)
    
    print(Pi)
    print(A)
    
    print(np.allclose(Pi, [ 0.36363636,  0.        ,  0.63636364]))
    print(np.allclose(A, [[ 0.84444444,  0.06666667,  0.08888889],
                          [ 0.        ,  0.83333333,  0.16666667],
                          [ 0.11382114,  0.06504065,  0.82113821]]))
            
    # 1.(log)Probabilité d'une séquence dans un modèle
    model = []
    for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
        model.append(learnMarkovModel(res[index[cl]], nbEtats))
        
    probas = probasSequence(res[0], model)
    print(probas)
    print(np.allclose(probas, 
                [-13.491086  ,         -np.inf,         -np.inf,         -np.inf,
               -np.inf,         -np.inf,         -np.inf,         -np.inf,
               -np.inf,         -np.inf,         -np.inf,         -np.inf,
               -np.inf,         -np.inf,         -np.inf,         -np.inf,
               -np.inf,         -np.inf,         -np.inf,         -np.inf,
               -np.inf,         -np.inf,         -np.inf,         -np.inf,
               -np.inf, -12.48285678]))
    
    # 4. Stocker les modèles dans une liste 
    
    d = 20     # paramètre de discrétisation
    Xd = discretise(X, d)  # application de la discrétisation
    models = []
    for cl in range(len(np.unique(Y))): # parcours de toutes les classes et optimisation des modèles
        models.append(learnMarkovModel(Xd[index[cl]], d))
    
    # 2. Application de la méthode précédente pour tous les signaux et tous les modèles de lettres pour nbEtats = 20
    proba20 = np.array([probasSequence(Xd[i], models) for i in range (Xd.size)])
    
    # 3. Evaluation des performances
    Ynum = np.zeros(Y.shape)
    for num, char in enumerate(np.unique(Y)):
        Ynum[Y == char] = num
        
    #pour nbEtats = 20
    pred20 = proba20.argmax(1) # max ligne par ligne
    print(np.where(pred20 != Ynum, 0.,1.).mean())
    
    #pour nbEtats = 3
    proba = np.array([probasSequence(res[i], model) for i in range (res.size)])
    pred = proba.argmax(1) # max ligne par ligne
    print(np.where(pred != Ynum, 0.,1.).mean())
    
#==============================================================================
# Biais d'évaluation, notion de sur-apprentissage
#==============================================================================
    tauxTrain20, tauxTest20 = evalModel(X, Y, 20, 0.8)
    print(tauxTrain20, tauxTest20)
    
    #test de l'évolution des performances en fonction de la discrétisation 
    nbEtats = np.arange(3, 21)
    tauxTrain = np.zeros(nbEtats.size)
    tauxTest = np.zeros(nbEtats.size)
    for i, d in enumerate(nbEtats):
        tauxTrain[i], tauxTest[i] = evalModel(X, Y, d, 0.8) 
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(nbEtats, tauxTrain, label = "Train")
    ax.plot(nbEtats, tauxTest, label = "Test")
    ax.legend()
    ax.set_ylim((0,1))
    #plt.show()

#==============================================================================
# Lutter contre le sur-apprentissage
#==============================================================================
    tauxTrain20, tauxTest20 = evalModel(X, Y, 20, 0.8, learnMarkovModelAmeliore)
    print(tauxTrain20, tauxTest20)
    
    #test de l'évolution des performances en fonction de la discrétisation 
    nbEtats = np.arange(3, 21)
    tauxTrain = np.zeros(nbEtats.size)
    tauxTest = np.zeros(nbEtats.size)
    for i, d in enumerate(nbEtats):
        tauxTrain[i], tauxTest[i] = evalModel(X, Y, d, 0.8, learnMarkovModelAmeliore) 
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(nbEtats, tauxTrain, label = "Train")
    ax.plot(nbEtats, tauxTest, label = "Test")
    ax.legend()
    ax.set_ylim((0,1))
    #plt.show()
    
    
    
