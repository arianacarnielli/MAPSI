# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:04:25 2019

@author: arian
"""

import numpy as np
import pickle as pkl

# fonction de suppression des 0 (certaines variances sont nulles car les pixels valent tous la même chose)
def woZeros(x):
    y = np.where(x==0., 1., x)
    return y

# Apprentissage d'un modèle naïf où chaque pixel est modélisé par une gaussienne (+hyp. d'indépendance des pixels)
# cette fonction donne 10 modèles (correspondant aux 10 classes de chiffres)
# USAGE: theta = learnGauss ( X,Y )
# theta[0] : modèle du premier chiffre,  theta[0][0] : vecteur des moyennes des pixels, theta[0][1] : vecteur des variances des pixels
def learnGauss (X,Y):
    theta = [(X[Y==y].mean(0),woZeros(X[Y==y].var(0))) for y in np.unique(Y)]
    return (np.array(theta))

# Application de TOUS les modèles sur TOUTES les images: résultat = matrice (nbClasses x nbImages)
def logpobs(X, theta):
    logp = [[(-0.5*np.log(mod[1,:] * (2 * np.pi )) + -0.5 * ( ( x - mod[0,:] )**2 / mod[1,:] )).sum () for x in X] for mod in theta ]
    return np.array(logp)

######################################################################
#########################     script      ############################



#==============================================================================
# Tests
#==============================================================================

if __name__ == "__main__":
    
    # Données au format pickle: le fichier contient X, XT, Y et YT
    # X et Y sont les données d'apprentissage; les matrices en T sont les données de test
    with open("usps_small.pkl", 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    
    X = np.array(data['X'], dtype=float)
    Y = data['Y']
    XT = np.array(data['XT'], dtype=float)
    YT = data['YT']
    
    theta = learnGauss ( X,Y ) # apprentissage
    
    logp  = logpobs(X, theta)  # application des modèles sur les données d'apprentissage
    logpT = logpobs(XT, theta) # application des modèles sur les données de test
    
    ypred  = logp.argmax(0)    # indice de la plus grande proba (colonne par colonne) = prédiction
    ypredT = logpT.argmax(0)
   
    # Le valeurs ne sont pas les mêmes que celles données à l'énoncé. Il semble
    # que cela est lié à un problème d'overflow avec les float16 qu'il y avait
    #à l'origine dans X.
    print ("Taux bonne classification en apprentissage : ",np.where(ypred != Y, 0.,1.).mean())
    print ("Taux bonne classification en test : ",np.where(ypredT != YT, 0.,1.).mean())
    
    
    