# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:20:34 2019

@author: arian
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

with open("ressources/lettres.pkl", 'rb') as f:
    data = pkl.load(f, encoding='latin1')
X = np.array(data.get('letters'))
Y = np.array(data.get('labels'))

nCl = 26

def discretise(X, nbEtats):
    """
    """
    intervalle = 360 / nbEtats
    res = np.empty(X.shape, dtype = X.dtype)
    for i in range(X.size): 
        res[i] = np.floor(X[i]/intervalle)
    return res

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
# Hypothèse Gauche-Droite
#==============================================================================

def initGD(X,N): 
    seq = np.empty(X.size, dtype = X.dtype)
    for i, xi in enumerate(X):
        seq[i] = np.floor(np.linspace(0, N-.00000001, len(xi)))
        #seq[i] = np.floor(np.linspace(0, N, len(xi), endpoint = False))
    return seq

#==============================================================================
# Apprentissage
#==============================================================================

def learnHMM(allx, alls, N, K, initTo0 = False, eps = 1e-8):
    if initTo0:
        A = np.zeros((N, N))
        B = np.zeros((N, K))
        Pi = np.zeros(N)
    else:
        A = np.ones((N, N)) * eps
        B = np.ones((N, K)) * eps
        Pi = np.ones(N) * eps
        
    for (x, s) in zip(allx, alls):
        for i in range(x.size):
            if i == 0:
                Pi[int(s[i])] += 1
            else:
                ai = int(s[i - 1])
                aj = int(s[i])
                A[ai, aj] += 1
            B[int(s[i]), int(x[i])] += 1
    
    A = A / np.maximum(A.sum(1).reshape(N, 1), eps) # normalisation
    B = B / np.maximum(B.sum(1).reshape(N, 1), eps) # normalisation
    Pi = Pi / Pi.sum()
   
    return Pi, A, B

#==============================================================================
# Viterbi (en log)
#==============================================================================
def viterbi(x, Pi, A, B):
    delta = np.zeros((x.size, A.shape[0]))
    psi = np.zeros((x.size, A.shape[0]), dtype = int)
    
    #initialisation
    delta[0, : ] = np.log(Pi) + np.log(B[:, int(x[0])])
    psi[0, : ] = -1

    #récursion
    for t in range(1, delta.shape[0]):
        for j in range(delta.shape[1]):
            aux = delta[t - 1, : ] + np.log(A[ : , j])
            psi[t, j] = aux.argmax()
            delta[t, j] = aux[psi[t, j]] + np.log(B[j, int(x[t])])
    
    state_est = np.zeros((x.size), dtype = int)  
    state_est[-1] = delta[-1, : ].argmax()
    for t in range(x.size - 2, -1, -1):
        state_est[t] = psi[t + 1, state_est[t + 1]]  
    
    proba_est = delta[-1, state_est[-1]]
    return state_est, proba_est

#==============================================================================
# Probabilité d'une séquence d'observation
#==============================================================================
def calc_log_pobs_v2(x, Pi, A, B):
    alpha = np.zeros((x.size, A.shape[0]))

    
    #initialisation
    alpha[0, : ] = np.log(Pi) + np.log(B[:, int(x[0])])
    
    #récursion
    for t in range(1, alpha.shape[0]):
        for j in range(alpha.shape[1]):
            aux = alpha[t - 1, : ] + np.log(A[ : , j])
            auxmax = aux.max()
            if auxmax == - np.inf:
                alpha[t, j] = - np.inf
            else:
                alpha[t, j] = np.log((np.exp(aux[:] - auxmax)).sum()) + auxmax + np.log(B[j, int(x[t])])     
      
    alphamax = alpha[-1, :].max()
    if alphamax == - np.inf:
        return - np.inf
    proba_est =  np.log((np.exp(alpha[-1, :] - alphamax)).sum()) + alphamax   
    return proba_est

#==============================================================================
# Apprentissage complet (Baum-Welch simplifié)
#==============================================================================
def learnBW(Xd, Y, N, K, initTo0 = False, eps = 1e-8, tol = 1e-4):
    modeles = np.empty(np.unique(Y).size, dtype = object)
    index = groupByLabel(Y)
    
    etats = initGD(Xd, N)
    error = np.inf
    vraisemblance = [np.inf]
    
    while abs(error) > tol:
        vraisemblance.append(0)
        for i in range(len(index)): 
            Pi, A, B = learnHMM(Xd[index[i]], etats[index[i]], N, K, initTo0, eps)
            modeles[i] = (Pi, A, B)
            for l in index[i]:
                s, v = viterbi(Xd[l], Pi, A, B)
                vraisemblance[-1] += v
                etats[l] = s
        error = (vraisemblance[-1] - vraisemblance[-2]) / vraisemblance[-1]    
    
    return modeles, vraisemblance[1:]

#==============================================================================
# Evaluation des performances
#==============================================================================
def evalModel(X, Y, N, K, p = 0.8):
    itrain, itest = separeTrainTest(Y, p)
    ia = []
    for i in itrain:
        ia += i.tolist()    
    it = []
    for i in itest:
        it += i.tolist()

    Xd = discretise(X, K)  # application de la discrétisation
    
    modeles, _ = learnBW(Xd[ia], Y[ia], N, K)
    
    proba = np.zeros((Xd.size, modeles.size))
    for i, x in enumerate(Xd):
        for j, (Pi, A, B) in enumerate(modeles):        
            _, proba[i, j] = viterbi(x, Pi, A, B)
    pred = proba.argmax(1) # max ligne par ligne
    
    Ynum = np.zeros(Y.shape)
    for num, char in enumerate(np.unique(Y)):
        Ynum[Y == char] = num
    
    tauxTrain = np.where(pred[ia] != Ynum[ia], 0.,1.).mean()
    tauxTest = np.where(pred[it] != Ynum[it], 0.,1.).mean()
    
    return tauxTrain, tauxTest

#==============================================================================
# Tests
#==============================================================================

if __name__ == "__main__":
    
#==============================================================================
# Apprentissage
#==============================================================================
    
    K = 10 # discrétisation (=10 observations possibles)
    N = 5  # 5 états possibles (de 0 à 4 en python) 
    Xd = discretise(X, K)
    SeqEtats = initGD(Xd, N)
      
    Pi, A, B = learnHMM(Xd[Y=='a'],SeqEtats[Y=='a'], N, K, initTo0 = True)
    #on diminue la précision de allclose pour être capable de tester avec les valeurs données à l'énoncé.
    print(Pi)
    print(np.allclose(Pi, [ 1.,  0.,  0.,  0.,  0.]))
    print(A)
    print(np.allclose(A, [[ 0.79,  0.21,  0.,    0.,    0.  ],
                          [ 0.,    0.76,  0.24,  0.,    0.  ],
                          [ 0.,    0.,    0.77,  0.23,  0.  ],
                          [ 0.,    0.,    0.,    0.76,  0.24],
                          [ 0.,    0.,    0.,    0.,    1.  ]], atol=1e-2))
    print(B)
    print(np.allclose(B, [[ 0.06,  0.02,  0.,    0.,    0.,    0.,    0.,    0.04,  0.49,  0.4 ],
                          [ 0.,    0.04,  0.,    0.13,  0.09,  0.13,  0.02,  0.09,  0.41,  0.09],
                          [ 0.,    0.,    0.,    0.02,  0.12,  0.5,   0.31,  0.04,  0.,    0.  ],
                          [ 0.07,  0.,    0.,    0.,    0.,    0.,    0.26,  0.33,  0.2,   0.15],
                          [ 0.73,  0.12,  0.,    0.,    0.,    0.,    0.,    0.02,  0.02,  0.12]], atol=1e-2))
    
    s_est, p_est = viterbi(Xd[0], Pi, A, B)
    print(s_est)
    print(np.allclose(s_est, [ 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 2., 2.,\
                              2., 2., 3., 3., 3., 3., 4., 4., 4., 4., 4.]))
    print(p_est)
    print(np.allclose(p_est, -38.0935655456))
    
    p_est_bis = calc_log_pobs_v2(Xd[0], Pi, A, B)
    print(p_est_bis)
    print(np.allclose(p_est_bis, -34.8950422805))
    
    modeles, vraisemblance = learnBW(Xd, Y, N, K)
    
#==============================================================================
#Tracer la courbe de l'évolution de la vraisemblance au cours des itérations      
#============================================================================== 
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(vraisemblance)
    ax.set_title("Évolution de la vraisemblance")
    ax.set_xlabel("itération")
    ax.set_ylabel("log(Vraisemblance)")
    ax.set_xticks(range(len(vraisemblance)))
    #plt.show()       

#==============================================================================
# Evaluation des performances
#==============================================================================         
    p = 0.8
    tauxTrain, tauxTest = evalModel(X, Y, N, K, p = p)
    print("Taux de bonne classification pour l'ensemble d'aprentissage, pc = 0.8 : ", tauxTrain)
    print("Taux de bonne classification pour l'ensemble de test, pc = 0.8 : ", tauxTest)
    
#==============================================================================
# matrice de confusion
#==============================================================================
    conf = np.zeros((26,26))
    #On recrée les variables pour être sûr de ce qu'on a :

    
    itrain, itest = separeTrainTest(Y, p)
    ia = []
    for i in itrain:
        ia += i.tolist()    
    it = []
    for i in itest:
        it += i.tolist()
        
    Ynum = np.zeros(Y.shape)
    for num, char in enumerate(np.unique(Y)):
        Ynum[Y == char] = num
    
    modeles, _ = learnBW(Xd[ia], Y[ia], N, K)    
    proba = np.zeros((Xd.size, modeles.size))
    for i, x in enumerate(Xd):
        for j, (Pi, A, B) in enumerate(modeles):        
            _, proba[i, j] = viterbi(x, Pi, A, B)
    pred = proba.argmax(1) # max ligne par ligne
    
    for i in it:
        conf[pred[i], int(Ynum[i])] += 1
        
    plt.figure()
    plt.imshow(conf, interpolation = 'nearest')
    plt.colorbar()
    plt.xticks(np.arange(26), np.unique(Y))
    plt.yticks(np.arange(26), np.unique(Y))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("mat_conf_lettres.png")
    #plt.show();



