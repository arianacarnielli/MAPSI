# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:56:25 2019

@author: arian
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def tirage(m):
    return np.random.uniform(-m, m, 2)

def monteCarlo(N):
    X, Y = np.random.uniform(-1, 1, (2, N))
    proba = ((X**2 + Y**2) <= 1).sum() / N 
    return 4 * proba, X, Y

def swapF(taut):
    tau = taut.copy()
    c1, c2 = np.random.choice(list(tau.keys()), 2, False)   
    tau[c1] = taut[c2]
    tau[c2] = taut[c1]
    return tau

def decrypt(mess, tau):
    return "".join([tau[l] for l in mess]) 

def logLikelihood(mess, mu, A, chars2index):
    res = 0
    for i, l in enumerate(mess):
        index = chars2index[l]
        if i == 0:
            res += np.log(mu[index])
        else:
            indexPrec = chars2index[mess[i - 1]]
            res += np.log(A[indexPrec, index])
    return res
    
def MetropolisHastings(mess, mu, A, tau, N, chars2index, verbose = False):
    decV = decrypt(mess, tau)
    logV = logLikelihood(decV, mu, A, chars2index)
    bestLog = logV
    bestTau = tau
    bestDec = decV
    for _ in range(N):
        tauN = swapF(tau)
        decN = decrypt(mess, tauN)
        logN = logLikelihood(decN, mu, A, chars2index)
        alpha = min(1, np.exp(logN - logV))
        if np.random.random() < alpha:
            tau = tauN
            decV = decN
            logV = logN
            if logV > bestLog:
                bestLog = logV
                bestTau = tau
                bestDec = decN
                if verbose:
                    print("log Vraisemblance :")
                    print(bestLog)
                    print("msg décodé au moment :")
                    print(bestDec)
                    print("")
    return bestDec, bestLog, bestTau

def identityTau (count):
    tau = {}
    for k in list(count.keys ()):
        tau[k] = k
    return tau

def MCMC(N, p):

    
#==============================================================================
# Tests
#==============================================================================


if __name__ == "__main__":
    
#==============================================================================
#     fig, ax = plt.subplots()
#     
#     # trace le carré
#     ax.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')
#     
#     # trace le cercle
#     x = np.linspace(-1, 1, 100)
#     y = np.sqrt(1- x*x)
#     ax.plot(x, y, 'b')
#     ax.plot(x, -y, 'b')
#     
#     # estimation par Monte Carlo
#     pi, x, y = monteCarlo(int(1e4))
#     
#     # trace les points dans le cercle et hors du cercle
#     dist = x*x + y*y
#     ax.plot(x[dist <=1], y[dist <=1], marker = "o", mec = "k", mfc = "g", ls = "")
#     ax.plot(x[dist>1], y[dist>1], marker = "o", mec = "k", mfc = "r", ls = "")
#     ax.set_aspect("equal")
#     fig.show()
#==============================================================================
    
    # si vos fichiers sont dans un repertoire "ressources"
    with open("countWar.pkl", 'rb') as f:
        (count, mu, A) = pkl.load(f, encoding='latin1')
    
    with open("secret2.txt", 'r') as f:
        secret2 = f.read()[0:-1] # -1 pour supprimer le saut de ligne
        
    with open("fichierHash.pkl", 'rb') as f:
        chars2index = pkl.load(f, encoding='latin1')
        
    tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
    
    #MetropolisHastings(secret2, mu, A, identityTau(count), 10000, chars2index, verbose = True)
    
    # ATTENTION: mu = proba des caractere init, pas la proba stationnaire
    # => trouver les caractères fréquents = sort (count) !!
    # distribution stationnaire des caracteres
    freqKeys = np.array(list(count.keys()))
    freqVal  = np.array(list(count.values()))
    
    # indice des caracteres: +freq => - freq dans la references
    rankFreq = (-freqVal).argsort()
    
    # analyse mess. secret: indice les + freq => - freq
    cles = np.array(list(set(secret2))) # tous les caracteres de secret2
    rankSecret = np.argsort(-np.array([secret2.count(c) for c in cles]))
    # ATTENTION: 37 cles dans secret, 77 en général... On ne code que les caractères les plus frequents de mu, tant pis pour les autres
    # alignement des + freq dans mu VS + freq dans secret
    tau_init = dict([(cles[rankSecret[i]], freqKeys[rankFreq[i]]) for i in range(len(rankSecret))])
    
    # Ajout des lettres pour permettres des permutations qui améliorent la qualité
    # Lettres qui ne sont pas dans les clés du dictionnaire
#    lettresManquantesSecret = set(freqKeys) - set(cles)
#    lettresManquantesFreq = freqKeys[rankFreq[rankSecret.size:]]
#    for ls, lf in zip(lettresManquantesSecret, lettresManquantesFreq):
#        assert ls not in tau_init
#        tau_init[ls] = lf
    # Cela rend le résultat pire : l'espace de recherche est beaucoup plus grand
    # maintenant, à la fin des itérations il y a encore beaucoup d'erreurs dans
    # le texte !
    
    MetropolisHastings(secret2, mu, A, tau_init, 50000, chars2index, verbose = True)
    
    