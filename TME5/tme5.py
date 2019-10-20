# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:12:54 2019

@author: arian
"""

import numpy as np
import scipy.stats as stats
import pydotplus as pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import graphviz as gv
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb

style = { "bgcolor" : "#6b85d1", "fgcolor" : "#FFFFFF" }

#==============================================================================
# Fonctions données
#==============================================================================


# fonction pour transformer les données brutes en nombres de 0 à n-1
def translate_data ( data ):
    # création des structures de données à retourner
    nb_variables = data.shape[0]
    nb_observations = data.shape[1] - 1 # - nom variable
    res_data = np.zeros ( (nb_variables, nb_observations ), int )
    res_dico = np.empty ( nb_variables, dtype=object )

    # pour chaque variable, faire la traduction
    for i in range ( nb_variables ):
        res_dico[i] = {}
        index = 0
        for j in range ( 1, nb_observations + 1 ):
            # si l'observation n'existe pas dans le dictionnaire, la rajouter
            if data[i,j] not in res_dico[i]:
                res_dico[i].update ( { data[i,j] : index } )
                index += 1
            # rajouter la traduction dans le tableau de données à retourner
            res_data[i,j-1] = res_dico[i][data[i,j]]
    return ( res_data, res_dico )


# fonction pour lire les données de la base d'apprentissage
def read_csv ( filename ):
    data = np.loadtxt ( filename, delimiter=',', dtype=np.str ).T
    names = data[:,0].copy ()
    data, dico = translate_data ( data )
    return names, data, dico

# etant donné une BD data et son dictionnaire, cette fonction crée le
# tableau de contingence de (x,y) | z
def create_contingency_table ( data, dico, x, y, z ):
    # détermination de la taille de z
    size_z = 1
    offset_z = np.zeros ( len ( z ) )
    j = 0
    for i in z:
        offset_z[j] = size_z      
        size_z *= len ( dico[i] )
        j += 1

    # création du tableau de contingence
    res = np.zeros ( size_z, dtype = object )

    # remplissage du tableau de contingence
    if size_z != 1:
        z_values = np.apply_along_axis ( lambda val_z : val_z.dot ( offset_z ),
                                         1, data[z,:].T )
        i = 0
        while i < size_z:
            indices, = np.where ( z_values == i )
            a,b,c = np.histogram2d ( data[x,indices], data[y,indices],
                                     bins = [ len ( dico[x] ), len (dico[y] ) ] )
            res[i] = ( indices.size, a )
            i += 1
    else:
        a,b,c = np.histogram2d ( data[x,:], data[y,:],
                                 bins = [ len ( dico[x] ), len (dico[y] ) ] )
        res[0] = ( data.shape[1], a )
    return res

def display_BN ( node_names, bn_struct, bn_name, style ):
    graph = pydot.Dot( bn_name, graph_type='digraph')

    # création des noeuds du réseau
    for name in node_names:
        new_node = pydot.Node( name,
                               style="filled",
                               fillcolor=style["bgcolor"],
                               fontcolor=style["fgcolor"] )
        graph.add_node( new_node )

    # création des arcs
    for node in range ( len ( node_names ) ):
        parents = bn_struct[node]
        for par in parents:
            new_edge = pydot.Edge ( node_names[par], node_names[node] )
            graph.add_edge ( new_edge )

    # sauvegarde et affaichage
    outfile = bn_name + '.png'
    graph.write_png( outfile )
    img = mpimg.imread ( outfile )
    plt.imshow( img )

def learn_parameters (bn_struct, ficname):
    # création du dag correspondant au bn_struct
    graphe = gum.DAG ()
    nodes = [ graphe.addNode () for i in range ( bn_struct.shape[0] ) ]
    for i in range ( bn_struct.shape[0] ):
        for parent in bn_struct[i]:
            graphe.addArc ( nodes[parent], nodes[i] )

    # appel au BNLearner pour apprendre les paramètres
    learner = gum.BNLearner ( ficname )
    learner.useScoreLog2Likelihood ()
    learner.useAprioriSmoothing ()
    return learner.learnParameters ( graphe )


#==============================================================================
# Exercice 2
#==============================================================================

def sufficient_statistics(data, dico, x, y, z):
    cont_tab = create_contingency_table(data, dico, x, y, z)
    
    chi_carre = 0
    for nz, tab_xyz in cont_tab:
        if nz != 0:
            nxz = tab_xyz.sum(1)
            nyz = tab_xyz.sum(0)
            
            for x in range(tab_xyz.shape[0]):
                for y in range(tab_xyz.shape[1]):
                    frac = (nxz[x] * nyz[y]) / nz 
                    if frac != 0:
                        chi_carre += ((tab_xyz[x, y] - frac)**2) / frac
    return chi_carre

#==============================================================================
# Exercice 3
#==============================================================================

def sufficient_statistics_degree(data, dico, x, y, z):
    cont_tab = create_contingency_table(data, dico, x, y, z)
    
    chi_carre = 0
    #|z != 0| 
    quant_z = 0
    for nz, tab_xyz in cont_tab:
        if nz != 0:
            nxz = tab_xyz.sum(1)
            nyz = tab_xyz.sum(0)
            quant_z += 1
            for x in range(tab_xyz.shape[0]):
                for y in range(tab_xyz.shape[1]):
                    frac = (nxz[x] * nyz[y]) / nz 
                    if frac != 0:
                        chi_carre += ((tab_xyz[x, y] - frac)**2) / frac
    #degre de liberte pour le test chi_carré                    
    degre = (len(dico[x]) - 1) * (len(dico[y]) - 1) * quant_z
    return chi_carre, degre

#==============================================================================
# Exercice 4
#==============================================================================

def indep_score(data, dico, x, y, z):
    quant_z = 1
    for zi in z:
        quant_z *= len(dico[zi])
    
    if len(data[0]) >= 5 * len(dico[x]) * len(dico[y]) * quant_z:
        chi_carre, degre = sufficient_statistics_degree(data, dico, x, y, z)
        return stats.chi2.sf(chi_carre, degre), degre
    return (-1, 1)

#==============================================================================
# Exercice 5
#==============================================================================

def best_candidate(data, dico, x, z, alpha):
    ymin = 1.1
    y = -1
    for yi in range(x):
        p_value, _ = indep_score(data, dico, x, yi, z)
        if p_value < ymin :
            ymin = p_value
            y = yi
    if ymin < alpha:
        return [y]
    return []

#==============================================================================
# Exercice 6
#==============================================================================

def create_parents(data, dico, x, alpha):
    z = []
    candi = best_candidate(data, dico, x, z, alpha) 
    while candi != []:
        z += candi
        candi = best_candidate(data, dico, x, z, alpha) 
    return z

#==============================================================================
# Exercice 7
#==============================================================================

def learn_BN_structure(data, dico, alpha):
    parents = np.empty(dico.size, dtype = object)
    for xi in range(dico.size):
        parents[xi] = create_parents(data, dico, xi, alpha)
    return parents

#==============================================================================
# Exercice 7bis
#==============================================================================

def display_BN_graphviz(node_names, bn_struct, bn_name, style):
    graph = gv.Digraph(name = bn_name, format = "svg", engine = "dot", encoding = "utf-8")
   
    # création des noeuds du réseau
    for name in node_names:
        graph.node(name, name, style="filled",
                               fillcolor=style["bgcolor"],
                               fontcolor=style["fgcolor"])
    # création des arcs
    for node in range(len(node_names)):
        parents = bn_struct[node]
        for par in parents:
            graph.edge(node_names[par], node_names[node])

    # sauvegarde et affaichage
    graph.render(view = True)


if __name__ == "__main__": 

    # Exercice 1
    
    # names : tableau contenant les noms des variables aléatoires
    # data  : tableau 2D contenant les instanciations des variables aléatoires
    # dico  : tableau de dictionnaires contenant la correspondance (valeur de variable -> nombre)
    names, data, dico = read_csv ("2015_tme5_asia.csv")
    
    # Exercice 2
    
    print(sufficient_statistics(data, dico, 1, 2, [3]))
    print(np.allclose(sufficient_statistics(data, dico, 1, 2, [3]), 3.9466591186668296))
    
    print(sufficient_statistics(data, dico, 0, 1, [2,3]))
    print(np.allclose(sufficient_statistics(data, dico, 0, 1, [2,3]), 16.355207462350094))
    
    print(sufficient_statistics(data, dico, 1, 3, [2]))
    print(np.allclose(sufficient_statistics(data, dico, 1, 3, [2]), 81.807449348140295))
    
    print(sufficient_statistics(data, dico, 5, 2, [1, 3, 6]))
    print(np.allclose(sufficient_statistics(data, dico, 5, 2, [1, 3, 6]), 1897.0))
    
    print(sufficient_statistics(data, dico, 0, 7, [4, 5]))
    print(np.allclose(sufficient_statistics(data, dico, 0, 7, [4, 5]), 3.2223237760949699))
    
    print(sufficient_statistics(data, dico, 2, 3, [5]))
    print(np.allclose(sufficient_statistics(data, dico, 2, 3, [5]), 130.0))
    
    # Exercice 3
    
    print(sufficient_statistics_degree(data, dico, 1, 2, [3]))
    print(np.allclose(sufficient_statistics_degree(data, dico, 1, 2, [3]), (3.9466591186668296, 2)))
    
    print(sufficient_statistics_degree(data, dico, 0, 1, [2,3]))
    print(np.allclose(sufficient_statistics_degree(data, dico, 0, 1, [2,3]), (16.355207462350094, 3)))
    
    print(sufficient_statistics_degree(data, dico, 1, 3, [2]))
    print(np.allclose(sufficient_statistics_degree(data, dico, 1, 3, [2]), (81.807449348140295, 2)))
    
    print(sufficient_statistics_degree(data, dico, 5, 2, [1, 3, 6]))
    print(np.allclose(sufficient_statistics_degree(data, dico, 5, 2, [1, 3, 6]), (1897.0, 8)))
    
    print(sufficient_statistics_degree(data, dico, 0, 7, [4, 5]))
    print(np.allclose(sufficient_statistics_degree(data, dico, 0, 7, [4, 5]), (3.2223237760949699, 4)))
    
    print(sufficient_statistics_degree(data, dico, 2, 3, [5]))
    print(np.allclose(sufficient_statistics_degree(data, dico, 2, 3, [5]), (130.0, 2)))
    
    # Exercice 4
    
    print(indep_score(data, dico, 1,3,[]))
    print(np.allclose(indep_score(data, dico, 1,3,[]), (2.38520176938e-19, 1)))
    
    print(indep_score(data, dico, 1, 7, []))
    print(np.allclose(indep_score(data, dico, 1, 7, []), (1.12562784979e-10, 1)))
    
    print(indep_score(data, dico, 0, 1,[2, 3]))
    print(np.allclose(indep_score(data, dico, 0, 1,[2, 3]), (0.000958828236575, 3)))
    
    print(indep_score(data, dico, 1, 2,[3, 4]))
    print(np.allclose(indep_score(data, dico, 1, 2,[3, 4]), (0.475266197894, 4)))  
    
    # Exercice 5
    
    print(best_candidate(data, dico, 1, [], 0.05))
    print(best_candidate(data, dico, 1, [], 0.05) == [])
    
    print(best_candidate(data, dico, 4, [], 0.05))
    print(best_candidate(data, dico, 4, [], 0.05) == [1])
    
    print(best_candidate(data, dico, 4, [1], 0.05))
    print(best_candidate(data, dico, 4, [1], 0.05) == [])
    
    print(best_candidate(data, dico, 5, [], 0.05))
    print(best_candidate(data, dico, 5, [], 0.05) == [3])
    
    print(best_candidate(data, dico, 5, [6], 0.05))
    print(best_candidate(data, dico, 5, [6], 0.05) == [3])
    
    print(best_candidate(data, dico, 5, [6, 7], 0.05))
    print(best_candidate(data, dico, 5, [6, 7], 0.05) == [2])
    
    # Exercice 6
    
    print(create_parents(data, dico, 1, 0.05))
    print(create_parents(data, dico, 1, 0.05) == [])
    
    print(create_parents(data, dico, 4, 0.05))
    print(create_parents(data, dico, 4, 0.05) == [1])
    
    print(create_parents(data, dico, 5, 0.05))
    print(create_parents(data, dico, 5, 0.05) == [3, 2])
    
    print(create_parents(data, dico, 6, 0.05))
    print(create_parents(data, dico, 6, 0.05) == [4, 5])
    
    # Exercice 7
    
    alpha = 0.05
    bn_struct = learn_BN_structure(data, dico, alpha)
    #display_BN(names, bn_struct, "asia", style)
    display_BN_graphviz(names, bn_struct, "asia", style)
    
    # Exercice 7bis
    
    # création du réseau bayésien à la aGrUM
    bn = learn_parameters ( bn_struct, "2015_tme5_asia.csv")
    
    # affichage de sa taille
    print(bn)
    
    # récupération de la ''conditional probability table'' (CPT) et affichage de cette table
    gnb.showPotential(bn.cpt(bn.idFromName('bronchitis?')))
    
    # calcul de la marginale
    proba = gum.getPosterior(bn, {}, 'bronchitis?')
    
    # affichage de la marginale
    gnb.showPotential(proba)
    
    #calcul d'une distribution marginale a posteriori : P(bronchitis? | smoking? = true, turberculosis? = false ) 
    gnb.showPotential(gum.getPosterior( bn,{'smoking?': 'true', 'tuberculosis?' : 'false' }, 'bronchitis?'))