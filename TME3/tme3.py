# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 01:00:00 2019

@author: arian
"""
import numpy as np
from utils import read_file
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


def learnML_class_parameters(tab_img):
    return np.mean(tab_img, 0), np.var(tab_img, 0)

def learnML_all_parameters(tab_classes):
    return [learnML_class_parameters(x) for x in tab_classes]
    
def log_likelihood(img, class_params):
    mu, sigma2 = class_params
    valides = (sigma2 != 0)
    tab_log = -1 / 2 * np.log(2 * np.pi * sigma2[valides]) \
    - 1 / 2 * (img[valides] - mu[valides])**2 / sigma2[valides]
    return tab_log.sum()

def log_likelihoods(img, params):
    return np.array([log_likelihood(img, p) for p in params])
    
def classify_image(img, params):
    return log_likelihoods(img, params).argmax()

def  classify_all_images(tab_img, params):
    T = np.zeros((10,10))
    for i in range(tab_img.size):
        for img in tab_img[i]:
            T[i, classify_image(img, params)] += 1
        T[i, :] /= T[i, :].sum()
    return T

def dessine(classified_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace (0, 9, 10)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride = 1)

def dessine_bis(classified_matrix):
    fig, ax = plt.subplots()
    x = y = np.linspace (0, 9, 10)
    X, Y = np.meshgrid(x, y)
    mappable = ax.contourf(X, Y, classified_matrix)
    fig.colorbar(mappable)

if __name__ == "__main__":    
    training_data = read_file("2015_tme3_usps_train.txt")
    test_data = read_file("2015_tme3_usps_test.txt")
    #print(learnML_class_parameters(training_data[0]))
    #print(learnML_class_parameters(training_data[1]))
    parameters = learnML_all_parameters(training_data)
    print(log_likelihood(test_data[2][3], parameters[1]))
    print([log_likelihood(test_data[0][0], parameters[i]) for i in range(10)])
    print(np.allclose([log_likelihood(test_data[0][0], parameters[i]) for i in range(10)], \
                        [-80.594309481001218, -2030714328.0707991, -339.70961551873495, \
                         -373.97785273732529, -678.16479308314922, -364.62227994586954, \
                         -715.4508284953547,  -344286.66839952325, -499.88159107145611, \
                         -35419.208662902507]))
    print(log_likelihoods(test_data[1][5], parameters))
    print(np.allclose(log_likelihoods(test_data[1][5], parameters), \
                      np.array([-889.22508387,  184.03163176, -185.29589129, -265.13424326, \
                                -149.54804688, -215.85994204,  -94.86965712, -255.60771575, \
                                -118.95170104,  -71.5970028 ])))
    print(classify_image(test_data[1][5], parameters))
    print(classify_image(test_data[4][1], parameters))
    T = classify_all_images(test_data, parameters)
    print(T[0, 0])
    print(T[2, 3])
    print(T[5, 3])
    
    dessine(T)
    dessine_bis(T)