import pandas as pd
import numpy as np
import scipy
from numpy.linalg import norm
from numpy.random import default_rng
from random import choice
import json

import scipy.stats as st
from sklearn.metrics.pairwise import pairwise_distances
import pickle
import importlib
import os, sys
import ot

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rc
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import StrMethodFormatter

import time 

from scipy.optimize import linear_sum_assignment
import re

def cost_matrix(X, Y):
    """
    Функция для построения ценовой матрицы C, где c_ij = ||X_i - Y_j||_2^2

    Параметры:
        X - array - выборка
        Y - array - выборка 
    Выход:
        cost - list[list] - ценовая матрица
    """
    x = np.shape(X)[0]
    y = np.shape(Y)[0]
    cost  = [[norm(X[i] - Y[j])**2 for i in range(x)] for j in range(y)]
    return(cost)

def gen_random_ball(dimension, num_points, rs = None, radius = 1):
    """
    Функция для генерации точек из равномерного распределения на шаре

    Параметры:
        dimension - int - размерность
        num_points - int - число объектов
        rs - int - для фиксации случайности
        radius - int - радиус шара
    Выход:
        ball - array - матрица размером num_points x dimension
    """

    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball)
    rng = default_rng(rs)
    random_directions = rng.standard_normal(size=(dimension, num_points))
    random_directions /= norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = rng.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    ball = radius * (random_directions * random_radii).T
    return ball

def split_data_randomly(d):
    """
    Функция случайно делит выборку из 2n точек на 2 группы по n точек

    Параметры:
        d - array -  матрица с данными
    Выход:
        s1 - array - объекты, попавшие в первую выборку
        s2 - array - объекты, попавшие во вторую выборку
    """
    ind = np.arange(d.shape[0])
    np.random.shuffle(ind)
    n = int(len(d)/2)
    ind1 = ind[:n]
    ind2 = ind[n:]

    s1 = d[ind1]
    s2 = d[ind2]
    return s1, s2

def compute_critical_level(ball, N, alpha):
    """
    Функция калибровки критического значения (алгоритм 3 из раздела 2.2.1 отчета)

    Параметры:
        ball - array - матрица с точками равномерного распределения на шаре
        N - int - число итераций
        alpha - float - уровень значимости
    Выход:
        cr_level - float - критическое значение
    """
    n = int(len(ball)/2)
    stat = []
    m0     = [1/n] * n
    m1     = [1/n] * n
    for i in range(0, N):
        s1, s2 = split_data_randomly(d = ball)
        M = cost_matrix(s1, s2)
        M = np.array(M)
        plan = ot.emd2(m0, m1, M) #ot distance
        stat.append(plan)

    q = (1 - alpha)*100
    cr_level = np.percentile(stat, q)
    return cr_level

def computeOT(s, t, mode):
    """
    Функция для подсчета оптимального транспортного плана 

    Параметры:
        s - array - выборка
        t - array - выборка
        mode - str - если равен "plan", то в функции находится оптимальный транспортный план, если значение равно "distance", то возвращается расстояние
    Выход:
        Если mode = "plan", то plan - array - искомый оптимальный план 
        Если mode = "distance", то plan - float - найденное расстояние
    """
    n = len(s)
    k = len(t)
    m0     = [1/n] * n
    m1     = [1/k] * k
    M = cost_matrix(s, t)
    M = np.array(M)
    if mode == 'plan':
        plan = ot.emd(m0, m1, M) #transport plan
    elif mode == 'distance':
        plan = ot.emd2(m0, m1, M)
    return plan

def compute_distance(s1, s2, t):
    """
    Функция для подсчета расстояние 2-Вассерштейна между эмпирическими распределениями (шаг 2 из алгоритма 3)

    Параметры:
        s1 - array - выборка
        s2 - array - выборка
        t - array - целевое распределение
    Выход:
        dst_fin - float - расстояние
    """  
    ##Step 2 in Algorithm 3: OT distance between ball partitions 
    ## induced by the transport of two data distributions
    L = len(s1)
    s = np.concatenate((s1, s2), axis = 0)
    plan = computeOT(s = t, t = s, mode = 'plan')
    # Матрица оптимального плана содержит только 2L ненулевых элементов,
    # однако из-за вычислительных погрешностей у найденного решения
    # будет много элементов, близких к нулю.
    # Отсекаем их вручную на уровне 10^{-6}
    plan = plan * np.abs(plan > 1e-6)
    ind = [np.nonzero(plan[i])[0][0] for i in range(2*L)] #indices of assigned distributions
    #split target t
    ind_t1 = []
    ind_t2 = []
    for i in range(0, len(s)):
        if i < L:
            ind_t1.append(ind[i])
        else:
            ind_t2.append(ind[i])
    t1 = t[ind_t1]
    t2 = t[ind_t2]
    dst_fin = computeOT(s = t1, t = t2, mode = 'distance')
    return dst_fin

# Вспомогательные функции для отрисовки графиков
def decompose(X, method = 'numpy.eigh'):
    """Собственное разложение матрицы X"""
    if method == "tf.eig":
        import tensorflow as tf
        A_tf = tf.convert_to_tensor(X)
        eigvals, eigvects = tf.linalg.eig(A_tf)
        eigvals, eigvects = eigvals.numpy(), eigvects.numpy()
    elif method == "tf.eigh":
        import tensorflow as tf
        A_tf = tf.convert_to_tensor(X)
        eigvals, eigvects = tf.linalg.eigh(A_tf)
        eigvals, eigvects = eigvals.numpy(), eigvects.numpy()
    elif method == "numpy.eig":
        eigvals, eigvects = np.linalg.eig(X)
    elif method == "numpy.eigh":
        eigvals, eigvects = np.linalg.eigh(X)
    else:
        raise NotImplementedError('Unknown method: ' + method)
    return eigvals, eigvects

def sqrtmInv(X, method='numpy.eigh'):
    """Вычисление обратной матрицы к корню матрицы X"""
    eigval, eigvects = decompose(X, method)
    Y = (eigvects / np.sqrt(np.maximum(eigval, 0))[np.newaxis,:]).dot(eigvects.T)
    return(Y)

def sqrtm(X, method='numpy.eigh'):
    """Вычисление корня симметричной матрицы"""
    eigval, eigvects = decompose(X, method)
    Y = (eigvects * np.sqrt(np.maximum(eigval, 0))[np.newaxis,:]).dot(eigvects.T)
    return(Y)

def BW(K1, K2, method='numpy.eigh'):
    """Вычисление 2-Вассерштейн расстояния между матрицами ковариаций нормального распределения"""
    Q = sqrtm(K1, method)
    d = np.sqrt(np.maximum(0, K1.trace() + K2.trace() - 2 * sqrtm(Q.dot(K2).dot(Q), method).trace()))
    return d

def OT_map(V, U):
    #map from V to U
    sqU = sqrtm(U,method='numpy.eigh')
    Cn  =  (sqU @ V) @ sqU
    Z = sqrtmInv(Cn, method='numpy.eigh')
#     pinvCn = pinvsq(Cn)
    T = (sqU @ Z) @ sqU
    return T

def OT_geod(V, T, t):
    E = np.array([[1,0],[0, 1]])
    Z = E * (1-t) + t*T
    V = Z @ V 
    W = V @ Z
    return W

#main part begins
#таблица, для которой фиксируются критические значения
grd = 50*np.array(1 + np.arange(20))
print(grd)

test_values  = []

critical_levels = []
for g in tqdm(grd):
    # Алгоритм 2, шаг 2
    start_time = time.time()
    b = gen_random_ball(dimension=1, num_points=2*g, rs=None, radius = 1)
    print("gen random ball time: ",time.time() - start_time)
    # Алгоритм 3
    start_time = time.time()
    cr_l = compute_critical_level(ball = b, N=1000, alpha=.05)
    print("gen critical level time: ",time.time() - start_time)
    critical_levels.append(cr_l)

data = {'grid':[],'crit_vals':[]}
data['grid'] = grd.tolist()
data['crit_vals'] = critical_levels

with open('crit_vals_d_1.json', 'w') as fp:
    json.dump(data, fp)
