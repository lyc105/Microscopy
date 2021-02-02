# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:53:38 2021

@author: lucas_lyc
"""
import os, sys
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, invgamma, bernoulli, binom, gamma, multivariate_normal, poisson
import scipy.stats as ss
from numpy import random
import numpy as np

'''
Suppose there are m targets and n beacons, the input matrix is a m*n read matrix.
The read u_ij ~ Poisson(theta_ij)
where log(theta_ij) = beta_0 + beta_1*log d_ij
'''

def loglikeli(input_mat, beta_0, beta_1, pos_target, pos_beacon):
    '''
    input_mat: m*n read matrix;
    beta_0, beta_1: coefficents;
    pos_target: a 2-d vector, the positions of target
    pos_beacon: a 2-d vector, the positions of beacon
    '''
    res = 0
    m, n = input_mat.shape #WLOG, suppose m >= n
    for i in range(m):
        for j in range(n):
            lam = beta_0 + beta_1*np.log(np.linalg.norm(pos_target[i] - pos_beacon[j]))
            res += -np.exp(lam) + input_mat[i,j]*lam
    return res


def initialize(input_mat):
    # The initialization of beta_0: the overall mean of read
    beta_0 = np.exp(np.mean(input_mat))
    beta_1 = norm.rvs()
    # The initialization of positions
    pos_target = np.zeros(shape = (m,2))
    pos_beacon = np.zeros(shape = (n,2)) + 1
# =============================================================================
#     for j in range(1, n):
#         pos_beacon[j] = multivariate_normal.rvs(mean = np.zeros(2), cov = ((np.exp(input_mat[0, j]) - beta_0)/beta_1)**2*np.eye(2))
#     for i in range(1, m):
#         pos_target[i] = multivariate_normal.rvs(mean = np.zeros(2), cov = ((np.exp(input_mat[i, 0]) - beta_0)/beta_1)**2*np.eye(2))
# =============================================================================
    return beta_0, beta_1, pos_target, pos_beacon

def one_step_update(input_mat, beta_0, beta_1, pos_target, pos_beacon, mh_step = 0.1):
    m, n = input_mat.shape
    if_update = False
    #update beta_0 by mh
    ll0 = loglikeli(input_mat, beta_0, beta_1, pos_target, pos_beacon)
    beta_0_prop = norm.rvs(beta_0, mh_step)
    ll1 = loglikeli(input_mat, beta_0_prop, beta_1, pos_target, pos_beacon)
    accept_rate = min(np.exp(ll1 - ll0), 1) 
    if bernoulli.rvs(accept_rate):
        beta_0_new = beta_0_prop
        if_update = True
    else:
        beta_0_new = beta_0
    
    #update beta_1 by mh
    if if_update:
        ll0 = loglikeli(input_mat, beta_0_new, beta_1, pos_target, pos_beacon) #If not updated, we don't have to calculate ll0 again.
    beta_1_prop = norm.rvs(beta_1, mh_step)
    ll1 = loglikeli(input_mat, beta_0_new, beta_1_prop, pos_target, pos_beacon)
    accept_rate = min(np.exp(ll1 - ll0), 1)
    if bernoulli.rvs(accept_rate):
        beta_1_new = beta_1_prop
    else:
        beta_1_new = beta_1
    
    #update positions by hm, in 2-d case, we don't have to apply hmc, because a local update is also feasible.
    for i in range(m):
        ll0 = np.sum(list(map(lambda x: -np.exp(x[0]) + x[0]*x[1],\
                                 zip([beta_0 + beta_1*np.log(np.linalg.norm(pos_target[i] - posj)) for posj in pos_beacon], input_mat[i,:]))))
        pos_prop = multivariate_normal.rvs(mean = pos_target[i], cov = np.eye(2)/beta_1_new**2)        
        ll1 = np.sum(list(map(lambda x: -np.exp(x[0]) + x[0]*x[1],\
                                 zip([beta_0 + beta_1*np.log(np.linalg.norm(pos_prop - posj)) for posj in pos_beacon], input_mat[i,:]))))
        accept_rate = min(np.exp(ll1 - ll0), 1)
        if bernoulli.rvs(accept_rate):
            pos_target[i] = pos_prop.copy()
                
        
    for j in range(n):
        ll0 = np.sum(list(map(lambda x: -np.exp(x[0]) + x[0]*x[1],\
                                 zip([beta_0 + beta_1*np.log(np.linalg.norm(pos_beacon[j] - posi)) for posi in pos_target], input_mat[:,j]))))
        pos_prop = multivariate_normal.rvs(mean = pos_beacon[j], cov = np.eye(2)/beta_1_new**2)        
        ll1 = np.sum(list(map(lambda x: -np.exp(x[0]) + x[0]*x[1],\
                                 zip([beta_0 + beta_1*np.log(np.linalg.norm(pos_prop - posi)) for posi in pos_target], input_mat[:,j]))))
        accept_rate = min(np.exp(ll1 - ll0), 1)
        if bernoulli.rvs(accept_rate):
            pos_beacon[j] = pos_prop.copy()    
            
    pos_target_new = pos_target
    pos_beacon_new = pos_beacon
    
    likeli = loglikeli(input_mat, beta_0_new, beta_1_new, pos_target_new, pos_beacon_new)
    
    return beta_0_new, beta_1_new, pos_target_new, pos_beacon_new, likeli


def main_gibbs(input_mat, mh_step = 0.1, max_iter = 10000):
    likeli = []
    beta_0_seq = []
    beta_1_seq = []
    pos_target_seq = []
    pos_beacon_seq = []
    beta_0, beta_1, pos_target, pos_beacon = initialize(input_mat)
# =============================================================================
#     beta_0_seq.append(beta_0)
#     beta_1_seq.append(beta_1)
#     pos_target_seq.append(pos_target)
#     pos_beacon_seq.append(pos_beacon)
# =============================================================================
    start_time = time.time()
    for _ in range(max_iter):
        if _%20 == 0:
            print(_, time.time()-start_time)
        beta_0, beta_1, pos_target, pos_beacon, cur_likeli = one_step_update(input_mat, beta_0, beta_1, pos_target, pos_beacon, mh_step)

        beta_0_seq.append(beta_0.copy())
        beta_1_seq.append(beta_1.copy())
        pos_target_seq.append(pos_target.copy())
        pos_beacon_seq.append(pos_beacon.copy())
        
        likeli.append(cur_likeli)
    return np.array(beta_0_seq), np.array(beta_1_seq), np.array(pos_target_seq), np.array(pos_beacon_seq), np.array(likeli)

if __name__ == "main":
    #Generating Data
    m, n = 50, 20
    true_beta_0 = 10
    true_beta_1 = -1
    true_pos_target = np.array([[np.cos(theta), np.sin(theta)] for theta in np.arange(0, 2*np.pi, 2*np.pi/m)]) +\
        multivariate_normal.rvs(mean = np.zeros(2), cov = np.eye(2), size = m)/10
    true_pos_beacon = np.array([[np.cos(theta), np.sin(theta)] for theta in np.arange(0, 2*np.pi, 2*np.pi/n)]) +\
        multivariate_normal.rvs(mean = np.zeros(2), cov = np.eye(2), size = n)/10
    input_mat = [[poisson.rvs(np.log(true_beta_0 + true_beta_1*np.linalg.norm(true_pos_target[i] - true_pos_beacon[j])))\
                              for j, pos_j in enumerate(true_pos_beacon)]  for i, pos_i in enumerate(true_pos_target) ]    
    input_mat = np.array(input_mat)
    #Inference by Gibbs Sampler
    beta_0, beta_1, pos_target, pos_beacon, likeli_curve = main_gibbs(input_mat)
    






