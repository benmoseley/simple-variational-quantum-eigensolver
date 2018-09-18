#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:38:02 2018

@author: bmoseley
"""

'''
Main code which searches for the ground state energy of a molecule specified by a Hamiltonian instance
using an Ansatz circuit object, a VQE instance and an optimiser.

Tests both SPSA and Bayesian optimisation optimisers.

Runs each optimiser with a n_initialisations different starting initialisations so that statistics on
convergence can be gathered. Plots the result.

'''

import numpy as np
import pickle
import multiprocessing

from HamiltonianFile import HamiltonianFile
from VQE import VQE
from H2Ansatz import H2Ansatz
from optimisers import SPSA, bayesOptimisation, best_value


# get ansatz
Ansatz = H2Ansatz

# get Hamiltonian
Ham = HamiltonianFile('hamiltonians/H2at075.txt')
Hamiltonian, hamiltonian = Ham.Hamiltonian, Ham.hamiltonian
H2 = VQE(hamiltonian=hamiltonian, Hamiltonian=Hamiltonian, Ansatz=H2Ansatz)

# fixed parameters
n_steps = 100
quickEnergy = True
n_repeats = 0
stepSize = 0.05
lookSize = 0.05
acquisition_type = "EI"

# other parameters
n_initialisations = 8# number of random initialisations (samples) to use
n_processes = 8# number of parallel processes to use

## RUN OPTIMISATIONS

# TODO: parallelisation should only be used for quickEnergy=True option (otherwise need to worry about random seed in VQE)
def run(i):
    
    print("Running initialisation %i of %i.."%(i+1, n_initialisations))
    
    np.random.seed(i)
    initial_parameters = np.pi*(np.random.rand(Ansatz.n_parameters))# randomly vary initialisation
        
    X,Y,Y_true,_,_ = SPSA(stepSize, lookSize, n_steps, initial_parameters, quickEnergy, H2, n_repeats, seed=i)
    
    Xb,Yb,Yb_true = bayesOptimisation(acquisition_type, n_steps, quickEnergy, H2, n_repeats, seed=i)
    
    return [X,Y,Y_true,Xb,Yb,Yb_true]

pool = multiprocessing.Pool(processes=n_processes)
results = pool.map(run, np.arange(n_initialisations))# returns list of results
pool.close()# clean up file pointers
pool.join()

#SAVE
pickle.dump(results, open("results/results.pickle", "wb"))


# PLOT
import matplotlib.pyplot as plt

plt.figure()
plt.title("SPSA")
for result in results: plt.plot(best_value(result[0],result[1],result[2])[2])
plt.ylabel("Energy (Hartree)")

plt.figure()
plt.title("Bayesian optimisation")
for result in results: plt.plot(best_value(result[3],result[4],result[5])[2])
plt.ylabel("Energy (Hartree)")

