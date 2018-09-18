#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 00:08:52 2018

@author: bmoseley
"""

import numpy as np
import random

import GPyOpt# for bayesian optimisation

'''
A collection of classical optimisers for searching for the ground state energy using a VQE object.
'''

def SPSA(stepSize, lookSize, n_steps, parameters, quickEnergy, VQE, n_repeats, seed=123):
    ''' Carries out the Simultaneous Pertubation Stochastic Approximation (SPSA) algorithm 
        to find the ground state energy. 
        Shown to converge stochastically to minima, 
        with a number of iterations similar to FD gradient desecent'''
    
    print("Runing SPSA..")
    
    parameters = np.copy(parameters)# important! otherwise modified in-place
    
    def getDeltaVector(parameters):
        ''' Creates the displacement vector for the SPSA algorithm. '''
    
        deltaVec = np.zeros(len(parameters))
        for i in range(len(parameters)):
            randForSign = random.random()
            if randForSign < 0.5:
                deltaVec[i] = -1
            else:
                deltaVec[i] = 1
        return deltaVec
    
    alpha = 0.3
    gamma = 0.2
    
    X = []
    Y = []
    Y_true = []
    lookSizes = []
    stepSizes = []
    
    random.seed(seed)
    for i in range(n_steps):
        
        if (i+1)%10 == 0: print("Running SPSA step %i of %i"%(i+1, n_steps))
        
        aK = stepSize/((i+1)**alpha)# gets smaller with i, starts at stepSize
        cK = lookSize/((i+1)**gamma)# gets smaller with i, starts at lookSize

        # Get displaced parameter values.
        deltaVector = getDeltaVector(parameters)
        parametersPlus = parameters + cK*deltaVector# look from increasingly smaller steps
        parametersMinus = parameters - cK*deltaVector
        
        # Calculate the energy at each displacement point, get gradient.
        if quickEnergy is True:
            energyPlus = VQE.runCheatExpectation(parametersPlus)
            energyMinus = VQE.runCheatExpectation(parametersMinus)
        else:
            energyPlus = VQE.runExpectation(parametersPlus, n_repeats)
            energyMinus = VQE.runExpectation(parametersMinus, n_repeats)
        
        gradientVector = (energyPlus-energyMinus)/(2*cK*deltaVector)# gradient vector
        
        # Update parameters
        parameters -= aK*gradientVector
        
        # convergence metrics
        X.append(np.copy(parameters))# important! otherwise modified in-place
        Y.append(np.average([energyPlus, energyMinus]))
        Y_true.append(VQE.runCheatExpectation(parameters))# append exact energy
        stepSizes.append(np.abs(aK*(energyPlus-energyMinus)/(2*cK)))
        lookSizes.append(cK)

    return np.array(X), np.array(Y), np.array(Y_true), np.array(stepSizes), np.array(lookSizes)



def bayesOptimisation(acquisition_type, n_steps, quickEnergy, VQE, n_repeats, seed=123):
    ''' Carries out Bayesian optimisation to find the ground state energy.'''
        
    def f_objective(X):
        # n x d  ->  n x 1 array
        Y = np.zeros((X.shape[0],1), dtype=float)
        for i,parameters in enumerate(X):# each row is a parameter example
            if quickEnergy is True:
                Y[i] = VQE.runCheatExpectation(parameters)
            else:
                Y[i] = VQE.runExpectation(parameters,  n_repeats)
        return Y
    domain = VQE.Ansatz.domain# class attribute
    
    np.random.seed(seed)
    myBopt = GPyOpt.methods.BayesianOptimization(f = f_objective,        # function to optimize       
                                                 domain = domain,        # box-constrains of the problem
                                                 acquisition_type = acquisition_type,
                                                 exact_feval = quickEnergy, # whether objective function is noisy
                                                 initial_design_numdata = 5)

    print("Runing Bayesian optimization..")
    
    myBopt.run_optimization(max_iter=n_steps-myBopt.initial_design_numdata, eps=-1, verbosity=True)
    
    # finally calculate true expectation values for best parameters
    print("Calculating true expectation values..")

    quickEnergy = True
    Y_true = f_objective(myBopt.X)
        
    return np.copy(myBopt.X), np.copy(myBopt.Y), Y_true


def best_value(X, Y, Y_true):
    '''
    Returns a vectors whose components i correspond to the minimum of Y[:i]
    '''
    n = Y.shape[0]
    X_best = np.zeros(X.shape)
    Y_best = np.zeros(Y.shape)
    Y_true_best = np.zeros(Y_true.shape)
    for i in range(n):
            imin = Y[:(i+1)].argmin()
            Y_best[i]=Y[imin]
            X_best[i]=X[imin]
            Y_true_best[i]=Y_true[imin]
    return X_best, Y_best, Y_true_best


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from HamiltonianFile import HamiltonianFile
    from H2Ansatz import H2Ansatz
    from VQE import VQE
    
    # TEST
    
    # get hamiltonian
    Ham = HamiltonianFile('hamiltonians/H2at075.txt')
    Hamiltonian, hamiltonian = Ham.Hamiltonian, Ham.hamiltonian
    
    H2 = VQE(hamiltonian=hamiltonian, Hamiltonian=Hamiltonian, Ansatz=H2Ansatz)
    
    ## RUN OPTIMISATIONS
    
    # fixed parameters
    n_steps = 100
    quickEnergy = True
    stepSize = 0.05
    lookSize = 0.05
    initial_parameters = np.array([np.pi/4, np.pi/4, np.pi/4])
    n_repeats = 0
    acquisition_type = "EI"
    
    print("\n** SPSA **")
    
    X,Y,Y_true,_,_ = SPSA(stepSize, lookSize, n_steps, initial_parameters, quickEnergy, H2, n_repeats)

    plt.figure()
    plt.title("SPSA")
    plt.plot(best_value(X,Y,Y_true)[2])
    plt.ylabel("Energy (Hartree)")
    print("Final energy guess: %.5f"%(np.min(Y_true)))
    
    print("\n** BAYESIAN OPTIMISATION **")
    
    X,Y,Y_true = bayesOptimisation(acquisition_type, n_steps, quickEnergy, H2, n_repeats)
    
    plt.figure()
    plt.title("Bayesian optimisation")
    plt.plot(best_value(X,Y,Y_true)[2])
    plt.ylabel("Energy (Hartree)")
    
    print("Final energy guess: %.5f"%(np.min(Y_true)))

    plt.show()