#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:38:02 2018

@author: bmoseley
"""
import numpy as np
from projectq import MainEngine
from projectq.ops import All, Measure, H, Rx

from helper import measureAllQubits

class VQE:
    ''' A VQE object for calculating the expected energy of a parameterised trial state.
    The expected energy can either be estimated by carrying out repeated measurements of the state (slow)
    or by cheating and calculating it exactly using the hidden state amplitudes in the simulator (fast)
    '''
    
    def __init__(self, hamiltonian, Hamiltonian, Ansatz):
        
        self.hamiltonian = hamiltonian
        self.Hamiltonian = Hamiltonian
        self.Ansatz = Ansatz
        self.n_qubits = Ansatz.n_qubits
        
        # basic error check
        for hamiltonian_term in hamiltonian:
            if self.n_qubits != len(hamiltonian_term)-2:
                raise Exception("Error: n_qubits does not match length of hamiltonian term (%i, %s)"%(self.n_qubits, hamiltonian_term))

    def _subtermMeasurement(self, qubits, engine, hamiltonian_term):
        ''' Performs a single measurement of one of the Pauli
            matrix terms in the Hamiltonian.
        '''
        # Changes basis such that measuring Pauli term becomes
        # equivalent to measuring in the computational basis.
        maskArray = np.zeros(len(qubits))
        for i,qubit in enumerate(qubits):
            gate = hamiltonian_term[i+2]
            if gate == 'I':
                maskArray[i] = 0
            elif gate == 'X':
                H | qubit
                maskArray[i] = 1
            elif gate == 'Y':
                Rx(np.pi/2) | qubit
                maskArray[i] = 1
            else:
                maskArray[i] = 1
        result = measureAllQubits(qubits, engine)
        # Calculate parity of relevant qubits in term.
        parityArray = maskArray*result
        paritySum = np.sum(parityArray)
        parityMeasurement = (-1)**paritySum
    
        return parityMeasurement

    def _runSubtermExpectation(self, parameters, n_repeats, hamiltonian_term):
        '''Find the expectation value of a single subterm in the Hamiltonian'''
        # this step can be parallelised across many CPU nodes, but be careful when handling the random seed in MainEngine
        
        n_qubits = self.n_qubits
        
        expectationSum = 0.
        for _ in range(n_repeats):
            engine = MainEngine()# Note each time this is called a new random seed is used by engine
            qubits = engine.allocate_qureg(n_qubits)
            self.Ansatz(parameters, qubits, engine)
            measuredValue = self._subtermMeasurement(qubits, engine, hamiltonian_term)
            expectationSum += measuredValue
        
        return expectationSum/n_repeats
    
    def runExpectation(self, parameters, n_repeats):
        ''' Estimates the expected energy of the state defined by the parameters by
            performing many state initialisations and measurements.
        '''
        hamiltonian = self.hamiltonian
        results = [self._runSubtermExpectation(parameters, n_repeats, hamiltonian_term) for hamiltonian_term in hamiltonian]
        energy = np.sum([results[i]*hamiltonian[i][1] for i in range(len(hamiltonian))])

        return energy

    def runCheatExpectation(self, parameters):
        ''' Uses ProjectQ cheat functions to quickly measure the exact energy
            expectation value.
        '''
        n_qubits = self.n_qubits
        Hamiltonian = self.Hamiltonian
        
        engine = MainEngine() # define engine
        qubits = engine.allocate_qureg(n_qubits)
        self.Ansatz(parameters, qubits, engine)
        
        engine.flush()# need to flush before accessing quantum results
        energy = engine.backend.get_expectation_value(Hamiltonian, qubits)
        
        All(Measure) | qubits# just to avoid deallocation error
        return energy
        
if __name__ == "__main__":
    
    import time
    
    from HamiltonianFile import HamiltonianFile
    from H2Ansatz import H2Ansatz
    
    # get hamiltonian
    Ham = HamiltonianFile('hamiltonians/H2at075.txt')
    Hamiltonian, hamiltonian = Ham.Hamiltonian, Ham.hamiltonian
    
    H2 = VQE(hamiltonian=hamiltonian, Hamiltonian=Hamiltonian, Ansatz=H2Ansatz)

    parameters = np.array([0.0, 0.0, 0.11483322638407246])

    start = time.time()
    energy = H2.runExpectation(parameters, n_repeats=1000)
    delta = time.time()-start
    print("runExpectation energy: %.7f (%.2f s  (%.0f min))"%(energy, delta, delta/60))
    
    start = time.time()
    energy = H2.runCheatExpectation(parameters)
    delta = time.time()-start
    print("runCheatExpectation energy: %.7f (%.2f s  (%.0f min))"%(energy, delta, delta/60))