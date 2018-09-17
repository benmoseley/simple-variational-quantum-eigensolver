#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:53:30 2018

@author: bmoseley
"""

import numpy as np
from projectq.ops import  (Rx, Rz, Ry, C)

# ANSATZ OPERATOR

class LiHAnsatz:
    
    n_parameters = 42
    n_qubits = 6
    domain = [{'name': 'p%i'%(i), 'type': 'continuous', 'domain': (0,2*np.pi)} for i in range(n_parameters)]
    name = "LiH"
    
    def __init__(self, parameters, qubits, engine):
        ''' Applies a 6-qubit hardware-efficient ansatz to qubit register'''
    
        if len(qubits) != LiHAnsatz.n_qubits: raise Exception("ERROR: Number of qubits in register does not equal LiHAnsatz.n_qubits")
        # Apply Ansatz circuit
        self.block1(parameters[0:18], qubits, engine)
        self.block2(parameters[18:42], qubits, engine)
        
        return
        
    def block1(self, parameters, qubits, engine):
        ''' Applies LiH block 1.'''
    
        for i,qubit in enumerate(qubits):
            Rx(parameters[2*i]) | qubit
            Rz(parameters[2*i+1]) | qubit
        
        C(Ry(parameters[12])) | (qubits[0], qubits[1])
        C(Ry(parameters[13])) | (qubits[1], qubits[2])
        C(Ry(parameters[14])) | (qubits[2], qubits[3])
        C(Ry(parameters[15])) | (qubits[3], qubits[4])
        C(Ry(parameters[16])) | (qubits[4], qubits[5])
        C(Ry(parameters[17])) | (qubits[5], qubits[0])
        
        engine.flush()
    
        return
    
    def block2(self, parameters, qubits, engine):
        ''' Applies LiH block 2.'''
    
        for i,qubit in enumerate(qubits):
            Rz(parameters[3*i]) | qubit
            Rx(parameters[3*i+1]) | qubit
            Rz(parameters[3*i+2]) | qubit
        
        C(Ry(parameters[18])) | (qubits[0], qubits[2])
        C(Ry(parameters[19])) | (qubits[1], qubits[3])
        C(Ry(parameters[20])) | (qubits[2], qubits[4])
        C(Ry(parameters[21])) | (qubits[3], qubits[5])
        C(Ry(parameters[22])) | (qubits[4], qubits[0])
        C(Ry(parameters[23])) | (qubits[5], qubits[1])
        
        engine.flush()
    
        return


if __name__ == "__main__":
    
    from projectq import MainEngine
    from projectq.ops import All, Measure
    
    from HamiltonianFile import HamiltonianFile
    from helper import printwf
    
    # get hamiltonian
    HFile = HamiltonianFile('hamiltonians/LiHat145.txt')
    Hamiltonian, hamiltonian = HFile.Hamiltonian, HFile.hamiltonian
    
    # measure energy expectation value using ansatz
    engine = MainEngine()
    parameters = np.arange(42)*0.01
    qubits = engine.allocate_qureg(6)
    LiHAnsatz(parameters, qubits, engine)# operators
    printwf(engine)
    engine.flush()# need to flush before accessing quantum results
    energy = engine.backend.get_expectation_value(Hamiltonian, qubits)
    All(Measure) | qubits# to avoid deallocation error
    print('Measured energy (ansatz): ', energy)
    
    # check with true eigenvalues and eigenvectors
    Hm, eigs = HFile.getMatrix(hamiltonian)
    gs = eigs[1][:,0]
    print()
    printwf(gs)
    engine = MainEngine()
    qubits = engine.allocate_qureg(6)
    engine.flush()# need to flush before accessing quantum results
    engine.backend.set_wavefunction(gs, qubits)
    engine.flush()
    energy = engine.backend.get_expectation_value(Hamiltonian, qubits)
    All(Measure) | qubits# to avoid deallocation error
    print('Expected ground state energy (eigenfunction expectation): ', energy)
    print('True ground state energy: ', eigs[0][0])