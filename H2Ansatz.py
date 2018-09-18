#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:53:30 2018

@author: bmoseley
"""

import numpy as np
from projectq.ops import  (H, X, Rx, Rz, CNOT)

# ANSATZ OPERATOR

class H2Ansatz:
    '''Defines a projectq ansatz circuit to be used to simulate H2'''
    
    n_parameters = 3
    n_qubits = 4
    domain = [{'name': 'p%i'%(i), 'type': 'continuous', 'domain': (0,np.pi)} for i in range(n_parameters)]
    name = "H2"
        
    def __init__(self, parameters, qubits, engine):
        ''' Applies the 4-qubit UCC ansatz circut to qubit register, starting from a HF state.'''
        
        if len(qubits) != H2Ansatz.n_qubits: raise Exception("ERROR: Number of qubits in register does not equal H2Ansatz.n_qubits")
        # Create Hartree-Fock initial state
        X | qubits[0]# (X is a not gate)
        X | qubits[1]
        engine.flush()
    
        # Apply UCC circuit
        self.hydrogenUCCterm2(parameters[2], qubits, engine)
        self.hydrogenUCCterm1a(parameters[1], qubits, engine)
        self.hydrogenUCCterm1b(parameters[1], qubits, engine)
        self.hydrogenUCCterm0a(parameters[0], qubits, engine)
        self.hydrogenUCCterm0b(parameters[0], qubits, engine)
        
        return
        
    def hydrogenUCCterm0a(self, parameter, qubits, engine):
        ''' Applies the UCC term exp(i/2 X3 Z2 Y1) to the qubit register.'''
    
        Rx(np.pi/2) | qubits[1]
        H | qubits[3]
        CNOT | (qubits[1], qubits[2])
        CNOT | (qubits[2], qubits[3])
    
        Rz(-2*parameter) | qubits[3]
    
        CNOT | (qubits[2], qubits[3])
        CNOT | (qubits[1], qubits[2])
        H | qubits[3]
        Rx(-np.pi/2) | qubits[1]
    
        engine.flush()
    
        return
    
    def hydrogenUCCterm0b(self, parameter, qubits, engine):
        ''' Applies the UCC term exp(-i/2 $\theta$ Y3 Z2 X1) to the register.'''
    
        H | qubits[1]
        Rx(np.pi/2) | qubits[3]
        CNOT | (qubits[1], qubits[2])
        CNOT | (qubits[2], qubits[3])
    
        Rz(+2*parameter) | qubits[3]
    
        CNOT | (qubits[2], qubits[3])
        CNOT | (qubits[1], qubits[2])
        Rx(-np.pi/2) | qubits[3]
        H | qubits[1]
    
        engine.flush()
    
        return
    
    def hydrogenUCCterm1a(self, parameter, qubits, engine):
        ''' Applies the UCC term exp(i/2 $\theta$ X2 Z1 Y0) to the register.'''
    
        Rx(np.pi/2) | qubits[0]
        H | qubits[2]
        CNOT | (qubits[0], qubits[1])
        CNOT | (qubits[1], qubits[2])
    
        Rz(-2*parameter) | qubits[2]
    
        CNOT | (qubits[1], qubits[2])
        CNOT | (qubits[0], qubits[1])
        H | qubits[2]
        Rx(-np.pi/2) | qubits[0]
    
        engine.flush()
    
        return
    
    def hydrogenUCCterm1b(self, parameter, qubits, engine):
        ''' Applies the UCC term exp(-i/2 $\theta$ Y2 Z1 X0) to the register.'''
    
        H | qubits[0]
        Rx(np.pi/2) | qubits[2]
        CNOT | (qubits[0], qubits[1])
        CNOT | (qubits[1], qubits[2])
    
        Rz(+2*parameter) | qubits[2]
    
        CNOT | (qubits[1], qubits[2])
        CNOT | (qubits[0], qubits[1])
        Rx(-np.pi/2) | qubits[2]
        H | qubits[0]
        
        engine.flush()
    
        return
    
    
    def hydrogenUCCterm2(self, parameter, qubits, engine):
        ''' Applies the UCC term exp(-i/2 $\theta$ X3 X2 X1 Y0) to the register.'''
    
        Rx(np.pi/2) | qubits[0]
        H | qubits[1]
        H | qubits[2]
        H | qubits[3]
        CNOT | (qubits[0], qubits[1])
        CNOT | (qubits[1], qubits[2])
        CNOT | (qubits[2], qubits[3])
    
        Rz(2*parameter) | qubits[3]
    
        CNOT | (qubits[2], qubits[3])
        CNOT | (qubits[1], qubits[2])
        CNOT | (qubits[0], qubits[1])
        H | qubits[3]
        H | qubits[2]
        H | qubits[1]
        Rx(-np.pi/2) | qubits[0]
        
        engine.flush()
        
        return


if __name__ == "__main__":
    
    from projectq import MainEngine
    from projectq.ops import All, Measure
    
    from HamiltonianFile import HamiltonianFile
    from helper import printwf
    
    # TEST
    
    # get hamiltonian
    HFile = HamiltonianFile('hamiltonians/H2at075.txt')
    Hamiltonian, hamiltonian = HFile.Hamiltonian, HFile.hamiltonian
    
    # measure energy expectation value using ansatz
    engine = MainEngine()
    parameters = np.array([0.0, 0.0, 0.11483322638407246])
    qubits = engine.allocate_qureg(4)
    H2Ansatz(parameters, qubits, engine)# operators
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
    qubits = engine.allocate_qureg(4)
    engine.flush()# need to flush before accessing quantum results
    engine.backend.set_wavefunction(gs, qubits)
    engine.flush()
    energy = engine.backend.get_expectation_value(Hamiltonian, qubits)
    All(Measure) | qubits# to avoid deallocation error
    print('Expected ground state energy (eigenfunction expectation): ', energy)
    print('True ground state energy: ', eigs[0][0])