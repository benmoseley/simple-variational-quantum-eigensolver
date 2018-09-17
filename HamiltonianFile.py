#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:47:55 2018

@author: bmoseley
"""

from projectq.ops import QubitOperator
import numpy as np
from helper import isHermitian

class HamiltonianFile:
    '''Get Hamiltonian operator from text file
    
    Expects Hamiltonian which are expressed as a linear combination of tensor products of Pauli operators.
    
    Expects each line in text file to specify each term in the linear combination, with the form
    
    distance term_amplitude qubit1_pauli_operator qubit2_pauli_operator ... qubitN_pauli_operator
    
    where the qubit{i}_pauli_operator indicies encode Pauli gates in the following way:
        0: I
        1: X
        2: Y
        3: Z
    '''
    
    def __init__(self, filename, verbose=False):
        self.verbose = verbose
        self.hamiltonian = self._get_hamiltonian(filename)
        self.Hamiltonian = self._get_Hamiltonian(filename)
    
    def _get_hamiltonian(self, filename):
        'Parse a Hamiltonian operator from file'
        
        # get raw data
        with open(filename,"r", encoding='utf-8-sig') as file:
            lines=file.readlines()
        
        # convert line element types
        h = []
        for line in lines:
            line = line.split()
            for i in range(len(line)):
                if i in [0,1]:
                    line[i] = float(line[i])
                elif line[i] == str(0):
                    line[i] = "I"
                elif line[i] == str(1):
                    line[i] = "X"
                elif line[i] == str(2):
                    line[i] = "Y"
                elif line[i] == str(3):
                    line[i] = "Z"
                else:
                    raise Exception("Error: Incorrect hamiltonian file format")
            h.append(line)
            
            if self.verbose: print(line)
            
        return h

    def _get_Hamiltonian(self, filename):
        'Parse a projectq Hamiltonian operator from file'

        # get raw data
        with open(filename,"r", encoding='utf-8-sig') as file:
            lines=file.readlines()
        
        # convert to projectq operator
        H = QubitOperator()
        for line in lines:
            line = line.split()
            term = ''
            for i in range(len(line)):
                if i in [0,1]:
                    pass
                elif line[i] == str(0):
                    pass
                elif line[i] == str(1):
                    term += 'X' + str(i-2) + ' '
                elif line[i] == str(2):
                    term += 'Y' + str(i-2) + ' '
                elif line[i] == str(3):
                    term += 'Z' + str(i-2) + ' '
                else:
                    raise Exception("Error: Incorrect hamiltonian file format")
                    
            if self.verbose: print(term)
            H += float(line[1])*QubitOperator(term)
    
        return H
    
    def getMatrix(self, hamiltonian):
        "Get the matrix representation of a hamiltonian, using Kronecker products"
        
        n_qubits = len(hamiltonian[0])-2
        i = complex(0,1)
        
        I = np.array([[1, 0],
                      [0, 1]], dtype=complex)
    
        X = np.array([[0, 1],
                      [1, 0]], dtype=complex)
        
        Y = np.array([[0, -i],
                      [i, 0]], dtype=complex)
        
        Z = np.array([[1, 0],
                      [0, -1]], dtype=complex)
        gates = {"I": I, "X": X, "Y": Y, "Z": Z}
        
        # get kronecker product of all terms in hamiltonian
        Hm = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        for hamiltonian_term in hamiltonian:
            term = gates[hamiltonian_term[-1]]
            for gate_str in hamiltonian_term[-2:1:-1]:
                gate = gates[gate_str]
                term = np.kron(term, gate)
            #print(term)
            #print()
            Hm += hamiltonian_term[1]*np.copy(term)
            
        # check it is hermitian
        if not isHermitian(Hm):
            raise Exception("Error: supplied hamiltonian file is not Hermitian")
        
        if self.verbose:
            np.set_printoptions(linewidth=200, precision=1)
            print(Hm)
            np.set_printoptions(linewidth=75, precision=8)
        
        # compute eigenvalues
        if self.verbose: print("Computing eigenvalues..")
        #eigs = np.linalg.eig(Hm)
        eigs = np.linalg.eigh(Hm)# routine for hermitian matrices. Returns eigenvalues in ascending order (all real)
        if self.verbose:
            for i in range(len(eigs[0])):
                print(sorted(eigs[0])[i].real)
                
        return Hm, eigs
        
        
if __name__ == "__main__":
    
    HFile = HamiltonianFile('hamiltonians/H2at075.txt', verbose=True)
    print(HFile.Hamiltonian)
    print(len(HFile.hamiltonian))
    
    HFile.getMatrix(HFile.hamiltonian)
    
    HFile = HamiltonianFile('hamiltonians/LiHat145.txt', verbose=True)
    print(HFile.Hamiltonian)
    print(len(HFile.hamiltonian))
    
    Hm, eigs = HFile.getMatrix(HFile.hamiltonian)