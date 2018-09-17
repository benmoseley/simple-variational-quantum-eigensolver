#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:28:13 2018

@author: bmoseley
"""
import numpy as np
from projectq.ops import All, Measure



def measureAllQubits(qubits, engine):
    ''' Measures all qubits such that it collapses to
        a single state in the computational basis.
    '''
    All(Measure) | qubits
    engine.flush()# need to flush before accessing quantum results
    result = np.array([int(qubit) for qubit in qubits])
    return result


def printwf(engine):
    '''Pretty prints the wavefunction.
    Engine can be projectq engine or numpy array of amplitudes'''
    
    if ("engine" in str(type(engine))) and ("projectq" in str(type(engine))):
        engine.flush()# need to flush before accessing quantum results
        qubit_dict, amplitudes = engine.backend.cheat()# get wavefunction amplitudes
        n_qubits = len(qubit_dict)
    else: # assume numpy array
        n_qubits = int(np.log2(len(engine)))
        amplitudes = engine
    
    for i in range(len(amplitudes)):
        amplitude = amplitudes[i]
        if abs(amplitude) < 1E-10: amplitude = 0
        print('|',('{:0'+str(n_qubits)+'d}').format(int(bin(i)[2:])),'>',
              ':', "{c.real:.2f} {c.imag:+.2f}j".format(c=amplitude),
              ":","%.2f"%(abs(amplitude)**2.))
    return


def isHermitian(a, tol=1e-8):
    "check if numpy matrix a is hermitian"
    return np.allclose(a, a.conj().T, atol=tol)


if __name__ == "__main__":
    
    
    ## TESTS
    
    from projectq import MainEngine
    from projectq.ops import H, Z, CNOT
    
    nqubits = 2
    engine = MainEngine()# define engine
    qubits = engine.allocate_qureg(nqubits)# circuit function
    H | qubits[0]
    Z | qubits[0]
    CNOT | (qubits[0],qubits[1])
    engine.flush()
    
    printwf(engine)
    print()
    printwf(np.arange(2**4))
    
    res = measureAllQubits(qubits, engine)
    print("result: ", res)

    
    
    

    
    