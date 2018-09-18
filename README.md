# simple-variational-quantum-eigensolver
A simple variational eigensolver implemented using ProjectQ and Bayesian optimisation.

## Description
A set of python scripts for classically simulating a simple variational eigensolver [[Peruzzo et al. 2014](https://www.nature.com/articles/ncomms5213), [McClean et al. 2016](http://iopscience.iop.org/article/10.1088/1367-2630/18/2/023023/meta)].

Requires a text file specifying the Hamiltonian to use as input.
Also requires an Ansatz circuit object to be defined.

Includes examples for finding the ground state energy of LiH and H2.

Bayesian optimisation and the [SPSA](http://www.jhuapl.edu/SPSA/) optimisation algorithm are used for optimisation.

## Requirements

[projectq](https://github.com/ProjectQ-Framework/ProjectQ) - for quantum simulation

[GPyOpt](https://github.com/SheffieldML/GPyOpt) - for Bayesian optimisation

numpy

matplotlib

`pip install projectq, gpyopt, numpy, matplotlib`

## Example

Convergence of ground state energy for LiH using Bayesian optimisation:

<img src="example_results.png"  width="500" />
