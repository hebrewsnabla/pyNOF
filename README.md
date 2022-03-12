# pyNOF

An implementation of Natural-orbital functional (a variation of RDMFT) based on PySCF. 

Currently only PNOF5 is supported and one subspace can only contain 2 orbitals, which is identical to GVB.

## Features
* GVB orbital optimization with augmented Hessian
* generate GVB init guess with 
  - UNO
  - paired LMO (MOKIT required)

## Limitations
nopen > 0 not supported.
