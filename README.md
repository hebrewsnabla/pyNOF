# pyNOF

An implementation of Natural-orbital functional (a variation of RDMFT) based on PySCF. 

Currently only PNOF5 is supported and one subspace can only contain 2 orbitals, which is identical to GVB.

## Features
* GVB orbital optimization with augmented Hessian
* generate GVB init guess with 
  - UNO
  - paired LMO (MOKIT required)

## Usage
```
git clone git@github.com:hebrewsnabla/pyAutoMR.git
git clone git@github.com:hebrewsnabla/pyNOF.git
```
Add `/path/to/pyAutoMR` and `/path/to/pyNOF` to your PYTHONPATH. 

## Limitations
* nopen > 0 not supported.
