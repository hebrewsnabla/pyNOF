import sys
sys.path.append("..")
import nof
from automr import guess, dump_mat
from automr.autocas import get_uno
from pyscf import lib
lib.num_threads(4)

xyz = '''N 0.0 0.0 0.0; N 0.0 0.0 2.0'''
bas = 'def2-svp'
mf = guess.mix(xyz, bas, conv='tight')

mf, unos, noon, nacto, nelecact, ncore, _ = get_uno(mf, thresh=1.98)
#dump_mat.dump_mo(mf.mol, mf.mo_coeff)

thenof = nof.SOPNOF(mf, nacto, nelecact)
thenof.verbose = 5
thenof.mo_occ = noon / 2
thenof.fcisolver = nof.fakeFCISolver()
thenof.fcisolver.ncore = ncore
thenof.fcisolver.npair = nacto//2
#thenof.internal_rotation = True
#thenof.mc2step()
#dump_mat.dump_mo(thenof.mol, thenof.mo_coeff)

# API for some intermediate steps
# 1-step GVB without orbital optimization
eris = thenof.ao2mo(mf.mo_coeff)
thenof.casci(mf.mo_coeff, eris=eris)
# GVB energy evaluation without occ iteration
mo_occ = thenof.mo_occ
h1eff, e_core, eri_cas = thenof.h_casci(mf.mo_coeff, eris=eris)
thenof.fcisolver.nof.kernel(h1eff, eri_cas, mf.mo_coeff, mo_occ, iter_occ=False)