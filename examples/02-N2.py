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
dump_mat.dump_mo(mf.mol, mf.mo_coeff)

thenof = nof.SOPNOF(mf, nacto, nelecact)
thenof.verbose = 5
thenof.fcisolver = nof.fakeFCISolver()
thenof.fcisolver.ncore = ncore
thenof.fcisolver.npair = nacto//2
thenof.internal_rotation = True
thenof.mc2step()
dump_mat.dump_mo(thenof.mol, thenof.mo_coeff)







