import sys
sys.path.append("..")
import nof
from automr import guess, dump_mat
from automr.autocas import get_locorb
from pyscf import lib
lib.num_threads(4)

xyz = '''N 0.0 0.0 0.0; N 0.0 0.0 1.1'''
bas = 'def2-svp'
mf = guess.gen(xyz, bas, 0, 0).run().to_rhf()

#mf, unos, noon, nacto, nelecact, ncore, _ = get_uno(mf, thresh=1.99999)

mf, lmos, npair, ndb = get_locorb(mf)
dump_mat.dump_mo(mf.mol, mf.mo_coeff)

thenof = nof.SOPNOF(mf, npair*2, npair*2)
thenof.verbose = 5
#thenof.mo_occ = noon / 2
thenof.fcisolver = nof.fakeFCISolver()
thenof.fcisolver.ncore = ndb
thenof.fcisolver.npair = npair
thenof.internal_rotation = True
thenof.mc2step()
dump_mat.dump_mo(thenof.mol, thenof.mo_coeff)







