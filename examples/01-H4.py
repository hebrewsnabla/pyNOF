import sys
sys.path.append("..")
import nof
from automr import guess, dump_mat
from pyscf import mcscf, lib
lib.num_threads(4)

mf = guess.from_fch_noiter('h4init.fch')
dump_mat.dump_mo(mf.mol, mf.mo_coeff)

thenof = nof.SOPNOF(mf, 4, 4)
thenof.verbose = 5
thenof.fcisolver = nof.fakeFCISolver()
thenof.fcisolver.ncore = 0
thenof.fcisolver.npair = 2
#thenof.fcisolver.sorting = 'gau'
thenof.internal_rotation = True
thenof.mc2step()
dump_mat.dump_mo(thenof.mol, thenof.mo_coeff)







