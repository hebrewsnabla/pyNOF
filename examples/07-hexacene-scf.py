import sys
sys.path.append("..")
import nof
from automr import guess, dump_mat
from automr.autocas import get_uno, loc_asrot
from pyscf import lib, gto
from pyscf.scf import chkfile

lib.num_threads(4)

mol = gto.Mole(
atom = 'hexacene.xyz',
basis = '3-21g',
max_memory=2000,
verbose=4).build()
mf = guess._mix(mol, conv='tight', level_shift=0.25, newton=True)
chkfile.dump_scf(mol, '07-hexacene-scf.pchk', mf.e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ)

exit()
mf, unos, noon, nacto, nelecact, ncore, _ = get_uno(mf, thresh=1.98)
mf = loc_asrot(mf, nacto, nelecact, ncore)
#dump_mat.dump_mo(mf.mol, mf.mo_coeff)

#exit()
thenof = nof.SOPNOF(mf, nacto, nelecact).density_fit()
thenof.verbose = 4
thenof.mo_occ = noon / 2
thenof.fcisolver = nof.fakeFCISolver()
thenof.fcisolver.ncore = ncore
thenof.fcisolver.npair = nacto//2
#thenof.fcisolver.guess_scal = 1.088
#thenof.fcisolver.with_df = True
thenof.fcisolver.gvbci_maxiter = 60
thenof.internal_rotation = True # important for this case !
thenof.max_cycle_macro = 30
thenof.max_stepsize = 0.05
thenof.mc2step()
#dump_mat.dump_mo(thenof.mol, thenof.mo_coeff)





