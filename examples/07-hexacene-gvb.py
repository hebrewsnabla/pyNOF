import sys
sys.path.append("..")
import nof
from automr import guess, dump_mat
from automr.autocas import get_uno, loc_asrot
from pyscf import lib, gto
from pyscf.scf import chkfile
import numpy as np

lib.num_threads(4)

mol, mf_dict = chkfile.load_scf('07-hexacene-scf.pchk')
mf = mol.UHF()
mf.mo_coeff = mf_dict['mo_coeff']
mf.mo_occ = mf_dict['mo_occ']

mf, unos, noon, nacto, nelecact, ncore, _ = get_uno(mf, thresh=1.98)
mf, npair = loc_asrot(mf, nacto, nelecact, ncore)
"""fock_ao = mf.get_fock()
mo_e = lib.einsum('mi, mn, ni -> i', mf.mo_coeff, fock_ao, mf.mo_coeff)
print(mo_e[ncore:ncore+nacto])

def obtain_occ(e, u, k):
    return 1 / (1 + np.exp((e - u) / k))
mo_e_act = mo_e[ncore:ncore+nacto]
ft_occ = obtain_occ(mo_e_act, np.average(mo_e_act), 0.15)
occ = np.zeros(len(mf.mo_occ))
occ[:ncore] = 1
occ[ncore:ncore+nacto] = ft_occ
print(ft_occ)
#exit()"""

thenof = nof.SOPNOF(mf, nacto, nelecact).density_fit()
thenof.verbose = 4
#thenof.mo_occ = occ       # you can provide init occ like np.array([2.0, 2.0, ..., 2.0, 
                           #                                         1.9, 1.9, 1.01, 0.99, 0.1, 0.1, 
                           #                                                      0.0, ..., 0.0])
                           # Attention: the length of occ array is nmo, not nacto
thenof.fcisolver = nof.fakeFCISolver()
thenof.fcisolver.ncore = ncore
thenof.fcisolver.npair = nacto//2
thenof.fcisolver.guess_scal = 0.9 # default init occ is 0.9 for each 1st-NO, you can scale it by guess_scal
#thenof.fcisolver.with_df = True
thenof.fcisolver.gvbci_maxiter = 60
#thenof.fcisolver.gvbci_maxstep = 0.08
thenof.fcisolver.optimizer = 'bfgs' # default optimizer is 'ms' (Murtagh-Sargent), 
                                    # you can switch it to 'bfgs'/'l-bfgs-b'/'newton-cg'/'trust-ncg' provided by scipy
                                    # we may add more optimizers later
thenof.internal_rotation = True # important for this case !
thenof.max_cycle_macro = 10
thenof.max_stepsize = 0.05
thenof.mc2step()
#dump_mat.dump_mo(thenof.mol, thenof.mo_coeff)





