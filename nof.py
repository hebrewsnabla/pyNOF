from pyscf import lib, ao2mo


einsum = lib.einsum

class PNOF():

    def __init__(self, _hf):
        self.mo_coeff = _hf.mo_coeff
        self.mo_occ = _hf.mo_occ
        self.max_memory = 2000

    def kernel(self):
        mo = self.mo_coeff
        h_mo = h1e(self._hf, mo)

    def ao2mo(self):
        mo_coeff = self.mo_coeff
        if self._scf._eri is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri

def energy_elec(mo_occ, h_mo, vj_mo, vk_mo, Delta, Pi):
    E1 = 2*np.dot(mo_occ, h_mo.diagonal())
    D2 = einsum('q,p->qp', mo_occ, mo_occ) - Delta
    E2 = np.dot(Pi, vk_mo) + np.dot(D2, 2*vj_mo - vk_mo)
    return E1 + E2

def h1e(hf, mo):
    hcore = hf.get_hcore()
    h_mo = einsum('ji,jk,kl->il', mo, hcore, mo)
    return h_mo

