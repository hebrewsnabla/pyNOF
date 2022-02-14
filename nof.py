from pyscf import lib, ao2mo, scf
import numpy as np
from automr.autocas import check_uno

einsum = lib.einsum

class PNOF():

    def __init__(self, _hf):
        self._hf = _hf
        self.mo_coeff = _hf.mo_coeff
        self.mo_occ = _hf.mo_occ / 2.0
        self.max_memory = 2000

        self.ncore = None
        self.npair = None
        if isinstance(_hf, scf.hf.RHF):
            self.nopen = 0
        else:
            self.nopen = _hf.spin
        self.nmo = None
        print('\n******** %s ********' % self.__class__)

    def kernel(self):
        mo = self.mo_coeff
        #self.nopen = 
        nact, ncore, nex = check_uno(self.mo_occ*2, 1.99999)
        print('ncore %d nact %d nex %d' % (ncore, nact, nex))
        self.ncore = ncore
        self.npair = (nact - self.nopen)//2
        Delta, Pi = get_DP(self.mo_occ, self.ncore, self.npair, self.nopen)
        h_mo = h1e(self._hf, mo)
        J,K = self.ao2mo()
        e0 = energy_elec(self.mo_occ, h_mo, J, K, Delta, Pi)
        e0 += self._hf.energy_nuc()
        print('PNOF5(GVB) energy %.6f' % e0)

    def ao2mo(self):
        mo = self.mo_coeff
        #if self._scf._eri is not None:
        #    eri = ao2mo.full(self._scf._eri, mo_coeff,
        #                     max_memory=self.max_memory)
        #else:
        #    eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
        #                     max_memory=self.max_memory)
        #if self._hf._eri is None:
        _eri = self._hf.mol.intor('int2e')
        #else:
        #    _eri = self._hf._eri
        print(_eri.shape, mo.shape)
        J = einsum('ijkl,ip,jp,kq,lq -> pq', _eri, mo, mo, mo, mo)
        K = einsum('ijkl,ip,jq,kp,lq -> pq', _eri, mo, mo, mo, mo)
        return J, K

def energy_elec(mo_occ, h_mo, J, K, Delta, Pi):
    E1 = 2*np.dot(mo_occ, h_mo.diagonal())
    D2 = einsum('q,p->qp', mo_occ, mo_occ) - Delta
    E2 = einsum('qp, pq', Pi, K) + einsum('qp,pq', D2, 2*J - K)
    return E1 + E2

def h1e(hf, mo):
    hcore = hf.get_hcore()
    h_mo = einsum('ji,jk,kl->il', mo, hcore, mo)
    return h_mo

def get_DP(f, ncore, npair, nopen):
    nmo = len(f)
    Pi = np.zeros((nmo,nmo))
    Delta = np.zeros((nmo,nmo))
    for i in range(nmo):
        Pi[i,i] = f[i]
        Delta[i,i] = f[i]**2
    for i in range(npair):
        upper = ncore+2*npair+nopen-1
        Pi[ncore+i, upper-i] = -np.sqrt(f[ncore+i]*f[upper-i])
        #print(ncore+i, upper-i, f[ncore+i]*f[upper-i])
        Delta[ncore+i, upper-i] = f[ncore+i]*f[upper-i]
    Pi += np.triu(Pi,1).T
    Delta += np.triu(Delta,1).T
    #print(Delta[:15,:15])
    #print(Pi[:15,:15])
    return Delta, Pi
