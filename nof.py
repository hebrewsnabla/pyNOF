from pyscf import lib, ao2mo, scf
import numpy as np
from automr.autocas import check_uno
from automr.dump_mat import dump_mo
from qnewton import qn_iter

einsum = lib.einsum

class PNOF():

    def __init__(self, _hf):
        self._hf = _hf
        self.mol = _hf.mol
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
        #exit()
        self.ncore = ncore
        self.npair = (nact - self.nopen)//2
        print('occ', self.mo_occ[:ncore+nact])
        Delta, Pi = get_DP(self.mo_occ, self.ncore, self.npair, self.nopen)
        h_mo = h1e(self._hf, mo)
        J,K = self.ao2mo()
        e0 = energy_elec(self.mo_occ, h_mo, J, K, Delta, Pi)
        e0 += self._hf.energy_nuc()
        print('PNOF5(GVB) energy %.6f' % e0)
        
        ao_ovlp = self.mol.intor_symmetric('int1e_ovlp')
        guess = np.ones(self.npair)*0.9
        guess = np.arange(0.99, 0.5, -0.48/(self.npair-1))
        new_occ = get_occ(self, mo, ao_ovlp, self.ncore, self.npair, self.nopen, guess,
                          h_mo, J, K)
        print('intrinsic occ', new_occ[:ncore+nact])

        get_grad(self, self.mo_occ, mo, h_mo, self.ncore, self.npair, self.nopen, Delta, Pi)
        
        
    def ao2mo(self):
        mo = self.mo_coeff
        #if self._scf._eri is not None:
        #    eri = ao2mo.full(self._scf._eri, mo_coeff,
        #                     max_memory=self.max_memory)
        #else:
        #    eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
        #                     max_memory=self.max_memory)
        #if self._hf._eri is None:
        _eri = self.mol.intor('int2e')
        #else:
        #    _eri = self._hf._eri
        #print(_eri.shape, mo.shape)
        #J = einsum('ijkl,ip,jp,kq,lq -> pq', _eri, mo, mo, mo, mo)
        #K = einsum('ijkl,ip,jq,kp,lq -> pq', _eri, mo, mo, mo, mo)
        J, K = _ao2mo(mo, mo, _eri)
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

def get_occ(nof, mo, ao_ovlp, ncore, npair, nopen, guess, h_mo, J, K):
    new_occ = np.zeros(mo.shape[-1])
    new_occ[:ncore] = 1.0
    #guess = np.ones(npair)*0.99
    #guess = np.array([0.995, 0.995, 0.981, 0.981, 0.713])**0.5
    guess = guess**0.5
    print('guess ci', guess)
    #guess = np.hstack((np.ones(ncore), guess))
    mo_act = mo[:, :ncore+2*npair+nopen]
    def update_occ(X, ncore, npair, nopen, nmo):
        occ = np.zeros(nmo)
        occ[:ncore] = 1.0
        occ[ncore:ncore+npair] = X**2
        occ[ncore+npair:ncore+2*npair] = np.flip(1.0 - X**2)
        return occ
    def get_grad(t):
        X = np.sin(t)
        return get_ci_grad(nof, X, mo_act, npair, ncore)*np.cos(t)
    def get_E(t):
        X = np.sin(t)
        occ = update_occ(X, ncore, npair, nopen, len(new_occ))
        Delta, Pi = get_DP(occ, ncore, npair, nopen)
        return energy_elec(occ, h_mo, J, K, Delta, Pi)

    new_t = qn_iter(np.arcsin(guess), get_grad, get_E)
    ci = np.sin(new_t)
    print('ci', ci)
    new_occ = update_occ(ci, ncore, npair, nopen, len(new_occ))
    
    return new_occ

def get_ci_grad(nof, guess, mo, npair, ncore):
    #print('ci', guess)
    mo_g = mo[:, ncore:ncore+npair]
    mo_u = np.flip(mo[:, ncore+npair:], axis=1) 
    mo_core = mo[:,:ncore]
    #dump_mo(nof.mol, mo_g)
    #dump_mo(nof.mol, mo_u)
    _eri = nof.mol.intor('int2e')
    Jgg, Kgg = _ao2mo(mo_g, mo_g, _eri)
    Juu, Kuu = _ao2mo(mo_u, mo_u, _eri)
    Jgu, Kgu = _ao2mo(mo_g, mo_u, _eri)
    Jgc, Kgc = _ao2mo(mo_g, mo_core, _eri)
    Juc, Kuc = _ao2mo(mo_u, mo_core, _eri)
    lam0 = h1e(nof._hf, mo_g).diagonal() - h1e(nof._hf, mo_u).diagonal() + einsum('ki->k', 2*Jgc - Kgc - 2*Juc + Kuc)
    lam0 += (Jgg.diagonal() - Juu.diagonal())*0.5
    def tri_k1(m):
        #return np.triu(m, k=1) + np.tril(m,k=-1)
        return m - np.diag(m.diagonal())
    uJgg, uKgg, uJuu, uKuu, uJgu, uKgu = map(tri_k1, (Jgg, Kgg, Juu, Kuu, Jgu, Kgu))
    #print(Jgg, '\n', Kgg, '\n', Juu, '\n', Kuu,  '\n',Jgu
    #, '\n', Kgu
    #)
    lam1 = einsum('i,ik->k', guess**2, (2*uJgg - uKgg -2*uJgu + uKgu)) 
    lam2 = einsum('i,ik->k', 1.0 - guess**2, (2*uJgu.T - uKgu.T -2*uJuu + uKuu))
    lam = lam0 + lam1 + lam2
    #print(1.0 - guess**2)
    #print((2*uJgu.T - uKgu.T -2*uJuu + uKuu))
    #print(lam0, lam1, lam2)
    lam *= 2.0
    tmp1 = 2*einsum('k,k->k', guess, lam) 
    tmp2 = 2*einsum('k,k,k->k', 1 - 2*guess**2, -1.0/np.sqrt(1.0 - guess**2), Kgu.diagonal())
    #print(1 - 2*guess**2, 1.0/np.sqrt(1.0 - guess**2), Kgu.diagonal())
    ci_grad = tmp1 + tmp2
    #print(tmp1, tmp2)
    #print('ci_grad', ci_grad)
    hdiag = 2*lam + 2*einsum('k,k,k,k->k', guess, 1.0/(1.0 - guess**2)**1.5, 1.0 + 2*(1.0 - guess**2), Kgu.diagonal())
    #print(lam)
    #print(hdiag)
    return ci_grad, hdiag

    

def _ao2mo(mo1, mo2, ao_eri):
    J = einsum('ijkl,ip,jp,kq,lq -> pq', ao_eri, mo1, mo1, mo2, mo2)
    K = einsum('ijkl,ip,jq,kp,lq -> pq', ao_eri, mo1, mo2, mo1, mo2)
    return J, K

def rotate_ci(ci, ):
    pass

def get_grad(nof, f, mo, h_mo, ncore, npair, nopen, Delta, Pi):
    occ = ncore + 2*npair + nopen
    mo_o = mo[:,:occ]
    mo_v = mo[:,occ:]
    a = 2*einsum('q,p->qp', f, f) - 2*Delta
    b = -einsum('q,p->qp', f, f) + Delta + Pi
    df = f[:occ, None] - f[None, :occ]
    a = a[:occ,:occ]
    b = b[:occ,:occ]
    da = np.expand_dims(a, axis=1) - np.expand_dims(a, axis=0)
    db = np.expand_dims(b, axis=1) - np.expand_dims(b, axis=0)
    print(df.shape, da.shape)
    _eri = nof.mol.intor('int2e')
    eri1 = ao2mo.general(_eri, (mo_o, mo_o, mo_o, mo_o), max_memory=nof.max_memory)
    print(eri1.shape)
    eri2 = ao2mo.general(_eri, (mo_o, mo_v, mo_o, mo_o), max_memory=nof.max_memory)
    goo = einsum('qp,pq->pq', df, h_mo[:occ,:occ]) + einsum('qpk,pqkk->pq', da, eri1) + einsum('qpk,pkqk->pq', db, eri1)
    gov = einsum('q,pq->pq', f[:occ], h_mo[occ:,:occ]) +  einsum('qk,qpkk->pq', a, eri2) + einsum('qk,kpkq->pq', b, eri2)
    #print(goo)
    #print(gov)

def gen_g_hop(mf, mo, mo_occ):
    pass