from pyscf import lib, ao2mo, scf, gto, df
from pyscf.lib import logger
#from pyscf.mcscf.mc1step import expmat, CASSCF
from pyscf.mcscf import mc1step, casci
from pyscf.soscf import ciah
from pyscf.df.addons import DEFAULT_AUXBASIS
import numpy as np
import scipy
from automr.autocas import check_uno
from automr.dump_mat import dump_mo
from qnewton import qn_iter
from timing import timing

einsum = lib.einsum

class PNOF():

    def __init__(self, _hf):
        self._hf = _hf
        self.mol = _hf.mol
        self.mo_coeff = _hf.mo_coeff
        self.mo_occ = None #_hf.mo_occ / 2.0
        self.max_memory = 2000

        self.ncore = None
        self.npair = None
        if isinstance(_hf, scf.hf.RHF):
            self.nopen = 0
        else:
            self.nopen = _hf.spin
        self.nmo = None
        self.with_df = False
        self.gvbci_maxstep = 0.05
        self.gvbci_maxiter = 35
        self.optimizer = 'ms'
        print('\n******** %s ********' % self.__class__)

    def kernel(self, h1eff, eri_cas, mo=None, mo_occ=None):
        if mo is None:
            mo = self.mo_coeff
        else:
            self.mo_coeff = mo
        if mo_occ is not None:
            self.mo_occ = mo_occ / 2.0
        #if self.sorting == 'gau':
        #    mo = rearrange(mo, self.ncore, self.npair)
        #dump_mo(self.mol, mo[:,:12])
        #self.nopen = 
        nmo = mo.shape[-1]
        if self.npair is None:
            nact, ncore, nex = check_uno(self.mo_occ*2, 1.99999)
            print('ncore %d nact %d nex %d' % (ncore, nact, nex))
            #exit()
            self.ncore = ncore
            self.npair = (nact - self.nopen)//2
        else:
            ncore = self.ncore
            npair = self.npair
            nact = npair*2
        #dump_mo(self.mol, mo[:,ncore:ncore+nact], ncol=10)
        #h_mo = h1e(self._hf, mo)
        h_mo = h1eff
        #J,K = self.ao2mo(mo)
        J,K = self.eri2jk(eri_cas)
        if self.mo_occ is None:
            print('using default guess')
            guess = np.ones(self.npair)*0.9
            #guess = np.arange(0.96, 0.55, -0.4/(self.npair-1))
            #guess = (np.arctan(np.arange(63.1, 0.0, -63/(self.npair-1)))+1)/2
            if self.guess_scal is not None:
                guess *= self.guess_scal
        else:
            print('using input mo_occ for guess')
            guess = self.mo_occ[ncore:ncore+npair]
        new_occ = get_occ(self, mo, self.ncore, self.npair, self.nopen, guess,
                          h_mo, J, K)
        print('intrinsic occ', new_occ[:ncore+nact])
        self.mo_occ = new_occ
        #print('occ', self.mo_occ[:ncore+nact])
        Delta, Pi = get_DP(self.mo_occ, self.ncore, self.npair, self.nopen)
        
        e_elec = energy_elec(self.mo_occ, ncore, nact, h_mo, J, K, Delta, Pi)
        e0 = e_elec 
        print('PNOF5(GVB) elec energy %.6f' % e0)
        
        
        self.Delta  = Delta
        self.Pi = Pi
        return e0, e_elec
        
    def ao2mo(self, mo=None):
        if mo is None:
            mo = self.mo_coeff
        #if self._scf._eri is not None:
        #    eri = ao2mo.full(self._scf._eri, mo_coeff,
        #                     max_memory=self.max_memory)
        #else:
        #    eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
        #                     max_memory=self.max_memory)
        #if self._hf._eri is None:
        M = mo.shape[0]
        if self.with_df:
            J, K = _ao2mo_df(mo, mo, self.mol)
        else:
            if self._is_mem_enough():
                _eri = self.mol.intor('int2e')
                J, K = _ao2mo_incore(mo, mo, _eri)
            else:
                #J, K = _ao2mo_sdirect(mo, mo, self.mol)
                J, K = _ao2mo_sdirect_s4(mo, mo, self.mol)
        print('current mem %d MB' % lib.current_memory()[0])
        return J, K

    def eri2jk(self, eri_cas):
        ni, nj, nk, nl = eri_cas.shape
        #assert np == nq and na == nb
        J = np.zeros((ni, nk))
        K = np.zeros((ni, nk))
        for i in range(ni):
            for j in range(nk):
                J[i,j] = eri_cas[i,i,j,j]
                K[i,j] = eri_cas[i,j,j,i]
        return J,K

    def _is_mem_enough(self):
        nbf = self.mol.nao_nr()
        return 8*nbf**4/1e6+lib.current_memory()[0] < self.max_memory*.95

def energy_elec(mo_occ, ncore, nact, h_mo, J, K, Delta, Pi):
    mo_occ = mo_occ[ncore:ncore+nact]
    Delta = Delta[ncore:ncore+nact,ncore:ncore+nact]
    Pi = Pi[ncore:ncore+nact,ncore:ncore+nact]
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

@timing
def get_occ(nof, mo, ncore, npair, nopen, guess, h_mo, J, K):
    maxstep = nof.gvbci_maxstep
    maxiter = nof.gvbci_maxiter
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
        X = t2X(t)
        f, hdiag = get_ci_grad(nof, X, mo_act, npair, 0, h_mo, J, K)
        return f*np.sin(t)*np.cos(t), hdiag*(np.sin(t)*np.cos(t))**2
    def get_E(t):
        X = t2X(t)
        occ = update_occ(X, ncore, npair, nopen, len(new_occ))
        Delta, Pi = get_DP(occ, ncore, npair, nopen)
        nact = npair*2+nopen
        return energy_elec(occ, ncore, nact, h_mo, J, K, Delta, Pi)
    
    if nof.optimizer == 'ms':
        new_t, conv = qn_iter(X2t(guess), get_grad, get_E, t2X, X2t, maxstep, maxiter)
    elif nof.optimizer in ['bfgs', 'l-bfgs-b']:
        res = scipy.optimize.minimize(get_E, X2t(guess), method=nof.optimizer,
                jac=lambda t: get_grad(t)[0], 
                #hess=lambda t: np.diag(get_grad(t)[1]),
                tol=1e-8,
                options={'disp': True, 'gtol':1e-5, 'maxiter':maxiter})
        new_t = res.x
        conv = True
    elif nof.optimizer in ['newton-cg', 'trust-ncg']:
        res = scipy.optimize.minimize(get_E, X2t(guess), method=nof.optimizer,
                jac=lambda t: get_grad(t)[0], 
                hess=lambda t: np.diag(get_grad(t)[1]),
                tol=1e-8,
                options={'disp': True})
        new_t = res.x
        conv = True
    else:
        raise ValueError(' optimizer unsupported')
    ci = t2X(new_t)
    print('ci', ci)
    new_occ = update_occ(ci, ncore, npair, nopen, len(new_occ))
    
    if not conv:
        raise RuntimeError('GVB ci fails to converge')
    return new_occ

def t2X(t):
    return 0.5+0.5*np.sin(t)**2

def X2t(X):
    return np.arcsin((X*2-1)**0.5)

def get_ci_grad(nof, guess, mo, npair, ncore, h1eff, J, K):
    #print('ci', guess)
    mo_g = mo[:, ncore:ncore+npair]
    mo_u = np.flip(mo[:, ncore+npair:], axis=1) 
    mo_core = mo[:,:ncore]
    """
    #dump_mo(nof.mol, mo_g)
    #dump_mo(nof.mol, mo_u)
    _eri = nof.mol.intor('int2e')
    Jgg, Kgg = _ao2mo(mo_g, mo_g, _eri)
    Juu, Kuu = _ao2mo(mo_u, mo_u, _eri)
    Jgu, Kgu = _ao2mo(mo_g, mo_u, _eri)
    Jgc, Kgc = _ao2mo(mo_g, mo_core, _eri)
    Juc, Kuc = _ao2mo(mo_u, mo_core, _eri)
    """
    g_slice = slice(ncore, ncore+npair)
    u_slice = slice(ncore+2*npair-1, ncore+npair-1,-1)
    c_slice = slice(0, ncore)
    Jgg = J[g_slice, g_slice]
    Juu = J[u_slice, u_slice]
    Jgu = J[g_slice, u_slice]
    Jgc = J[g_slice, c_slice]
    Juc = J[u_slice, c_slice]
    Kgg = K[g_slice, g_slice]
    Kuu = K[u_slice, u_slice]
    Kgu = K[g_slice, u_slice]
    Kgc = K[g_slice, c_slice]
    Kuc = K[u_slice, c_slice]
    #lam0 = h1e(nof._hf, mo_g).diagonal() - h1e(nof._hf, mo_u).diagonal() + einsum('ki->k', 2*Jgc - Kgc - 2*Juc + Kuc)
    lam0 = h1eff[g_slice,g_slice].diagonal() - h1eff[u_slice,u_slice].diagonal()
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

@timing    
def _ao2mo_df(mo1, mo2, mol):
    auxmol = mol.copy()
    auxmol.basis = DEFAULT_AUXBASIS[mol.basis][1]
    auxmol.build()
    nao = mol.nao
    nao_df = auxmol.nao    
    int2c2e = auxmol.intor("int2c2e")
    int2c2e.shape
    int3c2e = df.incore.aux_e2(mol, auxmol)
    int2c2e_half = scipy.linalg.cholesky(int2c2e, lower=True)
    V_df = scipy.linalg.solve_triangular(int2c2e_half, int3c2e.reshape(-1, nao_df).T, lower=True)\
               .reshape(nao_df, nao, nao).transpose((1, 2, 0))
    V_df_pp = einsum("uvP, up, vp -> pP", V_df, mo1, mo2)
    V_df_pq = einsum("uvP, up, vq -> pqP", V_df, mo1, mo2)
    J = einsum("pP,qP->pq", V_df_pp, V_df_pp)
    K = einsum("pqP,pqP->pq", V_df_pq, V_df_pq)
    return J,K

@timing    
def _ao2mo_incore(mo1, mo2, ao_eri):
    J = einsum('ijkl,ip,jp,kq,lq -> pq', ao_eri, mo1, mo1, mo2, mo2)
    K = einsum('ijkl,ip,jq,kp,lq -> pq', ao_eri, mo1, mo2, mo1, mo2)
    return J, K

@timing    
def _ao2mo_sdirect(mo1, mo2, mol):
    nbas = mol.nbas
    J = np.zeros((mo1.shape[-1], mo2.shape[-1]))
    K = np.zeros((mo1.shape[-1], mo2.shape[-1]))
    for i in range(nbas):
        g = mol.intor('int2e', shls_slice=(i, i+1, 0, nbas, 0, nbas, 0, nbas))
        id1, id2 = gto.nao_nr_range(mol,i,i+1)
        ao_slice = slice(id1,id2)
        #print(ao_slice, g.shape)
        J += einsum('ijkl,ip,jp,kq,lq -> pq', g, mo1[ao_slice,:], mo1, mo2, mo2)
        K += einsum('ijkl,ip,jq,kp,lq -> pq', g, mo1[ao_slice,:], mo2, mo1, mo2)
    return J, K

@timing    
def _ao2mo_sdirect_s4(mo1, mo2, mol):
    nbas = mol.nbas
    J = np.zeros((mo1.shape[-1], mo2.shape[-1]))
    K = np.zeros((mo1.shape[-1], mo2.shape[-1]))
    for i in range(nbas):
        id1, id2 = gto.nao_nr_range(mol,i,i+1)
        ao_slice_i = slice(id1,id2)
        g_jkl = mol.intor('int2e', shls_slice=(i, i+1, 0, nbas, 0, nbas, 0, nbas), aosym='s2kl')
        for j in range(i):
            #g = mol.intor('int2e', shls_slice=(i, i+1, j, j+1, 0, nbas, 0, nbas), aosym='s2kl')
            #print(g.shape)
            jd1, jd2 = gto.nao_nr_range(mol,j,j+1)
            ao_slice_j = slice(jd1,jd2)
            g = g_jkl[:,ao_slice_j,:]
            g = kl2full(g)
            #print(ao_slice, g.shape)
            J += 2*einsum('ijkl,ip,jp,kq,lq -> pq', g, mo1[ao_slice_i,:], mo1[ao_slice_j,:], mo2, mo2)
            tmp = einsum('ijkl,kp,lq -> ijpq', g, mo1, mo2)
            K += einsum('ijpq, ip,jq -> pq', tmp, mo1[ao_slice_i,:], mo2[ao_slice_j,:])
            K += einsum('ijpq, iq,jp -> pq', tmp, mo1[ao_slice_i,:], mo2[ao_slice_j,:])
        #g = mol.intor('int2e', shls_slice=(i, i+1, i, i+1, 0, nbas, 0, nbas), aosym='s4')
        g = g_jkl[:,ao_slice_i,:]
        g = kl2full(g)
        J += einsum('ijkl,ip,jp,kq,lq -> pq', g, mo1[ao_slice_i,:], mo1[ao_slice_i,:], mo2, mo2)
        K += einsum('ijkl,ip,jq,kp,lq -> pq', g, mo1[ao_slice_i,:], mo2[ao_slice_i,:], mo1, mo2)
    return J, K

@timing    
def _ao2mo_direct_s4(mo1, mo2, mol):
    nbas = mol.nbas
    J = np.zeros((mo1.shape[-1], mo2.shape[-1]))
    K = np.zeros((mo1.shape[-1], mo2.shape[-1]))
    for i in range(nbas):
        id1, id2 = gto.nao_nr_range(mol,i,i+1)
        ao_slice_i = slice(id1,id2)
        for j in range(i):
            g = mol.intor('int2e', shls_slice=(i, i+1, j, j+1, 0, nbas, 0, nbas), aosym='s2kl')
            #print(g.shape)
            g = kl2full(g)
            jd1, jd2 = gto.nao_nr_range(mol,j,j+1)
            ao_slice_j = slice(jd1,jd2)
            #print(ao_slice, g.shape)
            J += 2*einsum('ijkl,ip,jp,kq,lq -> pq', g, mo1[ao_slice_i,:], mo1[ao_slice_j,:], mo2, mo2)
            tmp = einsum('ijkl,kp,lq -> ijpq', g, mo1, mo2)
            K += einsum('ijpq, ip,jq -> pq', tmp, mo1[ao_slice_i,:], mo2[ao_slice_j,:])
            K += einsum('ijpq, iq,jp -> pq', tmp, mo1[ao_slice_i,:], mo2[ao_slice_j,:])
        g = mol.intor('int2e', shls_slice=(i, i+1, i, i+1, 0, nbas, 0, nbas), aosym='s4')
        g = tril2full(g)
        J += einsum('ijkl,ip,jp,kq,lq -> pq', g, mo1[ao_slice_i,:], mo1[ao_slice_i,:], mo2, mo2)
        K += einsum('ijkl,ip,jq,kp,lq -> pq', g, mo1[ao_slice_i,:], mo2[ao_slice_i,:], mo1, mo2)
    return J, K

def kl2full(g):
    ni,nj,nkl = g.shape
    nk = int(np.floor(np.sqrt(nkl*2)))
    g_new = np.zeros((ni,nj,nk,nk))
    for i in range(ni):
        for j in range(nj):
            g_new[i,j] = tofull(g[i,j], nk)
    return g_new
    
def tofull(g, nk):
    tmp = np.zeros((nk,nk))
    tmp[np.tril_indices(nk)] = g
    tmp += np.tril(tmp,-1).T
    #tmp[np.diag_indices(nk)] /= 2
    return tmp
    

def tril2full(g):
    #print(g.shape)
    n1, n2 = g.shape
    if n1==1:
        m2 = int(np.floor(np.sqrt(n2*2)))
        tmp = np.zeros((1,1,m2,m2))
        tmp[0,0] = tofull(g[0], m2) 
        return tmp
    m1 = int(np.floor(np.sqrt(n1*2)))
    m2 = int(np.floor(np.sqrt(n2*2)))
    g_new = np.zeros((m1,m1,m2,m2))
    for i in range(m1):
        for j in range(i+1):
            tmp = np.zeros((m2,m2))
            tmp[np.tril_indices(m2)] = g[i*(i+1)//2+j]
            tmp += np.tril(tmp,-1).T
            #tmp[np.diag_indices(m2)] /= 2
            g_new[i,j] = tmp
            if i != j: g_new[j,i] = tmp
    return g_new

#def rotate_ci(ci, ):
#    pass

def get_grad(nof, f, mo, h_mo, ncore, npair, nopen, Delta, Pi, J, K):
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
    print(goo)
    print(gov)
    #g = np.hstack((goo,gov))
    #Joo = J[:occ,:occ]
    #Koo = K[:occ,:occ]
    #Jvo = J[occ:,:occ]
    #Kvo = K[occ:,:occ]
    #hdiag = einsum('i,j->ij', f[:occ], h_mo[occ:,occ:].diagonal()) + einsum('ik,jk->ij', a, Jvo) + einsum('ik,jk->ij', b, Kvo)
    #hdiag -= einsum('i,i->i', f[:occ], h_mo[:occ,:occ].diagonal()) + einsum('ik,ik->i', a, Joo) + einsum('ik,ik->i', b, Koo)
    #return g, hdiag

def rotate_mo(mo, u):
    return np.dot(mo, u)

def rearrange(mo, ncore, npair):
    mo2 = np.zeros_like(mo)
    mo2[:,:ncore] = mo[:,:ncore]
    mo2[:,ncore+2*npair:] = mo[:,ncore+2*npair:]
    for i in range(npair):
        mo2[:,ncore+npair-i-1] = mo[:,ncore+2*i]
        mo2[:,ncore+npair+i] = mo[:,ncore+2*i+1]
    return mo2


class SOPNOF(mc1step.CASSCF):
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        mc1step.CASSCF.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.max_cycle_macro = 15
        # classic AH instead of CIAH
        #self.ah_start_tol = 1e-8
        #self.max_stepsize = 1.5
        #self.ah_grad_trust_region = 1e6
        
    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        #self._scf.mo_coeff = mo_coeff
        log = logger.new_logger(self, verbose)
        if eris is None:
            fnof = copy.copy(self)
            fnof.ao2mo = self.get_h2cas
        else:
            fnof = mc1step._fake_h_for_fast_casci(self, mo_coeff, eris) 
        eri_cas = fnof.get_h2eff(mo_coeff)
        h1eff, e_core = fnof.get_h1eff(mo_coeff)
        #print(eri_cas.shape, h1eff.shape)
        e_tot, e_cas = fnof.fcisolver.kernel(self._scf, mo_coeff, self.mo_occ, h1eff, e_core, eri_cas)
        #self.nof = thenof
        self.mo_occ = fnof.fcisolver.mo_occ
        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %.15g', e_cas)

            if getattr(self.fcisolver, 'spin_square', None):
                ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
            else:
                ss = None

            if 'imicro' in envs:  # Within CASSCF iteration
                if ss is None:
                    log.info('macro iter %d (%d JK  %d micro), '
                             'SOPNOF E = %.15g  dE = %.8g',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'])
                else:
                    log.info('macro iter %d (%d JK  %d micro), '
                             'SOPNOF E = %.15g  dE = %.8g  S^2 = %.7f',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'], ss[0])
                #if 'norm_gci' in envs and envs['norm_gci'] is not None:
                #    log.info('               |grad[o]|=%5.3g  '
                #             '|grad[c]|= %s  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                #             envs['norm_gorb0'],
                #             envs['norm_gci'], envs['norm_ddm'], envs['max_offdiag_u'])
                #else:
                #    log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g  |maxRot[o]|=%5.3g',
                #             envs['norm_gorb0'], envs['norm_ddm'], envs['max_offdiag_u'])
            else:  # Initialization step
                if ss is None:
                    log.info('SOPNOF E = %.15g', e_tot)
                else:
                    log.info('SOPNOF E = %.15g  S^2 = %.7f', e_tot, ss[0])
        return e_tot, e_cas, None


class fakeFCISolver():
    def __init__(self):
        #casci.CASCI.__init__(self, mf_or_mol, ncas, nelecas)
        self.nof = None
        self.mo_occ = None
        #self.with_df = False
        self.guess_scal = None
        self.gvbci_maxstep = 0.05
        self.gvbci_maxiter = 35
        self.optimizer = 'ms'

    def kernel(self, _scf, mo_coeff, mo_occ, h1eff, e_core, eri_cas, **kwargs):
        #eri_cas = self.get_h2eff()
        #h1eff, e_core = self.get_h1eff(mo_coeff, ncore, ncas)
        if self.nof is None:
            thenof = PNOF(_scf)
            thenof.ncore = self.ncore
            thenof.npair = self.npair
            #thenof.sorting = self.sorting
            #thenof.with_df = self.with_df
            thenof.guess_scal = self.guess_scal
            thenof.gvbci_maxstep = self.gvbci_maxstep
            thenof.gvbci_maxiter = self.gvbci_maxiter
            thenof.optimizer = self.optimizer
            e = thenof.kernel( h1eff, eri_cas, mo=mo_coeff, mo_occ=mo_occ)[0]
            self.nof = thenof
        else:
            e = self.nof.kernel( h1eff, eri_cas, mo=mo_coeff)[0]
        self.mo_occ = self.nof.mo_occ*2.0
        e_cas = e 
        e_tot = e + e_core
        return e_tot, e_cas
        
    def make_rdm12(self, ci, ncas, nelec):
        nof = self.nof
        ncore = nof.ncore
        f = nof.mo_occ
        Delta = nof.Delta
        Pi = nof.Pi
        a = 2*einsum('q,p->qp', f, f) - 2*Delta
        b = -einsum('q,p->qp', f, f) + Delta + Pi
        #print(a[:ncas,:ncas])
        #print(b[:ncas,:ncas])
        nmo = a.shape[0]
        rdm1 = 2*np.diag(f[ncore:ncore+ncas])
        rdm2_aa = np.zeros((ncas,ncas,ncas,ncas))
        #rdm2_ab = np.zeros((nmo,nmo,nmo,nmo))
        for i in range(ncas):
            for j in range(ncas):
                rdm2_aa[i,i,j,j] += a[ncore+i,ncore+j]
                rdm2_aa[i,j,j,i] += b[ncore+i,ncore+j]
        rdm2 = 2*rdm2_aa
        return rdm1, rdm2

    def make_rdm1(self, ci,ncas, nelec):
        return self.make_rdm12(ci, ncas, nelec)[0]
