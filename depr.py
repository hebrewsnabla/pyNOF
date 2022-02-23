
def gen_g_hop(nof, mo, mo_occ):
    J, K = nof.ao2mo(mo)
    g, hdiag = get_grad(nof, f, mo, h_mo, ncore, npair, nopen, Delta, Pi, J, K)
    def h_op():
        pass

    def gorb_update(u):
        pass


    return g.reshape(-1), gorb_update, h_op, hdiag.reshape(-1)

def update_rotate_matrix(dx, mo, ncore, npair, u0=1):
    dr = unpack_uniq_var(dx, mo, ncore, npair)
    return np.dot(u0, expmat(dr))

def pack_uniq_var(mat, mo, ncore, npair):
    nmo = mo.shape[-1]
    idx = uniq_var_indices(nmo, ncore, npair)
    return mat[idx]

def uniq_var_indices(nmo, ncore, npair):
    nocc = ncore + 2*npair
    mask = np.zeros((nmo,nmo), dtype=bool)
    mask[ncore:nocc, :ncore] = True
    mask[nocc:, :nocc] = True 
    mask[ncore:nocc, ncore:nocc][np.tril_indices(2*npair,-1)] = True
    return mask

def unpack_uniq_var(v, mo, ncore, npair):
    nmo = mo.shape[-1]
    idx = uniq_var_indices(nmo, ncore, npair)
    mat = np.zeros((nmo,nmo))
    mat[idx] = v
    return mat - mat.T

def rotate_orb_cc(nof, mo, mo_occ, x0_guess=None):
    max_stepsize = 0.03
    conv_tol_grad = 1e-4
    ah_conv_tol = 1e-12
    ah_max_cycle = 30
    ah_lindep = 1e-14
    ah_start_tol = 2.5
    ah_start_cycle = 3
    ah_grad_tr = 3.0
    kf_interval = 4
    kf_tr = 3.0
    scale_restoration = 0.5
    
    ncore = nof.ncore
    npair = nof.npair
    u = 1
    g_orb, gorb_update, h_op, h_diag = gen_g_hop(mo, u)
    g_kf = g_orb
    norm_gkf = norm_gorb = np.linalg.norm(g_orb)
    if norm_gorb < conv_tol_grad*0.3:
        u = update_rotate_matrix(g_orb*0, mo, ncore, npair)
        yield u, g_orb, 1, x0_guess
        return
    
    def precond(x, e):
        hdiagd = h_diag - e 
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        x = x/hdiagd
        norm_x = np.linalg.norm(x)
        x *= 1/norm_x
        return x
    
    jkcount = 0
    if x0_guess is None:
        x0_guess = g_orb
    imic = 0
    dr = 0
    ikf = 0
    g_op = lambda: g_orb
    for ah_end, ihop, w, dxi, hdxi, residual, seig in ciah.davison_cc(h_op, g_op, precond, x0_guess, 
                                                                     tol=ah_conv_tol, max_cycle=ah_max_cycle,
                                                                     lindep=ah_lindep):
        norm_residual = np.linalg.norm(residual)
        if (ah_end or ihop==ah_max_cycle or
            ((norm_residual < ah_start_tol) and (ihop >= ah_start_cycle)) or
            (seig < ah_lindep)):
            imic += 1
            dxmax = np.max(abs(dxi))
            if dxmax > max_stepsize:
                scale = max_stepsize / dxmax
                dxi *= scale
                hdxi *= scale
            else:
                scale = None
            
            g_orb = g_orb + hdxi
            dr = dr + dxi
            norm_gorb, norm_dxi, norm_dr = map(np.linalg.norm, (g_orb, dxi, dr))
            ikf += 1
            if ikf > 1 and norm_gorb > norm_gkf*ah_grad_tr:
                g_orb = g_orb - hdxi
                dr -= dxi
                break
            elif (norm_gorb < conv_tol_grad*0.3):
                break
            elif (ikf >= max(kf_interval, -np.log(norm_dr+1e-7)) or 
                  norm_gorb < norm_gkf/kf_tr):
                ikf = 0
                u = update_rotate_matrix(dr, mo, ncore, npair, u):
                yield u, g_kf, ihop+jkcount, dxi
                g_kf1 = gorb_update(u)
                jkcount += 1

                norm_gkf1, norm_dg = map(np.linalg.norm, (g_kf1, gkf1-g_orb))
                if (norm_dg > norm_gorb*ah_grad_tr and norm_gkf1 > norm_gkf and norm_gkf1 > norm_gkf*ah_grad_tr):
                    dr = -dxi*(1 - scale_restoration)
                    g_kf = g_kf1
                    break
                g_orb = g_kf = g_kf1
                norm_gorb = norm_gkf = norm_gkf1
                dr[:] = 0
    u = update_rotate_matrix(dr, mo, ncore, npair, u)
    yield u, g_kf, ihop+jkcount, dxi