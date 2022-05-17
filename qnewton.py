import numpy as np
import copy
import sys

a = 3.00
b = 2.00
k = 1.000
k1 = 0.100
k2 = 0.03 #float(sys.argv[1])
#A = 1
#B = 0.5
#v = 2



'''def update(X,E,h,Xs):
    print("update %f,%f -> %f,%f"%(X[h][0],X[h][1],Xs[0],Xs[1]))
    X[h] = Xs
    newX = X
    E[h] = getE(Xs)
    newE = E
    return newX, newE

def findh(E):
    Emax = -100.0
    h = -1
    for i in range(len(E)):
        if E[i] > Emax:
            h = i
            Emax = E[i]
    return h, Emax
def conv(dx):
    thresh = 0.002
    cv = abs(dx[0]) < thresh and abs(dx[1]) < thresh
    return cv
'''
def constrain(X, Xmin, Xmax):
    #print(X)
    over = np.heaviside(X - Xmax, 0.0)
    under = np.heaviside(X - Xmin, 0.0)
    X = X*(1.0-over) + Xmax*over
    X = X*under + Xmin*(1.0-under)
    #print(X)
    return X

def qn_iter(X, getf, getE, t2N, N2t, maxstep=0.05, maxiter=25, debug=False):
    #Xmin, Xmax = Xrange
    conv = False
    E = getE(X)
    f, hdiag = getf(X)
    print('cycle     maxq      dE      maxg')
    print("   0                       %.6f"% abs(f).max())
    if debug: print( 'X', X, 'f', f)
    if np.dot(f.T, f) < 1e-10 :
        conv = True
        return X
    nx = len(X)
    oldE = E
    oldX = copy.copy(X)
    oldf = copy.copy(f)
    oldalpha = 0.5
    #oldG = np.identity(nx)
    oldG = np.diag(hdiag**(-1))
    while(True):
        q = -oldalpha*oldf
        q = constrain(q, -maxstep, maxstep)
        X = oldX + q
        #X = constrain(X, Xmin, Xmax)
        E = getE(X)
        f, hdiag = getf(X)
        dE = E - oldE
        #print(dE)
        if dE > 1e-4:
            print("reset alpha")
            oldalpha /= 2
        else:
            #print("  1   %.3f %.3f %.3f "%(abs(q).max(), dE, abs(f).max()))
            dump_cyc(1, q, dE, f)
            if debug: print('q', q, 'X', X, 'f', f)
            #print("Start Forming U")
            U = -1*np.dot(oldG, f + oldf*(oldalpha-1))
            d = f - oldf
            #print("U: ", U, "d:", d)
            aa = 1.0/np.dot(U.T, d)
            T = np.dot(U.T, U) 
            #print("a: %f  T: %f"%(aa,T), "aUf: %f"%((1/aa)*np.dot(U,oldf)))
            if (1/aa < 1e-5*T) or (abs((1/aa)*np.dot(U.T,oldf)) > 1e-5):
            #if False:
                print("reset G")
                #G = np.identity(nx)
                G = np.diag(hdiag**(-1))
                alpha = 0.5
            else:
                G = oldG + aa*np.einsum('i,j->ij', U, U)
                alpha = 1
            break
    cyc = 2
    while(True):
        oldX = copy.copy(X)
        oldE = E
        oldalpha = alpha
        oldG = copy.copy(G)
        oldf = copy.copy(f)
        #print(oldX, oldE)
        q = -oldalpha*np.dot(oldG, oldf)
        q = constrain(q, -maxstep, maxstep)
        X = oldX + q
        #X = constrain(X, Xmin, Xmax)
        E = getE(X)
        dE = E - oldE
        f, hdiag = getf(X)
        #print("Cycle %d  X: "%cyc, X, "E:", E, 'f:', f)
        dump_cyc(cyc, q, dE, f)
        if debug: print('q', q, 'X', X, 'f', f)
        U = -1*np.dot(oldG, f + oldf*(oldalpha-1))
        d = f - oldf
        #print(U, d)
        aa = 1.0/np.dot(U.T, d)
        T = np.dot(U.T, U) 
        #print("a: %f  T: %.9f"%(aa,T), "aUf: %.9f"% ((1/aa)*np.dot(U,oldf)))
        if (abs(aa*T > 1e5)) or (abs((1/aa)*np.dot(U.T,oldf)) > 1e-5):
        #if False:
            print("reset G")
            #break
            #G = np.identity(nx)
            G = np.diag(hdiag**(-1))
            alpha = 0.5
        else:
            G = oldG + aa*np.einsum('i,j->ij',U, U)
            alpha = 1
    
        if check_conv(f, q, dE):
            print("occ opt converged")
            conv = True
            break
        else:
            cyc += 1
        if cyc > maxiter:
            print("conv not met")
            break
    return X, conv

def check_conv(f, q, dE):
    normf = np.linalg.norm(f)
    maxf = abs(f).max()
    normq = np.linalg.norm(q)
    maxq = abs(q).max()
    conv = normf < 1e-6 and maxf < 1e-5 and normq < 1e-5 and maxq < 7e-5 and dE < 5e-8
    return conv
    

def dump_cyc(cyc, q, dE, f):
    print("  %2d    %-9.5g %-9.5g %-9.5g "%(cyc, abs(q).max(), dE, abs(f).max()))
'''
'''
