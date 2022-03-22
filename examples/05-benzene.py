import sys
sys.path.append("..")
import nof
from automr import guess, dump_mat
from automr.autocas import get_uno, loc_asrot
from pyscf import lib
#lib.num_threads(8)

xyz = '''
 C                 -2.94294278    0.39039038    0.00000000
 C                 -1.54778278    0.39039038    0.00000000
 C                 -0.85024478    1.59814138    0.00000000
 C                 -1.54789878    2.80665038   -0.00119900
 C                 -2.94272378    2.80657238   -0.00167800
 C                 -3.64032478    1.59836638   -0.00068200
 H                 -3.49270178   -0.56192662    0.00045000
 H                 -0.99827478   -0.56212262    0.00131500
 H                  0.24943522    1.59822138    0.00063400
 H                 -0.99769878    3.75879338   -0.00125800
 H                 -3.49284578    3.75885338   -0.00263100
 H                 -4.73992878    1.59854938   -0.00086200
'''
bas = 'ccpvqz'
mf = guess.mix(xyz, bas, conv='tight', newton=True)

mf, unos, noon, nacto, nelecact, ncore, _ = get_uno(mf, thresh=1.98)
#dump_mat.dump_mo(mf.mol, mf.mo_coeff)
mf = loc_asrot(mf, nacto, nelecact, ncore)

thenof = nof.SOPNOF(mf, nacto, nelecact)
thenof.verbose = 5
thenof.mo_occ = noon / 2
thenof.fcisolver = nof.fakeFCISolver()
thenof.fcisolver.ncore = ncore
thenof.fcisolver.npair = nacto//2
#thenof.fcisolver.with_df = True
#thenof.internal_rotation = True
thenof.mc2step()
#dump_mat.dump_mo(thenof.mol, thenof.mo_coeff)







