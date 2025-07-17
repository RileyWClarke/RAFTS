from config import *
from utils import make_bb
from utils import sed_integ
from mdwarf_interp import *

def initialize():
    
    global bb10k
    bb10k = make_bb(WAVELENGTH, 10000)

    global BBnorm
    BBnorm = 1 / sed_integ(WAVELENGTH[WMIN:WMAX], bb10k[WMIN:WMAX])

    global MDarea
    mdinterp = mdwarf_interp(MDSPEC, plotit=False)
    md = mdinterp(WAVELENGTH[WMIN:WMAX])
    MDarea = sed_integ(WAVELENGTH[WMIN:WMAX], md)
    BBnorm *= MDarea

    global airmass
    airmass = 1.1

    global FF #default model filling factor
    FF = 0.0025

    global LF #what is this?
    LF = [0.115, 0.115]
    #LF = [0.0, 0.0]

    global ROOTDIR
    ROOTDIR = "/Users/fbianco/RAFTS"

    global flareid
    flareid = 11

    global window
    window = 11

