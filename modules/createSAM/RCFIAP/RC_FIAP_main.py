## ############################################################### ##  # noqa: INP001, D100
## RC_FIAP (Reinforced Concrete Frame Inelastic Analysis Platform) ##
##                                                                 ##
## Developed by:                                                   ##
##       Victor F. Ceballos (vceballos@uninorte.edu.co)            ##
##       Carlos A. Arteta (carteta@uninorte.edu.co                 ##
## RC_FIAP_main.py : this is the main script that calls            ##
## GUIFrameNonLinearACI.py : graphical environment                 ##
## mplwidget.py : cript to help plot the plastic hinge projector   ##
## ############################################################### ##

# Modified by Dr. Stevan Gavrilovic @ SimCenter, UC Berkeley

import argparse
import json
import os
import sys
from math import ceil, floor, pi, sqrt

import numpy as np  # load the numpy module, calling it np
import openseespy.opensees as op
import pandas as pd
from scipy import interpolate

# Definition of units
m = 1.0  # define basic units -- output units
kN = 1.0  # define basic units -- output units  # noqa: N816
sec = 1.0  # define basic units -- output units
mm = m / 1000.0  # define engineering units
cm = m / 100.0
N = kN / 1000.0
MPa = N / mm**2
GPa = MPa * 1000
m2 = m**2  # m^2
m3 = m**3  # m^3
m4 = m**4  # m^4
inch = cm * 2.54
ft = 12.0 * inch
g = 9.81 * m / sec**2  # gravitational acceleration
kip = 4.448 * kN
ksi = kip / inch**2
psi = ksi / 1000.0
lbf = psi * inch**2  # pounds force
pcf = lbf / ft**3  # pounds per cubic foot
psf = lbf / ft**3  # pounds per square foot
in2 = inch**2  # inch^2
in4 = inch**4  # inch^4
GConc = 24.0 * kN / m**3  # Specific gravity of concrete
cbar = False
np.set_printoptions(precision=6)


class BeamElasticElement:  # noqa: D101
    def __init__(  # noqa: ANN204, D107, PLR0913
        self,
        EleTag,  # noqa: ANN001, N803
        Nod_ini,  # noqa: ANN001, N803
        Nod_end,  # noqa: ANN001, N803
        AEle,  # noqa: ANN001, N803
        EcEle,  # noqa: ANN001, N803
        IzEle,  # noqa: ANN001, N803
        LEle,  # noqa: ANN001, N803
        BEle,  # noqa: ANN001, N803
        HEle,  # noqa: ANN001, N803
        ElegTr,  # noqa: ANN001, N803
        RZi,  # noqa: ANN001, N803
        RZe,  # noqa: ANN001, N803
    ):
        self.EleTag = EleTag
        self.Nod_ini = Nod_ini
        self.Nod_end = Nod_end
        self.AEle = AEle
        self.EcEle = EcEle
        self.IzEle = IzEle
        self.LEle = LEle
        self.BEle = BEle
        self.HEle = HEle
        self.ElegTr = ElegTr
        self.RZi = RZi
        self.RZe = RZe


class BeamDesing:  # noqa: D101
    def __init__(  # noqa: ANN204, D107, PLR0913
        self,
        EleTag,  # noqa: ANN001, N803
        b,  # noqa: ANN001
        h,  # noqa: ANN001
        Ast1,  # noqa: ANN001, N803
        dt1,  # noqa: ANN001
        Mn_n1,  # noqa: ANN001, N803
        Asb1,  # noqa: ANN001, N803
        db1,  # noqa: ANN001
        Mn_p1,  # noqa: ANN001, N803
        ns1,  # noqa: ANN001
        ss1,  # noqa: ANN001
        Ast2,  # noqa: ANN001, N803
        dt2,  # noqa: ANN001
        Mn_n2,  # noqa: ANN001, N803
        Asb2,  # noqa: ANN001, N803
        db2,  # noqa: ANN001
        Mn_p2,  # noqa: ANN001, N803
        ns2,  # noqa: ANN001
        ss2,  # noqa: ANN001
        Nod_ini,  # noqa: ANN001, N803
        Nod_end,  # noqa: ANN001, N803
        db_t1,  # noqa: ANN001
        db_b1,  # noqa: ANN001
        db_t2,  # noqa: ANN001
        db_b2,  # noqa: ANN001
    ):
        self.EleTag = EleTag
        self.b = b
        self.h = h
        self.Ast1 = Ast1
        self.dt1 = dt1
        self.Mn_n1 = Mn_n1
        self.Asb1 = Asb1
        self.db1 = db1
        self.Mn_p1 = Mn_p1
        self.ns1 = ns1
        self.ss1 = ss1
        self.Ast2 = Ast2
        self.dt2 = dt2
        self.Mn_n2 = Mn_n2
        self.Asb2 = Asb2
        self.db2 = db2
        self.Mn_p2 = Mn_p2
        self.ns2 = ns2
        self.ss2 = ss2
        self.Nod_ini = Nod_ini
        self.Nod_end = Nod_end
        self.db_t1 = db_t1
        self.db_b1 = db_b1
        self.db_t2 = db_t2
        self.db_b2 = db_b2


class ColDesing:  # noqa: D101
    def __init__(  # noqa: ANN204, D107, PLR0913
        self,
        EleTag,  # noqa: ANN001, N803
        b,  # noqa: ANN001
        h,  # noqa: ANN001
        nbH,  # noqa: ANN001, N803
        nbB,  # noqa: ANN001, N803
        db,  # noqa: ANN001
        As,  # noqa: ANN001, N803
        Pu_v,  # noqa: ANN001, N803
        Mu_v,  # noqa: ANN001, N803
        fiPn,  # noqa: ANN001, N803
        fiMn,  # noqa: ANN001, N803
        Mn_i,  # noqa: ANN001, N803
        d,  # noqa: ANN001
        dist,  # noqa: ANN001
        ro,  # noqa: ANN001
        Mu_i,  # noqa: ANN001, N803
        sst,  # noqa: ANN001
        nsB,  # noqa: ANN001, N803
        nsH,  # noqa: ANN001, N803
        Nod_ini,  # noqa: ANN001, N803
        Nod_end,  # noqa: ANN001, N803
    ):
        self.EleTag = EleTag
        self.b = b
        self.h = h
        self.nbH = nbH
        self.nbB = nbB
        self.db = db
        self.As = As
        self.Pu_v = Pu_v
        self.Mu_v = Mu_v
        self.fiPn = fiPn
        self.fiMn = fiMn
        self.Mn_i = Mn_i
        self.d = d
        self.dist = dist
        self.ro = ro
        self.Mu_i = Mu_i
        self.sst = sst
        self.nsB = nsB
        self.nsH = nsH
        self.Nod_ini = Nod_ini
        self.Nod_end = Nod_end


class DuctilityCurve:  # noqa: D101
    def __init__(self, xi, xe, yi, ye, CD_i, CD_e):  # noqa: ANN001, ANN204, N803, D107, PLR0913
        self.xi = xi
        self.xe = xe
        self.yi = yi
        self.ye = ye
        self.CD_i = CD_i
        self.CD_e = CD_e


class TclLogger:  # noqa: D101
    def __init__(self):  # noqa: ANN204, D107
        self.list_of_lines = [
            '# This is an autogenerated .tcl file from SimCenter workflow'
        ]

    # Add a string line to the output file
    def add_line(self, line, addNewLine=True):  # noqa: ANN001, ANN201, FBT002, N803, D102
        if addNewLine == True:  # noqa: E712
            self.list_of_lines.append(line + '\n')
        else:
            self.list_of_lines.append(line)

    # Convenience function to create a line from an array of inputs to openseespy function  # noqa: E501
    def add_array(self, line, addNewLine=True):  # noqa: ANN001, ANN201, FBT002, N803, D102
        outLine = ''  # noqa: N806
        for item in line:
            outLine += str(item) + ' '  # noqa: N806

        #        # Remove the last space
        #        outLine = outLine.rstrip()  # noqa: ERA001
        #
        #        # Add the ; char to the end of the line
        #        outLine += ';'  # noqa: ERA001
        self.add_line(outLine, addNewLine)

    # Save the output file
    def save_as_file(self):  # noqa: ANN201, D102
        # Get the current directory
        workingDirectory = os.getcwd()  # noqa: PTH109, N806

        pathFile = os.path.join(workingDirectory, 'Model.tcl')  # noqa: PTH118, N806

        if os.path.exists(pathFile):  # noqa: PTH110
            os.remove(pathFile)  # noqa: PTH107

        with open(pathFile, 'a+') as file_object:  # noqa: PTH123
            appendEOL = False  # noqa: N806
            # Move read cursor to the start of file.
            file_object.seek(0)

            # Check if file is not empty
            data = file_object.read(100)

            if len(data) > 0:
                appendEOL = True  # noqa: N806

            # Iterate over each string in the list
            for line in self.list_of_lines:
                # If file is not empty then append '\n' before first line for
                # other lines always append '\n' before appending line
                if appendEOL == True:  # noqa: E712
                    file_object.write('\n')
                else:
                    appendEOL = True  # noqa: N806
                # Append element at the end of file
                file_object.write(line)

        # print(self.list_of_lines)  # noqa: ERA001


def runBuildingDesign(BIM_file, EVENT_file, SAM_file, getRV):  # noqa: ANN001, ANN201, ARG001, N802, N803, D103, PLR0915
    # Get the current directory
    workingDirectory = os.getcwd()  # noqa: PTH109, N806, F841

    rootSIM = {}  # noqa: N806

    # Try to open the BIM json
    with open(BIM_file, encoding='utf-8') as f:  # noqa: PTH123
        rootBIM = json.load(f)  # noqa: N806
    try:
        # rootSIM = rootBIM['StructuralInformation']  # noqa: ERA001
        rootSIM = rootBIM['Modeling']  # noqa: N806
        # KZ: append simulation attribute
        rootSIM['Simulation'] = rootBIM.get('Simulation', None)
    except:  # noqa: E722
        raise ValueError('RC_FIAP - structural information missing')  # noqa: B904, EM101, TRY003

    # Get the random variables from the input file
    try:
        rootRV = rootBIM['randomVariables']  # noqa: N806
    except:  # noqa: E722
        raise ValueError('RC_FIAP - randomVariables section missing')  # noqa: B904, EM101, TRY003

    RV_ARRAY = {}  # noqa: N806

    # Populate the RV array with name/value pairs.
    # If a random variable is used here, the RV array will contain its current value
    for rv in rootRV:
        # Try to get the name and value of the random variable
        rvName = rv['name']  # noqa: N806
        curVal = rv['value']  # noqa: N806

        # Check if the current value a realization of a RV, i.e., is not a RV label
        # If so, then set the current value as the mean
        if 'RV' in str(curVal):
            curVal = float(rv['mean'])  # noqa: N806

        RV_ARRAY[rvName] = curVal

    # *********************** Design Starts Here *************************
    # if getRV == "False":
    if getRV is False:
        print('Running seismic design in FIAP')  # noqa: T201

        # Create the tcl output logger
        outputLogger = TclLogger()  # noqa: N806

        outputLogger.add_line(
            '# Reinforced Concrete Frame Inelastic Analysis Platform (RCFIAP)', False  # noqa: FBT003
        )
        outputLogger.add_line(
            '# Developed by Victor Ceballos & Carlos Arteta', False  # noqa: FBT003
        )
        outputLogger.add_line(
            '# Modified by Stevan Gavrilovic - NHERI SimCenter for use in EE-UQ'
        )

        # Create a class object
        RCDes = RCFIAP()  # noqa: N806

        print('Starting seismic design')  # noqa: T201

        # Run the building design
        RCDes.Design(rootSIM)

        print('Creating nonlinear model')  # noqa: T201

        # Run a pushover analysis - for testing to compare with original code
        doPushover = False  # noqa: N806

        # Create the nonlinear model
        RCDes.CreateNLM(rootSIM, outputLogger, doPushover)

        # Save the output file from the logger
        outputLogger.save_as_file()

        if doPushover == True:  # noqa: E712
            print('Running pushover analysis')  # noqa: T201
            RCDes.Pushover(rootSIM)

    # Now create the SAM file for export
    root_SAM = {}  # noqa: N806

    root_SAM['mainScript'] = 'Model.tcl'
    root_SAM['type'] = 'OpenSeesInput'
    root_SAM['units'] = {
        'force': 'kN',
        'length': 'm',
        'temperature': 'C',
        'time': 'sec',
    }

    # Number of dimensions
    root_SAM['ndm'] = 2

    # Number of degrees of freedom at each node
    root_SAM['ndf'] = 3

    # The number of stories
    vecHeights = rootSIM['VecStoryHeights']  # noqa: N806
    vecHeights = vecHeights.split(',')  # noqa: N806
    vecHeights = np.array(vecHeights, dtype=float)  # noqa: N806

    numStories = len(vecHeights)  # noqa: N806
    root_SAM['numStory'] = numStories

    # The number of spans
    vecSpans = rootSIM['VecSpans']  # noqa: N806
    vecSpans = vecSpans.split(',')  # noqa: N806
    vecSpans = np.array(vecSpans, dtype=float)  # noqa: N806
    numSpans = len(vecSpans)  # noqa: N806

    # Get the node mapping
    # Consider a structure with 3 stories and 2 spans
    # Then the node numbering scheme is
    #   #9——————#10———————#11
    #   |        |        |
    #   |        |        |
    #   #6——————#7———————#8
    #   |        |        |
    #   |        |        |
    #   #3——————#4———————#5
    #   |        |        |
    #   |        |        |
    #   #0       #1      #2

    clineOffset = 0  # noqa: N806
    if numSpans > 1:
        clineOffset = int(numSpans / 2)  # noqa: N806

    node_map = []

    # Using nodes on column #1 to calculate story drift
    for i in range(numStories + 1):
        nodeTag = i * (numSpans + 1)  # noqa: N806

        # Create the node and add it to the node mapping array
        node_entry = {}
        node_entry['node'] = nodeTag
        node_entry['cline'] = 'response'
        node_entry['floor'] = f'{i}'
        node_map.append(node_entry)

        ## KZ & AZ: Add centroid for roof drift
        node_entry_c = {}
        node_entry_c['node'] = nodeTag + clineOffset
        node_entry_c['cline'] = 'centroid'
        node_entry_c['floor'] = f'{i}'
        node_map.append(node_entry_c)

    root_SAM['NodeMapping'] = node_map

    with open(SAM_file, 'w') as f:  # noqa: PTH123
        json.dump(root_SAM, f, indent=2)


# Main functionality
class RCFIAP:  # noqa: D101
    def Design(self, rootSIM):  # noqa: ANN001, ANN201, C901, N802, N803, D102, PLR0912, PLR0915
        def __init__(rootSIM):  # noqa: ANN001, ANN202, N803, N807
            self.rootSIM = rootSIM

        global \
            Loc_span, \
            Loc_heigth, \
            ListNodes, \
            Elements, \
            DataBeamDesing, \
            DataColDesing, \
            WDL, \
            WLL, \
            WDLS, \
            Wtotal, \
            cover  # noqa: PLW0603

        # Function: Reads Beams design data from table that allows the user to modify the default design from TAB2 of GUI  # noqa: E501
        def data_beams_table(self):  # noqa: ANN001, ANN202
            self.registros_beams = []

            for DB in DataBeamDesing:  # noqa: N806
                b = DB.b / cm
                h = DB.h / cm
                L_As_top = DB.Ast1 / cm**2  # noqa: N806
                L_As_bot = DB.Asb1 / cm**2  # noqa: N806
                R_As_top = DB.Ast2 / cm**2  # noqa: N806
                R_As_bot = DB.Asb2 / cm**2  # noqa: N806
                L_Leg_n = DB.ns1  # noqa: N806
                R_Leg_n = DB.ns2  # noqa: N806
                L_Sstirrup = DB.ss1 / cm  # noqa: N806
                R_Sstirrup = DB.ss2 / cm  # noqa: N806
                registro = RegistroBeams(  # noqa: F821
                    DB.EleTag,
                    b,
                    h,
                    L_As_top,
                    L_As_bot,
                    L_Leg_n,
                    L_Sstirrup,
                    R_As_top,
                    R_As_bot,
                    R_Leg_n,
                    R_Sstirrup,
                )
                self.registros_beams.append(registro)

        # Function: Reads Columns design data from table that allows the user to modify the default design from TAB2 of GUI.  # noqa: E501
        def data_columns_table(self):  # noqa: ANN001, ANN202
            self.registros_cols = []

            for DC in DataColDesing:  # noqa: N806
                b = DC.b / cm
                h = DC.h / cm
                db = DC.db / mm
                nbH = DC.nbH  # noqa: N806
                nbB = DC.nbB  # noqa: N806
                nsH = DC.nsH  # noqa: N806
                nsB = DC.nsB  # noqa: N806
                sst = DC.sst / cm
                registro = RegistroColumns(  # noqa: F821
                    DC.EleTag, b, h, db, nbH, nbB, nsH, nsB, sst
                )
                self.registros_cols.append(registro)

        # Compression block parameters beta as function f'c
        def beta1(fc):  # noqa: ANN001, ANN202
            if fc <= 28 * MPa:
                Beta1 = 0.85  # noqa: N806
            else:
                Beta1 = max([0.85 - 0.05 * (fc - 28.0) / 7.0, 0.65])  # noqa: N806
            return Beta1

        # Design load combinations
        def Combo_ACI(DL, LL, E):  # noqa: ANN001, ANN202, N802, N803
            U1 = 1.2 * DL + 1.6 * LL  # noqa: N806
            U2 = 1.2 * DL + 1.0 * LL + 1.0 * E  # noqa: N806
            U3 = 1.2 * DL + 1.0 * LL - 1.0 * E  # noqa: N806
            U4 = 0.9 * DL + 1.0 * E  # noqa: N806
            U5 = 0.9 * DL - 1.0 * E  # noqa: N806
            return U1, U2, U3, U4, U5

        # Flexural beams design
        def AsBeam(Mu, EleTag):  # noqa: ANN001, ANN202, N802, N803
            b, h = BBeam, HBeam
            Mu = abs(Mu)  # noqa: N806
            db_v = np.array([4, 5, 6, 7, 8, 10])
            for ndb in db_v:
                db = ndb / 8.0 * inch
                d = h - cover - dst - 0.5 * db
                if Mu == 0.0:
                    ro_req = ro_min_b
                else:
                    ro_req = (
                        0.85
                        * fcB
                        / fy
                        * (
                            1.0
                            - sqrt(1.0 - 2.0 * (Mu / 0.9 / b / d**2) / 0.85 / fcB)
                        )
                    )
                if ro_req < ro_min_b:
                    ro_req = ro_min_b
                As_req = ro_req * b * d  # noqa: N806
                Ab = pi * db**2 / 4.0  # noqa: N806
                nb = max(2.0, ceil(As_req / Ab))
                As_con = nb * Ab  # noqa: N806
                slb = (b - 2 * cover - 2 * dst - nb * db) / (
                    nb - 1.0
                )  # free clear bars
                if slb >= max(1.0 * inch, db):
                    break
                if ro_req > ro_max_b:
                    print(  # noqa: T201
                        'Steel percentage greater than the maximum in Beam '
                        + str(EleTag)
                    )
            if slb < min(1.0 * inch, db):
                print('Bar separation is not ok in Beam ' + str(EleTag))  # noqa: T201
            a = fy * As_con / 0.85 / fcB / b
            Mn = fy * As_con * (d - a / 2.0)  # noqa: N806
            return As_con, d, Mn, db

        # Shear beams design
        def AvBeam(Vu, db, d, EleTag):  # noqa: ANN001, ANN202, N802, N803
            Vc = 0.17 * sqrt(fcB / 1000.0) * MPa * BBeam * d  # noqa: N806
            Vs = (Vu - 0.75 * Vc) / 0.75  # noqa: N806
            if Vs > 4.0 * Vc:
                print('reshape by shear in Beam ' + str(EleTag))  # noqa: T201
            se_1 = min(d / 4.0, 8.0 * db, 24.0 * dst, 300.0 * mm)
            nr_v = np.array([2, 3, 4])  # vector de numero de ramas
            if Vs <= 0.0:
                se = se_1
                nra = 2.0
            else:
                for nra in nr_v:
                    Ave = Ast * nra  # area transversal del estribo  # noqa: N806
                    se_2 = Ave * fy * d / Vs
                    se = min(se_1, se_2)
                    if se >= 60.0 * mm:
                        break
            se = floor(se / cm) * cm
            if se < 60.0 * mm:
                print('Stirrup spacing is less than 6 cm in beam ' + str(EleTag))  # noqa: T201
            return nra, se

        # Colmuns P-M design
        def AsColumn():  # noqa: ANN202, C901, N802
            verif = False
            while verif == False:  # noqa: E712
                for ndb in db_v:
                    db = ndb / 8.0 * inch
                    Ab = pi * db**2.0 / 4.0  # noqa: N806
                    dp = cover + dst + 0.5 * db
                    d = h - dp
                    for nbH in nbH_v:  # noqa: N806
                        for nbB in nbB_v:  # noqa: N806
                            nbT = 2.0 * (nbB + nbH - 2.0)  # numero total de barras  # noqa: N806
                            Ast = nbT * Ab  # noqa: N806
                            ro = Ast / b / h
                            As = np.hstack(  # noqa: N806
                                [nbB * Ab, np.ones(nbH - 2) * 2 * Ab, nbB * Ab]
                            )
                            dist = np.linspace(dp, h - dp, nbH)
                            if ro >= ro_min:
                                Pn_max = 0.80 * (  # noqa: N806
                                    0.85 * fcC * (b * h - Ast) + fy * Ast
                                )
                                Tn_max = -fy * Ast  # noqa: N806
                                c = np.linspace(1.1 * h / npts, 1.1 * h, npts)
                                a = Beta1C * c
                                Pconc = 0.85 * fcC * a * b  # noqa: N806
                                Mconc = Pconc * (h - a) / 2.0  # noqa: N806
                                et = ecu * (d - c) / c
                                fiv = np.copy(et)
                                fiv = np.where(fiv >= 0.005, 0.9, fiv)  # noqa: PLR2004
                                fiv = np.where(fiv <= 0.002, 0.65, fiv)  # noqa: PLR2004
                                fiv = np.where(
                                    (fiv > 0.002) & (fiv < 0.005),  # noqa: PLR2004
                                    (0.65 + 0.25 * (fiv - 0.002) / 0.003),
                                    fiv,
                                )
                                c = c[:, np.newaxis]
                                es = ecu * (c - dist) / c
                                fs = Es * es
                                fs = np.where(fs > fy, fy, fs)
                                fs = np.where(fs < -fy, -fy, fs)
                                Pacer = np.sum(fs * As, axis=1)  # noqa: N806
                                Macer = np.sum(fs * As * (h / 2.0 - dist), axis=1)  # noqa: N806
                                Pn = np.hstack(  # noqa: N806
                                    [
                                        Tn_max,
                                        np.where(
                                            Pconc + Pacer > Pn_max,
                                            Pn_max,
                                            Pconc + Pacer,
                                        ),
                                        Pn_max,
                                    ]
                                )
                                Mn = np.hstack([0, Mconc + Macer, 0])  # noqa: N806
                                fiv = np.hstack([0.9, fiv, 0.65])
                                fiPn = fiv * Pn  # noqa: N806
                                fiMn = fiv * Mn  # noqa: N806
                                if np.all((Pu_v >= min(fiPn)) & (Pu_v <= max(fiPn))):
                                    Mu_i = np.interp(Pu_v, fiPn, fiMn)  # noqa: N806
                                    Mn_i = np.interp(Pu_v, Pn, Mn)  # noqa: N806
                                    if np.all(Mu_i >= Mu_v) == True:  # noqa: E712
                                        verif = True
                                        break
                        if verif == True:  # noqa: E712
                            break
                    if verif == True:  # noqa: E712
                        break
                if ndb == db_v[-1] and ro > ro_max:
                    print(  # noqa: T201
                        'column '
                        + str(EleTag)
                        + 'needs to be resized by reinforcement ratio'
                    )
                    break
            return nbH, nbB, db, As, fiPn, fiMn, Mn_i, d, dist, ro, Mu_i

        # Shear columns design
        def AvColumn():  # noqa: ANN202, N802
            fiv = 0.75
            Ag = b * h  # noqa: N806
            se_1 = min(
                8.0 * db, b / 2.0, h / 2.0, 200.0 * mm
            )  # separacion minima c.18.4.3.3 ACI-19
            dp = cover + dst + db / 2
            d = h - dp
            neH = floor(nbH / 2) + 1  # noqa: N806
            neB = floor(nbB / 2) + 1  # noqa: N806

            Ash_H = neH * Ast  # noqa: N806, F841
            Ash_B = neB * Ast  # noqa: N806

            Vc = (0.17 * sqrt(fcC * MPa) + Nu_min / 6 / Ag) * b * d  # noqa: N806
            Vs = (Vu - fiv * Vc) / fiv  # noqa: N806
            if Vs <= 1 / 3 * sqrt(fcC * MPa) * b * d:
                se_1 = se_1  # noqa: PLW0127
            elif Vs >= 1 / 3 * sqrt(fcC * MPa) * b * d:
                se_1 = min(se_1, h / 4)

            if Vs > 0.66 * sqrt(fcC * MPa) * b * d:
                print('Resize the column' + str(EleTag) + ' by shear ')  # noqa: T201

            if Vs <= 0.0:
                se = se_1
            else:
                Ave = Ash_B  # area transversal del estribo  # noqa: N806
                se_2 = Ave * fy * d / Vs
                se = min([se_1, se_2])
            if se < 60.0 * mm:
                print(  # noqa: T201
                    'Minimum spacing of stirrups is not met in column ' + str(EleTag)
                )
            return se, neB, neH

        # Input geometric, materials and seismic design parameters from TAB1 of GUI

        # Lafg = float(self.ui.Lafg.text())  # noqa: ERA001
        Lafg = float(rootSIM['TribLengthGravity'])  # noqa: N806

        # Lafs = float(self.ui.Lafs.text())  # noqa: ERA001
        Lafs = float(rootSIM['TribLengthSeismic'])  # noqa: N806

        # DL = float(self.ui.DL.text())  # noqa: ERA001
        DL = float(rootSIM['DeadLoad'])  # noqa: N806

        # LL = float(self.ui.LL.text())  # noqa: ERA001
        LL = float(rootSIM['LiveLoad'])  # noqa: N806

        # HColi = float(self.ui.HColi.text())  # Column inside Depth  # noqa: ERA001
        HColi = float(rootSIM['IntColDepth'])  # noqa: N806

        # BColi = float(self.ui.BColi.text())  # Column inside Width  # noqa: ERA001
        BColi = float(rootSIM['IntColWidth'])  # noqa: N806

        # HCole = float(self.ui.HCole.text())  # Column outside Depth  # noqa: ERA001
        HCole = float(rootSIM['ExtColDepth'])  # noqa: N806

        # BCole = float(self.ui.BCole.text())  # Column outside Width  # noqa: ERA001
        BCole = float(rootSIM['ExtColWidth'])  # noqa: N806

        # HBeam = float(self.ui.HBeam.text())  # noqa: ERA001
        HBeam = float(rootSIM['BeamDepth'])  # noqa: N806

        # BBeam = float(self.ui.BBeam.text())  # noqa: ERA001
        BBeam = float(rootSIM['BeamWidth'])  # noqa: N806

        # IFC = float(self.ui.InertiaColumnsFactor.text())  # noqa: ERA001
        IFC = float(rootSIM['ColIg'])  # noqa: N806

        # IFB = float(self.ui.InertiaBeamsFactor.text())  # noqa: ERA001
        IFB = float(rootSIM['BeamIg'])  # noqa: N806

        # heigth_v = self.ui.heigth_v.text()  # noqa: ERA001
        heigth_v = rootSIM['VecStoryHeights']

        heigth_v = heigth_v.split(',')
        heigth_v = np.array(heigth_v, dtype=float)

        # span_v = self.ui.span_v.text()  # noqa: ERA001
        span_v = rootSIM['VecSpans']

        span_v = span_v.split(',')
        span_v = np.array(span_v, dtype=float)

        # fy = float(self.ui.fy.text()) * MPa  # noqa: ERA001
        fy = float(rootSIM['FySteel']) * MPa

        # fcB = float(self.ui.fcB.text()) * MPa  # noqa: ERA001
        fcB = float(rootSIM['BeamFpc']) * MPa  # noqa: N806

        # fcC = float(self.ui.fcC.text()) * MPa  # noqa: ERA001
        fcC = float(rootSIM['ColFpc']) * MPa  # noqa: N806

        # R = float(self.ui.R.text())  # noqa: ERA001
        R = float(rootSIM['RParam'])  # noqa: N806

        # Cd = float(self.ui.Cd.text())  # noqa: ERA001
        Cd = float(rootSIM['CdParam'])  # noqa: N806, F841

        # Omo = float(self.ui.Omo.text())  # noqa: ERA001
        Omo = float(rootSIM['OmegaParam'])  # noqa: N806

        # Sds = float(self.ui.Sds.text())  # noqa: ERA001
        Sds = float(rootSIM['SDSParam'])  # noqa: N806

        # Sd1 = float(self.ui.Sd1.text())  # noqa: ERA001
        Sd1 = float(rootSIM['SD1Param'])  # noqa: N806

        # Tl = float(self.ui.Tl.text())  # noqa: ERA001
        Tl = float(rootSIM['TLParam'])  # noqa: N806

        WDL = Lafg * DL
        WDLS = Lafs * DL
        WLL = Lafg * LL

        #        print("heigth_v: ")  # noqa: ERA001
        #        print(heigth_v)  # noqa: ERA001
        #        print("span_v: ")  # noqa: ERA001
        #        print(span_v)  # noqa: ERA001
        #        print("Lafg: "+str(Lafg))  # noqa: ERA001
        #        print("Lafs: "+str(Lafs))  # noqa: ERA001
        #        print("DL: "+str(DL))  # noqa: ERA001
        #        print("LL: "+str(LL))  # noqa: ERA001
        #        print("HColi: "+str(HColi))  # noqa: ERA001
        #        print("BColi: "+str(BColi))  # noqa: ERA001
        #        print("HCole: "+str(HCole))  # noqa: ERA001
        #        print("BCole: "+str(BCole))  # noqa: ERA001
        #        print("HBeam: "+str(HBeam))  # noqa: ERA001
        #        print("BBeam: "+str(BBeam))  # noqa: ERA001
        #        print("IFC: "+str(IFC))  # noqa: ERA001
        #        print("IFB: "+str(IFB))  # noqa: ERA001
        print('********************fy: ', fy)  # noqa: T201
        #        print("fcB: "+str(fcB))  # noqa: ERA001
        #        print("fcC: "+str(fcC))  # noqa: ERA001
        #        print("R: "+str(R))  # noqa: ERA001
        #        print("Cd: "+str(Cd))  # noqa: ERA001
        #        print("Omo: "+str(Omo))  # noqa: ERA001
        #        print("Sds: "+str(Sds))  # noqa: ERA001
        #        print("Sd1: "+str(Sd1))  # noqa: ERA001
        #        print("Tl: "+str(Tl))  # noqa: ERA001

        # plt.close('all')  # noqa: ERA001
        op.wipe()
        op.model('Basic', '-ndm', 2, '-ndf', 3)

        # Nodes Creations
        Loc_span = np.append(0, np.cumsum(span_v))
        Loc_heigth = np.append(0, np.cumsum(heigth_v))
        n_col_axes = len(Loc_span)
        xn_v, yn_v = np.meshgrid(Loc_span, Loc_heigth)
        xn_vf = np.ravel(xn_v)
        yn_vf = np.ravel(yn_v)
        num_nodes = len(Loc_span) * len(Loc_heigth)
        ListNodes = np.empty([num_nodes, 3])
        nodeTag = 0  # noqa: N806
        for xn, yn in zip(xn_vf, yn_vf):
            ListNodes[nodeTag, :] = [nodeTag, xn, yn]
            op.node(nodeTag, xn, yn)
            if yn == 0.0:
                op.fix(nodeTag, 1, 1, 1)
            nodeTag += 1  # noqa: SIM113, N806
        for node in ListNodes:
            if node[2] > 0.0 and node[1] == 0.0:
                MasterNode = node[0]  # noqa: N806
            if node[2] > 0.0 and node[1] != 0.0:
                op.equalDOF(int(MasterNode), int(node[0]), 1)

        ListNodesDrift = ListNodes[np.where(ListNodes[:, 1] == 0.0)]  # noqa: N806
        MassType = '-lMass'  # -lMass, -cMass  # noqa: N806

        # Columns creation for elastic analysis
        op.geomTransf('Linear', 1, '-jntOffset', 0, 0, 0, -HBeam / 2)
        op.geomTransf('Linear', 2, '-jntOffset', 0, HBeam / 2, 0, -HBeam / 2)
        AColi = BColi * HColi  # cross-sectional area  # noqa: N806
        ACole = BCole * HCole  # cross-sectional area  # noqa: N806
        EcC = 4700 * sqrt(fcC * MPa)  # noqa: N806
        IzColi = 1.0 / 12.0 * BColi * HColi**3  # Column moment of inertia  # noqa: N806
        IzCole = 1.0 / 12.0 * BCole * HCole**3  # Column moment of inertia  # noqa: N806
        EleTag = 1  # noqa: N806
        Elements = []
        for Nod_ini in range(num_nodes):  # noqa: N806
            if ListNodes[Nod_ini, 2] != Loc_heigth[-1]:
                Nod_end = Nod_ini + n_col_axes  # noqa: N806
                if ListNodes[Nod_ini, 2] == 0.0:
                    gTr = 1  # noqa: N806
                    RZi = 0  # noqa: N806
                    RZe = HBeam / 2  # noqa: N806
                    LCol = ListNodes[Nod_end, 2] - ListNodes[Nod_ini, 2] - RZi - RZe  # noqa: N806
                else:
                    gTr = 2  # noqa: N806
                    RZi = HBeam / 2  # noqa: N806
                    RZe = HBeam / 2  # noqa: N806
                    LCol = ListNodes[Nod_end, 2] - ListNodes[Nod_ini, 2] - RZi - RZe  # noqa: N806
                if (
                    ListNodes[Nod_ini, 1] == 0.0
                    or ListNodes[Nod_ini, 1] == Loc_span[-1]
                ):
                    BCol, HCol = BCole, HCole  # noqa: N806
                    ACol = ACole  # noqa: N806
                    IzCol = IFC * IzCole  # noqa: N806
                else:
                    BCol, HCol = BColi, HColi  # noqa: N806
                    ACol = AColi  # noqa: N806
                    IzCol = IFC * IzColi  # noqa: N806
                MassDens = ACol * GConc / g  # noqa: N806
                Elements.append(
                    BeamElasticElement(
                        EleTag,
                        Nod_ini,
                        Nod_end,
                        ACol,
                        EcC,
                        IzCol,
                        LCol,
                        BCol,
                        HCol,
                        gTr,
                        RZi,
                        RZe,
                    )
                )
                op.element(
                    'elasticBeamColumn',
                    EleTag,
                    Nod_ini,
                    Nod_end,
                    ACol,
                    EcC,
                    IzCol,
                    gTr,
                    '-mass',
                    MassDens,
                    MassType,
                )
                EleTag += 1  # noqa: N806
        num_cols = EleTag

        # Beams creation for elastic analysis
        op.geomTransf('Linear', 3, '-jntOffset', HColi / 2.0, 0, -HColi / 2.0, 0)
        op.geomTransf('Linear', 4, '-jntOffset', HCole / 2.0, 0, -HColi / 2.0, 0)
        op.geomTransf('Linear', 5, '-jntOffset', HColi / 2.0, 0, -HCole / 2.0, 0)
        ABeam = BBeam * HBeam  # noqa: N806
        EcB = 4700 * sqrt(fcB * MPa)  # noqa: N806
        IzBeam = IFB * BBeam * HBeam**3 / 12  # noqa: N806
        MassDens = ABeam * GConc / g + WDLS / g  # noqa: N806
        for Nod_ini in range(num_nodes):  # noqa: N806
            if (
                ListNodes[Nod_ini, 1] != Loc_span[-1]
                and ListNodes[Nod_ini, 2] != 0.0
            ):
                Nod_end = Nod_ini + 1  # noqa: N806
                if ListNodes[Nod_ini, 1] == 0.0:
                    gTr = 4  # noqa: N806
                    RZi = HCole / 2.0  # noqa: N806
                    RZe = HColi / 2.0  # noqa: N806
                    LBeam = ListNodes[Nod_end, 1] - ListNodes[Nod_ini, 1] - RZi - RZe  # noqa: N806
                elif ListNodes[Nod_ini, 1] == Loc_span[-2]:
                    gTr = 5  # noqa: N806
                    RZi = HColi / 2.0  # noqa: N806
                    RZe = HCole / 2.0  # noqa: N806
                    LBeam = ListNodes[Nod_end, 1] - ListNodes[Nod_ini, 1] - RZi - RZe  # noqa: N806
                else:
                    gTr = 3  # noqa: N806
                    RZi = HColi / 2.0  # noqa: N806
                    RZe = HColi / 2.0  # noqa: N806
                    LBeam = ListNodes[Nod_end, 1] - ListNodes[Nod_ini, 1] - RZi - RZe  # noqa: N806
                Elements.append(
                    BeamElasticElement(
                        EleTag,
                        Nod_ini,
                        Nod_end,
                        ABeam,
                        EcB,
                        IzBeam,
                        LBeam,
                        BBeam,
                        HBeam,
                        gTr,
                        RZi,
                        RZe,
                    )
                )
                op.element(
                    'elasticBeamColumn',
                    EleTag,
                    Nod_ini,
                    Nod_end,
                    ABeam,
                    EcB,
                    IzBeam,
                    gTr,
                    '-mass',
                    MassDens,
                    MassType,
                )
                EleTag += 1  # noqa: N806
        num_elems = EleTag
        num_beams = num_elems - num_cols  # noqa: F841

        # Create a Plain load pattern for gravity loading with a Linear TimeSeries
        Pvig = ABeam * GConc  # noqa: N806
        PColi = AColi * GConc  # noqa: N806
        PCole = ACole * GConc  # noqa: N806
        op.timeSeries('Linear', 1)
        op.pattern('Plain', 1, 1)
        for Element in Elements:  # noqa: N806
            if ListNodes[Element.Nod_ini, 1] == ListNodes[Element.Nod_end, 1]:
                if (
                    ListNodes[Element.Nod_ini, 1] == 0.0
                    or ListNodes[Element.Nod_ini, 1] == Loc_span[-1]
                ):
                    PCol = PCole  # noqa: N806
                else:
                    PCol = PColi  # noqa: N806
                op.eleLoad('-ele', Element.EleTag, '-type', '-beamUniform', 0, -PCol)
            if ListNodes[Element.Nod_ini, 2] == ListNodes[Element.Nod_end, 2]:
                op.eleLoad(
                    '-ele', Element.EleTag, '-type', '-beamUniform', -Pvig - WDL
                )

        op.system('UmfPack')
        op.numberer('Plain')
        op.constraints('Plain')
        op.integrator('LoadControl', 1.0)
        op.algorithm('Linear')
        op.analysis('Static')
        op.analyze(1)
        ElemnsForceD = []  # noqa: N806
        for Element in Elements:  # noqa: N806
            Forces = op.eleForce(Element.EleTag)  # noqa: N806
            Forces.insert(0, Element.EleTag)
            ElemnsForceD.append(Forces)
        ElemnsForceD = np.array(ElemnsForceD)  # noqa: N806
        Wtotal = np.sum(ElemnsForceD[: len(Loc_span), 2]) * Lafs / Lafg

        op.loadConst('-time', 0.0)
        op.timeSeries('Linear', 2)
        op.pattern('Plain', 2, 1)
        for Element in Elements:  # noqa: N806
            if ListNodes[Element.Nod_ini, 2] == ListNodes[Element.Nod_end, 2]:
                op.eleLoad('-ele', Element.EleTag, '-type', '-beamUniform', -WLL)
        op.analyze(1)

        # Frame Geometry plot
        #        self.ui.DataFrame.canvas.axes.clear()  # noqa: ERA001
        #        self.ui.DataFrame.canvas.axes.plot(ListNodes[:, 1], ListNodes[:, 2], 'ks')  # noqa: ERA001, E501
        #
        #        self.ui.DataFrame.canvas.axes.axis('off')  # noqa: ERA001
        #        for Ele in Elements:
        #            xi = ListNodes[Ele.Nod_ini, 1]  # noqa: ERA001
        #            yi = ListNodes[Ele.Nod_ini, 2]  # noqa: ERA001
        #            xe = ListNodes[Ele.Nod_end, 1]  # noqa: ERA001
        #            ye = ListNodes[Ele.Nod_end, 2]  # noqa: ERA001
        #            self.ui.DataFrame.canvas.axes.plot([xi, xe], [yi, ye], 'k-', alpha=.3)  # noqa: ERA001, E501
        #            if xi == xe:
        #                self.ui.DataFrame.canvas.axes.text(xi, (ye + yi) / 2, r'C{}'.format(Ele.EleTag), style='italic',  # noqa: E501
        #                                                   fontsize=8,  # noqa: ERA001
        #                                                   rotation='vertical', verticalalignment='center')  # noqa: E501
        #            if yi == ye:
        #                self.ui.DataFrame.canvas.axes.text((xe + xi) / 2, yi, r'B{}'.format(Ele.EleTag), style='italic',  # noqa: E501
        #                                                   fontsize=8,  # noqa: ERA001
        #                                                   horizontalalignment='center')  # noqa: E501
        #        self.ui.DataFrame.canvas.axes.axis('equal')  # noqa: ERA001
        #        self.ui.DataFrame.canvas.draw()  # noqa: ERA001
        #        self.ui.DataFrame.canvas.show()  # noqa: ERA001

        ElemnsForceDL = []  # noqa: N806
        for Element in Elements:  # noqa: N806
            Forces = op.eleForce(Element.EleTag)  # noqa: N806
            Forces.insert(0, Element.EleTag)
            ElemnsForceDL.append(Forces)
        ElemnsForceDL = np.array(ElemnsForceDL)  # noqa: N806

        # Create a Plain load pattern for seismic loading with a Linear TimeSeries (LLEF)  # noqa: E501
        op.loadConst('-time', 0.0)
        Htotal = Loc_heigth[-1]  # noqa: N806
        Ct = 0.0466  # noqa: N806
        x = 0.9
        Ta = Ct * Htotal**x  # noqa: N806
        print('Ta =', Ta)  # noqa: T201
        Ie = 1.0  # noqa: N806
        Ts = Sd1 / Sds  # noqa: N806
        if Ta <= Ts:
            Sa = max(Sds * Ie / R, 0.044 * Sds * Ie, 0.01)  # noqa: N806
        elif Ta <= Tl:
            Sa = max(Sd1 * Ie / Ta / R, 0.044 * Sds * Ie, 0.01)  # noqa: N806
        else:
            Sa = max(Sd1 * Tl * Ie / (Ta**2) / R, 0.044 * Sds * Ie, 0.01)  # noqa: N806
        if Ta <= 0.5:  # noqa: PLR2004
            k = 1.0
        elif Ta <= 2.5:  # noqa: PLR2004
            k = 0.75 + 0.5 * Ta
        else:
            k = 2.0
        sumH = np.sum(np.power(Loc_heigth, k))  # noqa: N806

        op.timeSeries('Linear', 3)
        op.pattern('Plain', 3, 1)
        print('Wtotal =', Wtotal)  # noqa: T201
        Fp = Sa * Wtotal * np.power(Loc_heigth, k) / sumH  # noqa: N806
        print('FSis =', Fp)  # noqa: T201
        for fp, ind in zip(Fp, range(len(Loc_heigth))):
            op.load(int(ListNodesDrift[ind, 0]), fp, 0.0, 0.0)
        Vbasal = Sa * Wtotal  # noqa: N806, F841

        op.analyze(1)
        ElemnsForceDLE = []  # noqa: N806
        for Element in Elements:  # noqa: N806
            Forces = op.eleForce(Element.EleTag)  # noqa: N806
            Forces.insert(0, Element.EleTag)
            ElemnsForceDLE.append(Forces)
        ElemnsForceDLE = np.array(ElemnsForceDLE)  # noqa: N806
        np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)

        # Story drift caculations
        DriftMax = 0.02  # noqa: N806
        nodesDisp = []  # noqa: N806
        Id_Node_Drift = ListNodesDrift[:, 0]  # noqa: N806
        Id_Node_Drift = np.int64(Id_Node_Drift)  # noqa: N806
        Id_Node_Drift = Id_Node_Drift.tolist()  # noqa: N806
        for nodo in Id_Node_Drift:
            nodesDisp.append([nodo, op.nodeDisp(nodo, 1)])  # noqa: PERF401
        nodesDisp = np.array(nodesDisp)  # noqa: N806
        drift = nodesDisp[1:, 1] - nodesDisp[:-1, 1]
        drift_p = np.divide(drift, np.array(heigth_v))
        ver_drift = np.where(drift_p < DriftMax, 'ok', 'not ok')
        Id_Floor = np.arange(1, len(Loc_heigth))  # noqa: N806
        drift_table = pd.DataFrame(
            {'1.Floor': Id_Floor, '2.Drift': drift_p * 100, '3.': ver_drift}
        )
        print(drift_table)  # noqa: T201

        # Beams and columns design procedures
        Beta1B = beta1(fcB)  # noqa: N806
        cover = 4 * cm
        dst = 3 / 8 * inch
        Ast = pi * dst**2 / 4.0  # area de la barra del estribo  # noqa: N806
        ro_max_b = 0.85 * Beta1B * fcB * 3.0 / fy / 8.0  # maximun steel percentage
        ro_min_b = max(
            0.25 * sqrt(fcB / MPa) * MPa / fy, 1.4 * MPa / fy
        )  # minimun steel percentage
        DataBeamDesing = []
        for Ele, EleForceD, EleForceDL, EleForceDLE in zip(  # noqa: N806
            Elements, ElemnsForceD, ElemnsForceDL, ElemnsForceDLE
        ):
            if ListNodes[Ele.Nod_ini, 2] == ListNodes[Ele.Nod_end, 2]:
                VID = EleForceD[2]  # noqa: N806
                VIL = EleForceDL[2] - VID  # noqa: N806
                VIE = EleForceDLE[2] - VID - VIL  # noqa: N806
                VED = abs(EleForceD[5])  # noqa: N806
                VEL = abs(EleForceDL[5]) - VED  # noqa: N806
                VEE = abs(EleForceDLE[5]) - VED - VEL  # noqa: N806

                MID = EleForceD[3] - EleForceD[2] * Ele.RZi  # noqa: N806
                MIL = EleForceDL[3] - EleForceDL[2] * Ele.RZi - MID  # noqa: N806
                MIE = EleForceDLE[3] - EleForceDLE[2] * Ele.RZi - MID - MIL  # noqa: N806
                MED = EleForceD[6] + EleForceD[5] * Ele.RZe  # noqa: N806
                MEL = EleForceDL[6] + EleForceDL[5] * Ele.RZe - MED  # noqa: N806
                MEE = EleForceDLE[6] + EleForceDLE[5] * Ele.RZe - MED - MEL  # noqa: N806
                MED, MEL, MEE = -MED, -MEL, -MEE  # noqa: N806
                print(  # noqa: T201
                    'MID ',
                    MID,
                    'MED',
                    MED,
                    'MIL ',
                    MIL,
                    'MEL',
                    MEL,
                    'MIE ',
                    MIE,
                    'MEE',
                    MEE,
                )
                MI1, MI2, MI3, MI4, MI5 = Combo_ACI(MID, MIL, MIE)  # noqa: N806
                MNU1 = max(  # noqa: N806
                    [MI1, MI2, MI3, MI4, MI5, 0.0]
                )  # Momento negativo nudo inicial de diseño
                MPU1 = min(  # noqa: N806
                    [MI1, MI2, MI3, MI4, MI5, abs(MNU1) / 3]
                )  # Momento positivo nudo inicial de diseño
                ME1, ME2, ME3, ME4, ME5 = Combo_ACI(MED, MEL, MEE)  # noqa: N806
                MNU2 = max(  # noqa: N806
                    [ME1, ME2, ME3, ME4, ME5, 0.0]
                )  # Momento negativo nudo final de diseño
                MPU2 = min(  # noqa: N806
                    [ME1, ME2, ME3, ME4, ME5, abs(MNU2) / 3]
                )  # Momento positivo nudo final de diseño
                Mmax = max([MNU1, -MPU1, MNU2, -MPU2])  # noqa: N806
                MNU1 = max([MNU1, Mmax / 5])  # noqa: N806
                MPU1 = min([MPU1, -Mmax / 5])  # noqa: N806
                MNU2 = max([MNU2, Mmax / 5])  # noqa: N806
                MPU2 = min([MPU2, -Mmax / 5])  # noqa: N806

                Ast1, dt1, Mn_N1, db_t1 = AsBeam(MNU1, Ele.EleTag)  # noqa: N806
                Asb1, db1, Mn_P1, db_b1 = AsBeam(MPU1, Ele.EleTag)  # noqa: N806
                Ast2, dt2, Mn_N2, db_t2 = AsBeam(MNU2, Ele.EleTag)  # noqa: N806
                Asb2, db2, Mn_P2, db_b2 = AsBeam(MPU2, Ele.EleTag)  # noqa: N806

                VI1 = 1.2 * VID + 1.6 * VIL  # noqa: N806
                VI2 = 1.2 * VID + 1.0 * VIL - 1.0 * VIE  # noqa: N806
                VI3 = 0.9 * VID - 1.0 * VIE  # noqa: N806
                VI4 = (Mn_P1 + Mn_N2) / Ele.LEle + (1.2 * WDL + WLL) * Ele.LEle / 2.0  # noqa: N806
                VI5 = (Mn_N1 + Mn_P2) / Ele.LEle + (1.2 * WDL + WLL) * Ele.LEle / 2.0  # noqa: N806
                VI6 = 1.2 * VID + 1.0 * VIL - 2.0 * VIE  # noqa: N806
                VI7 = 0.9 * VID - 2.0 * VIE  # noqa: N806

                VU1a = max(VI1, VI2, VI3)  # noqa: N806
                VU1b = max(VI4, VI5)  # noqa: N806
                VU1c = max(VI6, VI7)  # noqa: N806

                VU1 = max(  # noqa: N806
                    VU1a, min(VU1b, VU1c)
                )  # Cortante negativo nudo inicial de diseño

                VE1 = 1.2 * VED + 1.6 * VEL  # noqa: N806
                VE2 = 1.2 * VED + 1.0 * VEL + 1.0 * VEE  # noqa: N806
                VE3 = 0.9 * VED + 1.0 * VEE  # noqa: N806
                VE4 = (Mn_P1 + Mn_N2) / Ele.LEle + (1.2 * WDL + WLL) * Ele.LEle / 2.0  # noqa: N806
                VE5 = (Mn_N1 + Mn_P2) / Ele.LEle + (1.2 * WDL + WLL) * Ele.LEle / 2.0  # noqa: N806
                VE6 = 1.2 * VED + 1.0 * VEL + 2.0 * VEE  # noqa: N806
                VE7 = 0.9 * VED + 2.0 * VEE  # noqa: N806

                VU2a = max(VE1, VE2, VE3)  # noqa: N806
                VU2b = max(VE4, VE5)  # noqa: N806
                VU2c = max(VE6, VE7)  # noqa: N806

                VU2 = max(  # noqa: N806
                    VU2a, min(VU2b, VU2c)
                )  # Cortante negativo nudo final de diseño

                nst1, sst1 = AvBeam(VU1, db_t1, dt1, Ele.EleTag)
                nst2, sst2 = AvBeam(VU2, db_t2, dt2, Ele.EleTag)

                DataBeamDesing.append(
                    BeamDesing(
                        Ele.EleTag,
                        BBeam,
                        HBeam,
                        Ast1,
                        dt1,
                        Mn_N1,
                        Asb1,
                        db1,
                        Mn_P1,
                        nst1,
                        sst1,
                        Ast2,
                        dt2,
                        Mn_N2,
                        Asb2,
                        db2,
                        Mn_P2,
                        nst2,
                        sst2,
                        Ele.Nod_ini,
                        Ele.Nod_end,
                        db_t1,
                        db_b1,
                        db_t2,
                        db_b2,
                    )
                )
                # self.ui.tbl_data_design_beams.setRowCount(0)  # noqa: ERA001
                # data_beams_table(self)  # noqa: ERA001

        # Column design procedure
        ro_min = 0.01
        ro_max = 0.08
        Beta1C = beta1(fcC)  # noqa: N806
        npts = 20
        ncom = 10  # noqa: F841
        ecu = 0.003
        Es = 200.0 * GPa  # noqa: N806

        db_v = np.array(
            [5, 6, 7, 8, 9, 10, 11, 14, 18]
        )  # vector de diametros de barras
        DataColDesing = []
        for Ele, EleForceD, EleForceDL, EleForceDLE in zip(  # noqa: N806
            Elements, ElemnsForceD, ElemnsForceDL, ElemnsForceDLE
        ):
            if ListNodes[Ele.Nod_ini, 1] == ListNodes[Ele.Nod_end, 1]:
                Mn_N_R, Mn_P_R, Mn_N_L, Mn_P_L = 0, 0, 0, 0  # noqa: N806
                for DB in DataBeamDesing:  # noqa: N806
                    if Ele.Nod_end == DB.Nod_ini:
                        Mn_N_R, Mn_P_R = DB.Mn_n1, DB.Mn_p1  # noqa: N806
                    if Ele.Nod_end == DB.Nod_end:
                        Mn_N_L, Mn_P_L = DB.Mn_n2, DB.Mn_p2  # noqa: N806
                Sum_Mn_B = max(Mn_P_R + Mn_N_L, Mn_N_R + Mn_P_L)  # noqa: N806, F841
                b, h = Ele.BEle, Ele.HEle
                nbB = ceil(b * 10)  # bars numbers along B  # noqa: N806
                nbH = ceil(h * 10)  # bars numbers along H  # noqa: N806
                D_c = 1.1 * h / npts  # noqa: N806, F841
                nbH_v = np.array([nbH - 1, nbH, nbH + 1])  # noqa: N806
                nbB_v = np.array([nbB - 1, nbB, nbB + 1])  # noqa: N806

                MID = EleForceD[3]  # noqa: N806
                MIL = EleForceDL[3] - MID  # noqa: N806
                MIE = EleForceDLE[3] - MID - MIL  # noqa: N806

                PID = EleForceD[2]  # noqa: N806
                PIL = EleForceDL[2] - PID  # noqa: N806
                PIE = EleForceDLE[2] - PID - PIL  # noqa: N806

                MI1, MI2, MI3, MI4, MI5 = Combo_ACI(MID, MIL, MIE)  # noqa: N806
                PI1, PI2, PI3, PI4, PI5 = Combo_ACI(PID, PIL, PIE)  # noqa: N806

                MED = -EleForceD[6]  # noqa: N806
                MEL = -EleForceDL[6] - MED  # noqa: N806
                MEE = -EleForceDLE[6] - MED - MEL  # noqa: N806
                print(  # noqa: T201
                    'MID ',
                    MID,
                    'MED',
                    MED,
                    'MIL ',
                    MIL,
                    'MEL',
                    MEL,
                    'MIE ',
                    MIE,
                    'MEE',
                    MEE,
                )

                PED = -EleForceD[5]  # noqa: N806
                PEL = -EleForceDL[5] - PED  # noqa: N806
                PEE = -EleForceDLE[5] - PED - PEL  # noqa: N806

                ME1, ME2, ME3, ME4, ME5 = Combo_ACI(MED, MEL, MEE)  # noqa: N806
                PE1, PE2, PE3, PE4, PE5 = Combo_ACI(PED, PEL, PEE)  # noqa: N806

                Nu_min = min([PI2, PI3, PI4, PI5, PE2, PE3, PE4, PE5])  # noqa: N806

                Pu_v = np.array([PI1, PI2, PI3, PI4, PI5, PE1, PE2, PE3, PE4, PE5])  # noqa: N806
                Mu_v = np.array([MI1, MI2, MI3, MI4, MI5, ME1, ME2, ME3, ME4, ME5])  # noqa: N806
                Mu_v = np.absolute(Mu_v)  # noqa: N806

                nbH, nbB, db, As, fiPn, fiMn, Mn_i, d, dist, ro, Mu_i = AsColumn()  # noqa: N806

                VID = EleForceD[1]  # noqa: N806
                VIL = EleForceDL[1] - VID  # noqa: N806
                VIE = EleForceDLE[1] - VID - VIL  # noqa: N806
                VID, VIL, VIE = abs(VID), abs(VIL), abs(VIE)  # noqa: N806

                Mu_is = Mu_i[[1, 2, 3, 4, 6, 7, 8, 9]]  # noqa: N806
                Mn_max = np.max(Mu_is)  # Momento maximo de todas las combo sismicas  # noqa: N806
                VI1, VI2, VI3, VI4, VI5 = Combo_ACI(VID, VIL, VIE)  # noqa: N806

                VI6 = 2.0 * Mn_max / Ele.LEle  # noqa: N806
                VI7 = 1.2 * VID + 1.0 * VIL + Omo * VIE  # noqa: N806
                VI8 = 1.2 * VID + 1.0 * VIL - Omo * VIE  # noqa: N806
                VI9 = 0.9 * VID + Omo * VIE  # noqa: N806
                VI10 = 0.9 * VID - Omo * VIE  # noqa: N806

                VUa = max([VI1, VI2, VI3, VI4, VI5])  # noqa: N806
                VUb = VI6  # noqa: N806
                VUc = max([VI7, VI8, VI9, VI10])  # noqa: N806

                Vu = max([VUa, min([VUb, VUc])])  # Cortante maximo de diseño  # noqa: N806
                sst, nsB, nsH = AvColumn()  # noqa: N806
                DataColDesing.append(
                    ColDesing(
                        Ele.EleTag,
                        b,
                        h,
                        nbH,
                        nbB,
                        db,
                        As,
                        Pu_v,
                        Mu_v,
                        fiPn,
                        fiMn,
                        Mn_i,
                        d,
                        dist,
                        ro,
                        Mu_i,
                        sst,
                        nsB,
                        nsH,
                        Ele.Nod_ini,
                        Ele.Nod_end,
                    )
                )

            # self.ui.tbl_data_design_columns.setRowCount(0)  # noqa: ERA001
            # data_columns_table(self)  # noqa: ERA001
            # self.ui.tabWidget.setCurrentIndex(1)  # noqa: ERA001

    # Creation of the nonlinear model
    def CreateNLM(self, rootSIM, outputLogger, preparePushover):  # noqa: ANN001, ANN201, C901, N802, N803, D102, PLR0912, PLR0915
        def __init__(rootSIM):  # noqa: ANN001, ANN202, N803, N807
            self.rootSIM = rootSIM
            self.outputLogger = outputLogger

        global T1m, T2m, EleCol, EleBeam  # noqa: PLW0603

        # Validation of beam and column design table data
        def validate_data(self):  # noqa: ANN001, ANN202, ARG001
            cover = 4 * cm
            dst = 3 / 8 * inch

            for DC in DataColDesing:  # noqa: N806
                dp = cover + dst + 0.5 * DC.db
                DC.dist = np.linspace(dp, DC.h - dp, DC.nbH)
                Ab = pi * DC.db**2.0 / 4.0  # noqa: N806
                DC.As = np.hstack(
                    [DC.nbB * Ab, np.ones(DC.nbH - 2) * 2 * Ab, DC.nbB * Ab]
                )

                # print("DC.EleTag",DC.EleTag)  # noqa: ERA001
                # print("DC.nbH",DC.nbH)  # noqa: ERA001
                # print("DC.db",DC.db)  # noqa: ERA001
                # print("Ab",Ab)  # noqa: ERA001
                # print("DC.nbB",DC.nbB)  # noqa: ERA001
                # print("DC.h",DC.h)  # noqa: ERA001
                # print("DC.b",DC.b)  # noqa: ERA001
                # print("dp",dp)  # noqa: ERA001
                # print("dst",dst)  # noqa: ERA001
                # print("cover",cover)  # noqa: ERA001
                # print("DC.As",DC.As)  # noqa: ERA001
                # print("DC.dist",DC.dist)  # noqa: ERA001

        # Function: Parameters of regularized unconfined concrete
        def con_inconf_regu():  # noqa: ANN202
            fpc = -fc
            epsc0 = 2 * fpc / Ec
            Gfc = max(2.0 * (-fpc / MPa) * N / mm, 25.0 * N / mm)  # noqa: N806
            epscu = Gfc / (0.6 * fpc * phl) - 0.8 * fpc / Ec + epsc0
            fcu = 0.2 * fpc
            lambdaU = 0.10  # noqa: N806
            ft = 0.33 * sqrt(-fpc * MPa)
            Ets = ft / 0.002  # noqa: N806
            return fpc, epsc0, fcu, epscu, lambdaU, ft, Ets

        # Function: Parameters of regularized confined concrete
        def con_conf_regu(b, h, nsB, nsH, sst):  # noqa: ANN001, ANN202, N803
            fpc = -fc
            bcx = h - 2.0 * cover - dst
            bcy = b - 2.0 * cover - dst
            Asx = nsB * Ast  # noqa: N806
            Asy = nsH * Ast  # noqa: N806
            Asvt = Asx + Asy  # noqa: N806
            flx = Asvt * fy / sst / bcx
            fly = Asvt * fy / sst / bcy
            slx = bcx / (nsB - 1)
            sly = bcy / (nsH - 1)
            k2x = min(0.26 * sqrt((bcx / sst) * (bcx / slx) * (1000.0 / flx)), 1)
            k2y = min(0.26 * sqrt((bcy / sst) * (bcy / sly) * (1000.0 / fly)), 1)
            flex = k2x * flx
            fley = k2y * fly
            fle = (flex * bcx + fley * bcy) / (bcx + bcy)
            k1 = 6.7 * (fle / 1000.0) ** (-0.17)
            fcc = fc + k1 * fle
            fpcc = -fcc
            Ecc = Ec  # noqa: N806
            Gfc = max(2.0 * (-fpc / MPa) * N / mm, 25.0 * N / mm)  # noqa: N806
            K = k1 * fle / fc  # noqa: N806
            epscc0 = eo1 * (1.0 + 5.0 * K)
            Gfcc = 1.7 * Gfc  # noqa: N806
            epsccu = Gfcc / (0.6 * fpcc * phl) - 0.8 * fpcc / Ecc + epscc0
            fccu = 0.2 * fpcc
            lambdaC = 0.10  # noqa: N806
            ft = 0.33 * sqrt(-fpc * MPa)
            Ets = ft / 0.002  # noqa: N806

            #            print("**** sst",sst)  # noqa: ERA001
            #            print("**** fpc",fpc)  # noqa: ERA001
            #            print("**** bcx",bcx)  # noqa: ERA001
            #            print("**** bcy",bcy)  # noqa: ERA001
            #            print("**** Asx",Asx)  # noqa: ERA001
            #            print("**** Asy",Asy)  # noqa: ERA001
            #            print("**** Asvt",Asvt)  # noqa: ERA001
            #            print("**** flx",flx)  # noqa: ERA001
            #            print("**** fly",fly)  # noqa: ERA001
            #            print("**** slx",slx)  # noqa: ERA001
            #            print("**** sly",sly)  # noqa: ERA001
            #            print("**** k2x",k2x)  # noqa: ERA001
            #            print("**** k2y",k2y)  # noqa: ERA001
            #            print("**** flex",flex)  # noqa: ERA001
            #            print("**** fley",fley)  # noqa: ERA001
            #            print("**** fle",fle)  # noqa: ERA001
            #            print("**** k1",k1)  # noqa: ERA001
            #            print("**** fcc",fcc)  # noqa: ERA001
            #            print("**** fpcc",fpcc)  # noqa: ERA001
            #            print("**** Ecc",Ecc)  # noqa: ERA001
            #            print("**** Gfc",Gfc)  # noqa: ERA001
            #            print("**** K",K)  # noqa: ERA001
            #            print("**** epscc0",epscc0)  # noqa: ERA001
            #            print("**** Gfcc",Gfcc)  # noqa: ERA001
            #            print("**** epsccu",epsccu)  # noqa: ERA001
            #            print("**** fccu",fccu)  # noqa: ERA001
            #            print("**** lambdaC",lambdaC)  # noqa: ERA001
            #            print("**** ft",ft)  # noqa: ERA001
            #            print("**** Ets",Ets)  # noqa: ERA001

            return fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets

        # Function: Regularized steel parameters
        def steel_mat_regu():  # noqa: ANN202
            FyTestN4 = 490.0 * MPa  # noqa: N806
            FsuTestN4 = 630.0 * MPa  # noqa: N806
            epsuTestN4 = 0.10  # noqa: N806
            LgageTestN4 = 200.0 * mm  # noqa: N806
            Es = 200.0 * GPa  # noqa: N806
            FyPosN4 = FyTestN4  # noqa: N806
            epsyPosN4 = FyPosN4 / Es  # noqa: N806
            FyNegN4 = FyTestN4  # noqa: N806
            epsyNegN4 = FyNegN4 / Es  # noqa: N806
            FsuPosN4 = FsuTestN4  # noqa: N806
            epsuPosN4 = epsyPosN4 + LgageTestN4 / phl * (epsuTestN4 - epsyPosN4)  # noqa: N806
            bPosN4 = (FsuPosN4 - FyPosN4) / (Es * (epsuPosN4 - epsyPosN4))  # noqa: N806
            epsuNegN4 = min(-epsccu, epsuPosN4)  # noqa: N806
            bNegN4 = bPosN4  # noqa: N806
            # FsuNegN4 = FsuTestN4  # noqa: ERA001
            FsuNegN4 = FyNegN4 + bNegN4 * (Es * (epsuNegN4 - epsyNegN4))  # noqa: N806
            FsrPosN4 = 0.2 * FyPosN4  # noqa: N806
            epsrPosN4 = (FsuPosN4 - FsrPosN4) / Es + 1.05 * epsuPosN4  # noqa: N806, F841
            FsrNegN4 = 0.2 * FsuNegN4  # noqa: N806
            epsrNegN4 = (FsuNegN4 - FsrNegN4) / Es + 1.05 * epsuNegN4  # noqa: N806, F841
            pinchX = 0.2  # noqa: N806, F841
            pinchY = 0.8  # noqa: N806, F841
            damage1 = 0.0  # noqa: F841
            damage2 = 0.0  # noqa: F841
            beta = 0.0  # noqa: F841
            # op.uniaxialMaterial('Hysteretic', Ele.EleTag * 6 + 4 + pos, FyPosN4, epsyPosN4, FsuPosN4, epsuPosN4  # noqa: E501
            #                     , FsrPosN4, epsrPosN4, -FyNegN4, -epsyNegN4, -FsuNegN4, -epsuNegN4, -FsrNegN4  # noqa: E501
            #                     , -epsrNegN4, pinchX, pinchY, damage1, damage2, beta)  # noqa: E501

            SteelN4Mat = Ele.EleTag * 6 + 4 + pos  # noqa: N806
            SteelMPFTag = 1e6 * SteelN4Mat  # noqa: N806
            R0 = 20.0  # noqa: N806
            cR1 = 0.925  # noqa: N806
            cR2 = 0.15  # noqa: N806
            a1 = 0.0
            a2 = 1.0
            a3 = 0.0
            a4 = 0.0
            print(  # noqa: T201
                'SteelMPF',
                int(SteelMPFTag),
                FyPosN4,
                FyNegN4,
                Es,
                bPosN4,
                bNegN4,
                R0,
                cR1,
                cR2,
                a1,
                a2,
                a3,
                a4,
            )
            op.uniaxialMaterial(
                'SteelMPF',
                SteelMPFTag,
                FyPosN4,
                FyNegN4,
                Es,
                bPosN4,
                bNegN4,
                R0,
                cR1,
                cR2,
                a1,
                a2,
                a3,
                a4,
            )
            outputLogger.add_array(
                [
                    'uniaxialMaterial',
                    'SteelMPF',
                    int(SteelMPFTag),
                    FyPosN4,
                    FyNegN4,
                    Es,
                    bPosN4,
                    bNegN4,
                    R0,
                    cR1,
                    cR2,
                    a1,
                    a2,
                    a3,
                    a4,
                ]
            )

            print(  # noqa: T201
                'MinMax',
                int(SteelN4Mat),
                int(SteelMPFTag),
                '-min',
                -epsuNegN4,
                '-max',
                epsuPosN4,
            )
            op.uniaxialMaterial(
                'MinMax',
                SteelN4Mat,
                SteelMPFTag,
                '-min',
                -epsuNegN4,
                '-max',
                epsuPosN4,
            )
            outputLogger.add_array(
                [
                    'uniaxialMaterial',
                    'MinMax',
                    int(SteelN4Mat),
                    int(SteelMPFTag),
                    '-min',
                    -epsuNegN4,
                    '-max',
                    epsuPosN4,
                ]
            )

        # Function: Parameters of non-regularized confined concrete
        def con_conf(b, h, nsB, nsH, sst):  # noqa: ANN001, ANN202, N803
            fpc = -fc
            bcx = h - 2.0 * cover - dst
            bcy = b - 2.0 * cover - dst
            Asx = nsB * Ast  # noqa: N806
            Asy = nsH * Ast  # noqa: N806
            Asvt = Asx + Asy  # noqa: N806
            flx = Asvt * fy / sst / bcx
            fly = Asvt * fy / sst / bcy
            slx = bcx / (nsB - 1)
            sly = bcy / (nsH - 1)
            k2x = min(0.26 * sqrt((bcx / sst) * (bcx / slx) * (1000.0 / flx)), 1)
            k2y = min(0.26 * sqrt((bcy / sst) * (bcy / sly) * (1000.0 / fly)), 1)
            flex = k2x * flx
            fley = k2y * fly
            fle = (flex * bcx + fley * bcy) / (bcx + bcy)
            k1 = 6.7 * (fle / 1000.0) ** (-0.17)
            fcc = fc + k1 * fle
            fpcc = -fcc
            K = k1 * fle / fc  # noqa: N806
            epscc0 = eo1 * (1.0 + 5.0 * K)
            rov = Asvt / sst / (bcx + bcy)
            e85 = 260 * rov * epscc0 + eo85
            epsccu = (e85 - epscc0) * (0.2 * fcc - fcc) / (0.85 * fcc - fcc) + epscc0
            fccu = 0.2 * fpcc
            lambdaC = 0.10  # noqa: N806
            ft = 0.33 * sqrt(-fpc * MPa)
            Ets = ft / 0.002  # noqa: N806
            return fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets

        # Function: Parameters of non-regularized steel
        def steel_mat():  # noqa: ANN202
            FyTestN4 = 490.0 * MPa  # noqa: N806
            FsuTestN4 = 630.0 * MPa  # noqa: N806
            epsuTestN4 = 0.10  # noqa: N806
            LgageTestN4 = phl  # noqa: N806
            Es = 200.0 * GPa  # noqa: N806
            FyPosN4 = FyTestN4  # noqa: N806
            epsyPosN4 = FyPosN4 / Es  # noqa: N806
            FyNegN4 = FyTestN4  # noqa: N806
            epsyNegN4 = FyNegN4 / Es  # noqa: N806
            FsuPosN4 = FsuTestN4  # noqa: N806
            epsuPosN4 = epsyPosN4 + LgageTestN4 / phl * (epsuTestN4 - epsyPosN4)  # noqa: N806
            bPosN4 = (FsuPosN4 - FyPosN4) / (Es * (epsuPosN4 - epsyPosN4))  # noqa: N806
            epsuNegN4 = min(-epsccu, epsuPosN4)  # noqa: N806
            bNegN4 = bPosN4  # noqa: N806
            # FsuNegN4 = FsuTestN4  # noqa: ERA001
            FsuNegN4 = FyNegN4 + bNegN4 * (Es * (epsuNegN4 - epsyNegN4))  # noqa: N806
            FsrPosN4 = 0.2 * FyPosN4  # noqa: N806
            epsrPosN4 = (FsuPosN4 - FsrPosN4) / Es + 1.05 * epsuPosN4  # noqa: N806, F841
            FsrNegN4 = 0.2 * FsuNegN4  # noqa: N806
            epsrNegN4 = (FsuNegN4 - FsrNegN4) / Es + 1.05 * epsuNegN4  # noqa: N806, F841
            pinchX = 0.2  # noqa: N806, F841
            pinchY = 0.8  # noqa: N806, F841
            damage1 = 0.0  # noqa: F841
            damage2 = 0.0  # noqa: F841
            beta = 0.0  # noqa: F841
            # op.uniaxialMaterial('Hysteretic', Ele.EleTag * 6 + 4 + pos, FyPosN4, epsyPosN4, FsuPosN4, epsuPosN4  # noqa: E501
            #                     , FsrPosN4, epsrPosN4, -FyNegN4, -epsyNegN4, -FsuNegN4, -epsuNegN4, -FsrNegN4  # noqa: E501
            #                     , -epsrNegN4, pinchX, pinchY, damage1, damage2, beta)  # noqa: E501

            SteelN4Mat = Ele.EleTag * 6 + 4 + pos  # noqa: N806
            SteelMPFTag = 1e6 * SteelN4Mat  # noqa: N806
            R0 = 20.0  # noqa: N806
            cR1 = 0.925  # noqa: N806
            cR2 = 0.15  # noqa: N806
            a1 = 0.0
            a2 = 1.0
            a3 = 0.0
            a4 = 0.0
            print(  # noqa: T201
                'SteelMPF',
                int(SteelMPFTag),
                FyPosN4,
                FyNegN4,
                Es,
                bPosN4,
                bNegN4,
                R0,
                cR1,
                cR2,
                a1,
                a2,
                a3,
                a4,
            )
            op.uniaxialMaterial(
                'SteelMPF',
                SteelMPFTag,
                FyPosN4,
                FyNegN4,
                Es,
                bPosN4,
                bNegN4,
                R0,
                cR1,
                cR2,
                a1,
                a2,
                a3,
                a4,
            )
            outputLogger.add_array(
                [
                    'uniaxialMaterial',
                    'SteelMPF',
                    int(SteelMPFTag),
                    FyPosN4,
                    FyNegN4,
                    Es,
                    bPosN4,
                    bNegN4,
                    R0,
                    cR1,
                    cR2,
                    a1,
                    a2,
                    a3,
                    a4,
                ]
            )

            print(  # noqa: T201
                'MinMax',
                int(SteelN4Mat),
                int(SteelMPFTag),
                '-min',
                -epsuNegN4,
                '-max',
                epsuPosN4,
            )
            op.uniaxialMaterial(
                'MinMax',
                SteelN4Mat,
                SteelMPFTag,
                '-min',
                -epsuNegN4,
                '-max',
                epsuPosN4,
            )
            outputLogger.add_array(
                [
                    'uniaxialMaterial',
                    'MinMax',
                    int(SteelN4Mat),
                    int(SteelMPFTag),
                    '-min',
                    -epsuNegN4,
                    '-max',
                    epsuPosN4,
                ]
            )

        # Function: Creation of fibers in beams
        def fiber_beam(Ast, Asb, pos):  # noqa: ANN001, ANN202, N803
            op.section('Fiber', Ele.EleTag * 2 + pos)
            op.patch(
                'rect',
                Ele.EleTag * 6 + pos,
                10,
                1,
                -y2 + dp,
                -z2 + dp,
                y2 - dp,
                z2 - dp,
            )
            op.patch(
                'rect',
                Ele.EleTag * 6 + 2 + pos,
                10,
                1,
                -y2 + dp,
                z2 - dp,
                y2 - dp,
                z2,
            )
            op.patch(
                'rect',
                Ele.EleTag * 6 + 2 + pos,
                10,
                1,
                -y2 + dp,
                -z2,
                y2 - dp,
                -z2 + dp,
            )
            op.patch('rect', Ele.EleTag * 6 + 2 + pos, 2, 1, -y2, -z2, -y2 + dp, z2)
            op.patch('rect', Ele.EleTag * 6 + 2 + pos, 2, 1, y2 - dp, -z2, y2, z2)
            print(  # noqa: T201
                'BeamL',
                Ele.EleTag * 6 + 4 + pos,
                1,
                Ast,
                y2 - dp,
                z2 - dp,
                y2 - dp,
                -z2 + dp,
            )
            op.layer(
                'straight',
                Ele.EleTag * 6 + 4 + pos,
                1,
                Ast,
                y2 - dp,
                z2 - dp,
                y2 - dp,
                -z2 + dp,
            )
            print(  # noqa: T201
                'BeamR',
                Ele.EleTag * 6 + 4 + pos,
                1,
                Asb,
                -y2 + dp,
                z2 - dp,
                -y2 + dp,
                -z2 + dp,
            )
            op.layer(
                'straight',
                Ele.EleTag * 6 + 4 + pos,
                1,
                Asb,
                -y2 + dp,
                z2 - dp,
                -y2 + dp,
                -z2 + dp,
            )

            outputLogger.add_line('# Creating fibres in beam')

            outputLogger.add_array(['section', 'Fiber', Ele.EleTag * 2 + pos, '{'])
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6 + pos,
                    10,
                    1,
                    -y2 + dp,
                    -z2 + dp,
                    y2 - dp,
                    z2 - dp,
                ]
            )
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6 + 2 + pos,
                    10,
                    1,
                    -y2 + dp,
                    z2 - dp,
                    y2 - dp,
                    z2,
                ]
            )
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6 + 2 + pos,
                    10,
                    1,
                    -y2 + dp,
                    -z2,
                    y2 - dp,
                    -z2 + dp,
                ]
            )
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6 + 2 + pos,
                    2,
                    1,
                    -y2,
                    -z2,
                    -y2 + dp,
                    z2,
                ]
            )
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6 + 2 + pos,
                    2,
                    1,
                    y2 - dp,
                    -z2,
                    y2,
                    z2,
                ]
            )
            outputLogger.add_array(
                [
                    'layer',
                    'straight',
                    Ele.EleTag * 6 + 4 + pos,
                    1,
                    Ast,
                    y2 - dp,
                    z2 - dp,
                    y2 - dp,
                    -z2 + dp,
                ]
            )
            outputLogger.add_array(
                [
                    'layer',
                    'straight',
                    Ele.EleTag * 6 + 4 + pos,
                    1,
                    Asb,
                    -y2 + dp,
                    z2 - dp,
                    -y2 + dp,
                    -z2 + dp,
                ]
            )
            outputLogger.add_line('}')

        validate_data(self)
        op.wipe()  # The models is restarted in opensees
        outputLogger.add_line('wipe;')

        op.model('Basic', '-ndm', 2, '-ndf', 3)
        outputLogger.add_array(['model', 'Basic', '-ndm', 2, '-ndf', 3])

        outputLogger.add_line('# Create the nodes')

        for node in ListNodes:
            op.node(int(node[0]), int(node[1]), int(node[2]))
            outputLogger.add_array(
                ['node', int(node[0]), int(node[1]), int(node[2])]
            )

            if node[2] == 0.0:
                op.fix(int(node[0]), 1, 1, 1)
                outputLogger.add_array(['fix', int(node[0]), 1, 1, 1])
            if node[2] > 0 and node[1] == 0:
                MasterNode = node[0]  # noqa: N806
            if node[2] > 0 and node[1] != 0:
                op.equalDOF(int(MasterNode), int(node[0]), 1)
                outputLogger.add_array(
                    ['equalDOF', int(MasterNode), int(node[0]), 1]
                )

        cover = 4 * cm
        dst = 3 / 8 * inch
        Ast = pi * dst**2 / 4.0  # area de la barra del estribo  # noqa: N806

        # creacion de columnas
        # HBeam = float(self.ui.HBeam.text())  # noqa: ERA001
        HBeam = float(rootSIM['BeamDepth'])  # noqa: N806

        # HColi = float(self.ui.HColi.text())  # Column inside Depth  # noqa: ERA001
        HColi = float(rootSIM['IntColDepth'])  # noqa: N806

        # HCole = float(self.ui.HCole.text())  # Column outside Depth  # noqa: ERA001
        HCole = float(rootSIM['ExtColDepth'])  # noqa: N806

        # fy = float(self.ui.fy.text()) * MPa  # noqa: ERA001
        fy = float(rootSIM['FySteel']) * MPa

        Es = 200.0 * GPa  # noqa: N806, F841

        # fcB = float(self.ui.fcB.text()) * MPa  # noqa: ERA001
        fcB = float(rootSIM['BeamFpc']) * MPa  # noqa: N806

        # fcC = float(self.ui.fcC.text()) * MPa  # noqa: ERA001
        fcC = float(rootSIM['ColFpc']) * MPa  # noqa: N806

        op.geomTransf('PDelta', 1, '-jntOffset', 0, 0, 0, -HBeam / 2)
        op.geomTransf('PDelta', 2, '-jntOffset', 0, HBeam / 2, 0, -HBeam / 2)
        op.geomTransf(
            'Corotational', 3, '-jntOffset', HColi / 2.0, 0, -HColi / 2.0, 0
        )
        op.geomTransf(
            'Corotational', 4, '-jntOffset', HCole / 2.0, 0, -HColi / 2.0, 0
        )
        op.geomTransf(
            'Corotational', 5, '-jntOffset', HColi / 2.0, 0, -HCole / 2.0, 0
        )

        outputLogger.add_line('# Define the geometric transformations')

        outputLogger.add_array(
            ['geomTransf', 'PDelta', 1, '-jntOffset', 0, 0, 0, -HBeam / 2]
        )
        outputLogger.add_array(
            ['geomTransf', 'PDelta', 2, '-jntOffset', 0, HBeam / 2, 0, -HBeam / 2]
        )
        outputLogger.add_array(
            [
                'geomTransf',
                'Corotational',
                3,
                '-jntOffset',
                HColi / 2.0,
                0,
                -HColi / 2.0,
                0,
            ]
        )
        outputLogger.add_array(
            [
                'geomTransf',
                'Corotational',
                4,
                '-jntOffset',
                HCole / 2.0,
                0,
                -HColi / 2.0,
                0,
            ]
        )
        outputLogger.add_array(
            [
                'geomTransf',
                'Corotational',
                5,
                '-jntOffset',
                HColi / 2.0,
                0,
                -HCole / 2.0,
                0,
            ]
        )

        EleCol = []
        EleBeam = []
        for Ele in Elements:  # noqa: N806
            if ListNodes[Ele.Nod_ini, 1] == ListNodes[Ele.Nod_end, 1]:
                EleCol.append(Ele)
            else:
                EleBeam.append(Ele)

        platicHingeOpt = int(rootSIM['PlasticHingeOpt'])  # noqa: N806
        includeRegularization = bool(rootSIM['IncludeRegularization'])  # noqa: N806

        # print("platicHingeOpt",platicHingeOpt)  # noqa: ERA001
        # print("includeRegularization",includeRegularization)  # noqa: ERA001

        # Creation of non-linear elements (beams and columns)
        eo1, eo85, eo20, lambdaU = -0.002, -0.0038, -0.006, 0.1  # noqa: N806
        for Ele, DC in zip(EleCol, DataColDesing):  # noqa: N806
            outputLogger.add_line(
                '# Creating materials and elements for column ' + str(DC.EleTag)
            )

            fc, Ec = fcC, Ele.EcEle  # noqa: N806
            if platicHingeOpt == 1:
                phl = 0.5 * DC.h
            elif platicHingeOpt == 2:  # noqa: PLR2004
                phl = 0.08 * Ele.LEle + 0.022 * fy / MPa * DC.db / mm
            elif platicHingeOpt == 3:  # noqa: PLR2004
                phl = 0.05 * Ele.LEle + 0.1 * fy / MPa * DC.db / mm / sqrt(fc * MPa)

            if includeRegularization == True:  # noqa: E712
                fpc, epsc0, fcu, epscu, lambdaU, ft, Ets = con_inconf_regu()  # noqa: N806
                print(  # noqa: T201
                    'Concrete02',
                    Ele.EleTag * 6,
                    fpc,
                    epsc0,
                    fcu,
                    epscu,
                    lambdaU,
                    ft,
                    Ets,
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6,
                    fpc,
                    epsc0,
                    fcu,
                    epscu,
                    lambdaU,
                    ft,
                    Ets,
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 1,
                    fpc,
                    epsc0,
                    fcu,
                    epscu,
                    lambdaU,
                    ft,
                    Ets,
                )

                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6,
                        fpc,
                        epsc0,
                        fcu,
                        epscu,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 1,
                        fpc,
                        epsc0,
                        fcu,
                        epscu,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )

                fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets = con_conf_regu(  # noqa: N806
                    DC.b, DC.h, DC.nsB, DC.nsH, DC.sst
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 2,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 3,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )

                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 2,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 3,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )

                pos = 0
                steel_mat_regu()
                pos = 1
                steel_mat_regu()
            # No regularization
            else:
                ft = 0.33 * sqrt(fcC * MPa)
                Ets = ft / 0.002  # noqa: N806
                print(  # noqa: T201
                    'Concrete02',
                    Ele.EleTag * 6,
                    -fcC,
                    eo1,
                    -0.2 * fcC,
                    eo20,
                    lambdaU,
                    ft,
                    Ets,
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6,
                    -fcC,
                    eo1,
                    -0.2 * fcC,
                    eo20,
                    lambdaU,
                    ft,
                    Ets,
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 1,
                    -fcC,
                    eo1,
                    -0.2 * fcC,
                    eo20,
                    lambdaU,
                    ft,
                    Ets,
                )

                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6,
                        -fcC,
                        eo1,
                        -0.2 * fcC,
                        eo20,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 1,
                        -fcC,
                        eo1,
                        -0.2 * fcC,
                        eo20,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )

                fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets = con_conf(  # noqa: N806
                    DC.b, DC.h, DC.nsB, DC.nsH, DC.sst
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 2,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 3,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )

                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 2,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 3,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )

                pos = 0
                steel_mat()
                pos = 1
                steel_mat()

            dp = DC.dist[0]
            y1 = DC.h / 2.0
            z1 = DC.b / 2.0

            outputLogger.add_line(
                '# Creating sections and fibres for element ' + str(Ele.EleTag)
            )

            op.section('Fiber', Ele.EleTag)
            op.patch(
                'rect',
                Ele.EleTag * 6 + 2,
                10,
                1,
                -y1 + dp,
                -z1 + dp,
                y1 - dp,
                z1 - dp,
            )
            op.patch('rect', Ele.EleTag * 6, 10, 1, -y1 + dp, z1 - dp, y1 - dp, z1)
            op.patch('rect', Ele.EleTag * 6, 10, 1, -y1 + dp, -z1, y1 - dp, -z1 + dp)
            op.patch('rect', Ele.EleTag * 6, 2, 1, -y1, -z1, -y1 + dp, z1)
            op.patch('rect', Ele.EleTag * 6, 2, 1, y1 - dp, -z1, y1, z1)

            outputLogger.add_array(['section', 'Fiber', int(Ele.EleTag), '{'])
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6 + 2,
                    10,
                    1,
                    -y1 + dp,
                    -z1 + dp,
                    y1 - dp,
                    z1 - dp,
                ]
            )
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6,
                    10,
                    1,
                    -y1 + dp,
                    z1 - dp,
                    y1 - dp,
                    z1,
                ]
            )
            outputLogger.add_array(
                [
                    'patch',
                    'rect',
                    Ele.EleTag * 6,
                    10,
                    1,
                    -y1 + dp,
                    -z1,
                    y1 - dp,
                    -z1 + dp,
                ]
            )
            outputLogger.add_array(
                ['patch', 'rect', Ele.EleTag * 6, 2, 1, -y1, -z1, -y1 + dp, z1]
            )
            outputLogger.add_array(
                ['patch', 'rect', Ele.EleTag * 6, 2, 1, y1 - dp, -z1, y1, z1]
            )

            for dist, As in zip(DC.dist, DC.As):  # noqa: N806
                print(  # noqa: T201
                    'Col ',
                    Ele.EleTag * 6 + 4,
                    1,
                    As,
                    -y1 + dist,
                    z1 - dp,
                    -y1 + dist,
                    -z1 + dp,
                )
                op.layer(
                    'straight',
                    Ele.EleTag * 6 + 4,
                    1,
                    As,
                    -y1 + dist,
                    z1 - dp,
                    -y1 + dist,
                    -z1 + dp,
                )
                outputLogger.add_array(
                    [
                        'layer',
                        'straight',
                        Ele.EleTag * 6 + 4,
                        1,
                        As,
                        -y1 + dist,
                        z1 - dp,
                        -y1 + dist,
                        -z1 + dp,
                    ]
                )

            outputLogger.add_line('}')

            MassDens = Ele.AEle * GConc / g  # noqa: N806
            op.beamIntegration(
                'HingeRadau',
                Ele.EleTag,
                Ele.EleTag,
                phl,
                Ele.EleTag,
                phl,
                Ele.EleTag,
            )

            # outputLogger.add_array(['beamIntegration','HingeRadau', Ele.EleTag, Ele.EleTag, phl, Ele.EleTag, phl, Ele.EleTag])  # noqa: ERA001, E501

            op.element(
                'forceBeamColumn',
                Ele.EleTag,
                Ele.Nod_ini,
                Ele.Nod_end,
                Ele.ElegTr,
                Ele.EleTag,
                '-mass',
                MassDens,
            )

            intgrStr = (  # noqa: N806
                '"HingeRadau'  # noqa: ISC003
                + ' '
                + str(Ele.EleTag)
                + ' '
                + str(phl)
                + ' '
                + str(Ele.EleTag)
                + ' '
                + str(phl)
                + ' '
                + str(Ele.EleTag)
                + '"'
            )
            outputLogger.add_array(
                [
                    'element',
                    'forceBeamColumn',
                    Ele.EleTag,
                    Ele.Nod_ini,
                    Ele.Nod_end,
                    Ele.ElegTr,
                    intgrStr,
                    '-mass',
                    MassDens,
                ]
            )

        for Ele, DB in zip(EleBeam, DataBeamDesing):  # noqa: N806
            fc, Ec, nsH = fcB, Ele.EcEle, 2  # noqa: N806
            if platicHingeOpt == 1:
                phl1 = 0.5 * DB.h
                phl2 = 0.5 * DB.h
            elif platicHingeOpt == 2:  # noqa: PLR2004
                phl1 = 0.08 * Ele.LEle + 0.022 * fy / MPa * DB.db_t1 / mm
                phl2 = 0.08 * Ele.LEle + 0.022 * fy / MPa * DB.db_t2 / mm
            elif platicHingeOpt == 3:  # noqa: PLR2004
                phl1 = 0.05 * Ele.LEle + 0.1 * fy / MPa * DB.db_t1 / mm / sqrt(
                    fc * MPa
                )
                phl2 = 0.05 * Ele.LEle + 0.1 * fy / MPa * DB.db_t2 / mm / sqrt(
                    fc * MPa
                )

            outputLogger.add_line(
                '# Creating materials and elements for beam ' + str(DB.EleTag)
            )

            if includeRegularization == True:  # noqa: E712
                phl = phl1
                fpc, epsc0, fcu, epscu, lambdaU, ft, Ets = con_inconf_regu()  # noqa: N806
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6,
                    fpc,
                    epsc0,
                    fcu,
                    epscu,
                    lambdaU,
                    ft,
                    Ets,
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6,
                        fpc,
                        epsc0,
                        fcu,
                        epscu,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )

                phl = phl2
                fpc, epsc0, fcu, epscu, lambdaU, ft, Ets = con_inconf_regu()  # noqa: N806
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 1,
                    fpc,
                    epsc0,
                    fcu,
                    epscu,
                    lambdaU,
                    ft,
                    Ets,
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 1,
                        fpc,
                        epsc0,
                        fcu,
                        epscu,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )

                phl, pos = phl1, 0
                fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets = con_conf_regu(  # noqa: N806
                    DB.b, DB.h, DB.ns1, nsH, DB.ss1
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 2,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 2,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )

                steel_mat_regu()
                phl, pos = phl2, 1
                fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets = con_conf_regu(  # noqa: N806
                    DB.b, DB.h, DB.ns2, nsH, DB.ss2
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 3,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 3,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )

                steel_mat_regu()
            # No regularization
            else:
                ft = 0.33 * sqrt(fcB * MPa)
                Ets = ft / 0.002  # noqa: N806
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6,
                    -fcB,
                    eo1,
                    -0.2 * fcB,
                    eo20,
                    lambdaU,
                    ft,
                    Ets,
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 1,
                    -fcB,
                    eo1,
                    -0.2 * fcB,
                    eo20,
                    lambdaU,
                    ft,
                    Ets,
                )

                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6,
                        -fcB,
                        eo1,
                        -0.2 * fcB,
                        eo20,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 1,
                        -fcB,
                        eo1,
                        -0.2 * fcB,
                        eo20,
                        lambdaU,
                        ft,
                        Ets,
                    ]
                )

                fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets = con_conf(  # noqa: N806
                    DB.b, DB.h, DB.ns1, nsH, DB.ss1
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 2,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 2,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )

                fpcc, epscc0, fccu, epsccu, lambdaC, ft, Ets = con_conf(  # noqa: N806
                    DB.b, DB.h, DB.ns2, nsH, DB.ss2
                )
                op.uniaxialMaterial(
                    'Concrete02',
                    Ele.EleTag * 6 + 3,
                    fpcc,
                    epscc0,
                    fccu,
                    epsccu,
                    lambdaC,
                    ft,
                    Ets,
                )
                outputLogger.add_array(
                    [
                        'uniaxialMaterial',
                        'Concrete02',
                        Ele.EleTag * 6 + 3,
                        fpcc,
                        epscc0,
                        fccu,
                        epsccu,
                        lambdaC,
                        ft,
                        Ets,
                    ]
                )

                pos = 0
                steel_mat()
                pos = 1
                steel_mat()
            y2 = DB.h / 2.0
            z2 = DB.b / 2.0
            dp = DB.h - min(DB.db1, DB.dt1)
            pos = 0

            fiber_beam(DB.Ast1, DB.Asb1, pos)
            dp = DB.h - min(DB.db2, DB.dt2)
            pos = 1
            fiber_beam(DB.Ast2, DB.Asb2, pos)
            MassDens = Ele.AEle * GConc / g + WDLS / g  # noqa: N806
            op.beamIntegration(
                'HingeRadau',
                Ele.EleTag,
                Ele.EleTag * 2,
                phl1,
                Ele.EleTag * 2 + 1,
                phl2,
                Ele.EleTag * 2,
            )
            # outputLogger.add_array(['beamIntegration','HingeRadau', Ele.EleTag, Ele.EleTag * 2, phl1, Ele.EleTag * 2 + 1, phl2, Ele.EleTag * 2])  # noqa: ERA001, E501

            op.element(
                'forceBeamColumn',
                Ele.EleTag,
                Ele.Nod_ini,
                Ele.Nod_end,
                Ele.ElegTr,
                Ele.EleTag,
                '-mass',
                MassDens,
            )

            intgrStr = (  # noqa: N806
                '"HingeRadau'  # noqa: ISC003
                + ' '
                + str(Ele.EleTag * 2)
                + ' '
                + str(phl1)
                + ' '
                + str(Ele.EleTag * 2 + 1)
                + ' '
                + str(phl2)
                + ' '
                + str(Ele.EleTag * 2)
                + '"'
            )
            outputLogger.add_array(
                [
                    'element',
                    'forceBeamColumn',
                    Ele.EleTag,
                    Ele.Nod_ini,
                    Ele.Nod_end,
                    Ele.ElegTr,
                    intgrStr,
                    '-mass',
                    MassDens,
                ]
            )

        list_beams = [Ele.EleTag for Ele in EleBeam]
        list_cols = [Ele.EleTag for Ele in EleCol]
        print('list_beams =', list_beams)  # noqa: T201
        print('list_cols =', list_cols)  # noqa: T201

        print('Model Nonlinear Built')  # noqa: T201

        # KZ: gravity analysis
        outputLogger.add_array(['timeSeries Linear 1'])
        outputLogger.add_array(['pattern Plain 1 Constant {'])
        for Ele in EleCol:  # noqa: N806
            outputLogger.add_array(
                [
                    f'eleLoad -ele {Ele.EleTag} -type -beamUniform 0 {-Ele.AEle * GConc}'  # noqa: E501
                ]
            )
        for Ele in EleBeam:  # noqa: N806
            outputLogger.add_array(
                [
                    f'eleLoad -ele {Ele.EleTag} -type -beamUniform {-Ele.AEle * GConc - WDL}'  # noqa: E501
                ]
            )
        outputLogger.add_array(['}'])
        outputLogger.add_array(['set Tol 1.0e-6'])
        outputLogger.add_array(['constraints Plain'])
        outputLogger.add_array(['numberer Plain'])
        outputLogger.add_array(['system BandGeneral'])
        outputLogger.add_array(['test NormDispIncr 1e-6 100'])
        outputLogger.add_array(['algorithm KrylovNewton'])
        outputLogger.add_array(['integrator LoadControl 0.1'])
        outputLogger.add_array(['analysis Static'])
        outputLogger.add_array(['analyze 10'])
        outputLogger.add_array(['loadConst -time 0.0'])

        # KZ: user defined damping
        xi = rootSIM.get('dampingRatio', 0.05)
        # KZ: modes  # noqa: ERA001
        if rootSIM.get('Simulation', None) is not None:
            tmp = rootSIM.get('Simulation')
            mode1 = tmp.get('firstMode', 1)
            mode2 = tmp.get('secnondMode', 3)
        else:
            mode1 = 1
            mode2 = 3
        outputLogger.add_array([f'set nEigenI {mode1}'])
        outputLogger.add_array([f'set nEigenJ {mode2}'])
        outputLogger.add_array(['set lambdaN [eigen [expr $nEigenJ]]'])
        outputLogger.add_array(['set lambdaI [lindex $lambdaN [expr $nEigenI-1]]'])
        outputLogger.add_array(['set lambdaJ [lindex $lambdaN [expr $nEigenJ-1]]'])
        outputLogger.add_array(['set lambda1 [expr pow($lambdaI,0.5)]'])
        outputLogger.add_array(['set lambda2 [expr pow($lambdaJ,0.5)]'])
        outputLogger.add_array(['set T1 [expr 2.0*3.14/$lambda1]'])
        outputLogger.add_array(['puts "T1 = $T1"'])
        outputLogger.add_array(
            [f'set a0 [expr {xi}*2.0*$lambda1*$lambda2/($lambda1+$lambda2)]']
        )
        outputLogger.add_array([f'set a1 [expr {xi}*2.0/($lambda1+$lambda2)]'])
        outputLogger.add_array(['rayleigh $a0 0.0 $a1 0.0'])

        if preparePushover == False:  # noqa: E712
            return

        if not os.path.exists('Pushover'):  # noqa: PTH110
            os.mkdir('Pushover')  # noqa: PTH102

        # Recording of forces and deformations from nonlinear analysis
        op.recorder(
            'Element',
            '-file',
            'Pushover/beams_force_1.out',
            '-time',
            '-ele',
            *list_beams,
            'section',
            1,
            'force',
        )
        op.recorder(
            'Element',
            '-file',
            'Pushover/beams_def_1.out',
            '-time',
            '-ele',
            *list_beams,
            'section',
            1,
            'deformation',
        )
        op.recorder(
            'Element',
            '-file',
            'Pushover/beams_force_6.out',
            '-time',
            '-ele',
            *list_beams,
            'section',
            6,
            'force',
        )
        op.recorder(
            'Element',
            '-file',
            'Pushover/beams_def_6.out',
            '-time',
            '-ele',
            *list_beams,
            'section',
            6,
            'deformation',
        )
        op.recorder(
            'Element',
            '-file',
            'Pushover/cols_force_1.out',
            '-time',
            '-ele',
            *list_cols,
            'section',
            1,
            'force',
        )
        op.recorder(
            'Element',
            '-file',
            'Pushover/cols_def_1.out',
            '-time',
            '-ele',
            *list_cols,
            'section',
            1,
            'deformation',
        )
        op.recorder(
            'Element',
            '-file',
            'Pushover/cols_force_6.out',
            '-time',
            '-ele',
            *list_cols,
            'section',
            6,
            'force',
        )
        op.recorder(
            'Element',
            '-file',
            'Pushover/cols_def_6.out',
            '-time',
            '-ele',
            *list_cols,
            'section',
            6,
            'deformation',
        )
        op.recorder(
            'Node',
            '-file',
            'Pushover/HoriNodes.out',
            '-time',
            '-node',
            *ListNodes,
            '-dof',
            1,
            'disp',
        )
        op.recorder(
            'Node',
            '-file',
            'Pushover/VertNodes.out',
            '-time',
            '-node',
            *ListNodes,
            '-dof',
            2,
            'disp',
        )

        # Create a Plain load pattern for gravity loading with a Linear TimeSeries
        op.timeSeries('Linear', 1)
        op.pattern('Plain', 1, 1)
        for Ele in EleCol:  # noqa: N806
            op.eleLoad(
                '-ele', Ele.EleTag, '-type', '-beamUniform', 0, -Ele.AEle * GConc
            )
        for Ele in EleBeam:  # noqa: N806
            op.eleLoad(
                '-ele', Ele.EleTag, '-type', '-beamUniform', -Ele.AEle * GConc - WDL
            )

        Tol = 1.0e-6  # convergence tolerance for test  # noqa: N806
        op.constraints('Plain')  # how it handles boundary conditions
        op.numberer(
            'Plain'
        )  # renumber dof to minimize band-width (optimization), if you want to
        op.system(
            'BandGeneral'
        )  # how to store and solve the system of equations in the analysis
        op.test(
            'NormDispIncr', Tol, 100
        )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
        op.algorithm(
            'KrylovNewton'
        )  # use Newton solution algorithm: updates tangent stiffness at every iteration  # noqa: E501
        NstepGravity = 10  # apply gravity in 10 steps  # noqa: N806
        DGravity = 1.0 / NstepGravity  # first load increment;  # noqa: N806
        op.integrator(
            'LoadControl', DGravity
        )  # determine the next time step for an analysis
        op.analysis('Static')  # define type of analysis static or transient
        op.analyze(NstepGravity)  # apply gravity
        op.loadConst('-time', 0.0)

        # xi = 0.05  # damping ratio  # noqa: ERA001
        MpropSwitch = 1.0  # noqa: N806
        KcurrSwitch = 0.0  # noqa: N806
        KcommSwitch = 1.0  # noqa: N806
        KinitSwitch = 0.0  # noqa: N806
        nEigenI = 1  # mode 1  # noqa: N806
        nEigenI2 = 2  # mode 2  # noqa: N806
        nEigenJ = 3  # mode 3  # noqa: N806
        lambdaN = op.eigen(nEigenJ)  # eigenvalue analysis for nEigenJ modes  # noqa: N806
        lambdaI = lambdaN[nEigenI - 1]  # eigenvalue mode i  # noqa: N806
        lambdaI2 = lambdaN[nEigenI2 - 1]  # eigenvalue mode i2  # noqa: N806
        lambdaJ = lambdaN[nEigenJ - 1]  # eigenvalue mode j  # noqa: N806
        print('lambdaN ', lambdaN)  # noqa: T201
        omegaI = pow(lambdaI, 0.5)  # noqa: N806
        omegaI2 = pow(lambdaI2, 0.5)  # noqa: N806
        omegaJ = pow(lambdaJ, 0.5)  # noqa: N806
        T1m = 2.0 * pi / omegaI
        T2m = 2.0 * pi / omegaI2

        print('Ta1=', T1m, 'seg', ' Ta2=', T2m, ' seg')  # noqa: T201
        alphaM = (  # noqa: N806
            MpropSwitch * xi * (2.0 * omegaI * omegaJ) / (omegaI + omegaJ)
        )  # M-prop. damping D = alphaM*M
        betaKcurr = (  # noqa: N806
            KcurrSwitch * 2.0 * xi / (omegaI + omegaJ)
        )  # current-K      +beatKcurr*KCurrent
        betaKcomm = (  # noqa: N806
            KcommSwitch * 2.0 * xi / (omegaI + omegaJ)
        )  # last-committed K   +betaKcomm*KlastCommitt
        betaKinit = (  # noqa: N806
            KinitSwitch * 2.0 * xi / (omegaI + omegaJ)
        )  # initial-K     +beatKinit*Kini
        op.rayleigh(alphaM, betaKcurr, betaKinit, betaKcomm)  # RAYLEIGH damping

    # Pushover function
    def Pushover(self, rootSIM):  # noqa: ANN001, ANN201, C901, N802, N803, D102, PLR0912, PLR0915
        def __init__(rootSIM):  # noqa: ANN001, ANN202, N803, N807
            self.rootSIM = rootSIM

        global cbar  # noqa: PLW0602

        def singlePush1(dref, mu, ctrlNode, dispDir, nSteps):  # noqa: ANN001, ANN202, C901, N802, N803, PLR0912, PLR0915
            IOflag = 2  # noqa: N806
            testType = 'RelativeNormDispIncr'  # noqa: N806
            # set testType	EnergyIncr;					# Dont use with Penalty constraints  # noqa: E501
            # set testType	RelativeNormUnbalance;		# Dont use with Penalty constraints  # noqa: E501
            # set testType	RelativeNormDispIncr;		# Dont use with Lagrange constraints  # noqa: E501
            # set testType	RelativeTotalNormDispIncr;	# Dont use with Lagrange constraints  # noqa: E501
            # set testType	RelativeEnergyIncr;			# Dont use with Penalty constraints  # noqa: E501
            tolInit = 1.0e-6  # the initial Tolerance, so it can be referred back to  # noqa: N806
            iterInit = 50  # the initial Max Number of Iterations  # noqa: N806
            algorithmType = 'KrylovNewton'  # the algorithm type  # noqa: N806

            op.test(
                testType, tolInit, iterInit
            )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
            op.algorithm(
                algorithmType
            )  # use Newton solution algorithm: updates tangent stiffness at every iteration  # noqa: E501
            disp = dref * mu
            dU = disp / (1.0 * nSteps)  # noqa: N806
            print('dref ', dref, 'mu ', mu, 'dU ', dU, 'disp ', disp)  # noqa: T201
            op.integrator(
                'DisplacementControl', ctrlNode, dispDir, dU
            )  # determine the next time step for an analysis
            op.analysis('Static')  # define type of analysis static or transient

            # Print values
            if IOflag >= 1:
                print('singlePush: Push ', ctrlNode, ' to ', mu)  # noqa: T201

            #      the initial values to start the while loop
            ok = 0
            step = 1
            loadf = 1.0
            # This feature of disabling the possibility of having a negative loading has been included.  # noqa: E501
            # This has been adapted from a similar script by Prof. Garbaggio
            htot = op.nodeCoord(ctrlNode, 2)
            maxDriftPiso = 0.0  # noqa: N806
            VBasal_v = []  # noqa: N806
            DriftTecho_v = []  # noqa: N806
            while step <= nSteps and ok == 0 and loadf > 0:
                # self.ui.progressBar.setValue(100 * step / nSteps)  # noqa: ERA001
                ok = op.analyze(1)
                loadf = op.getTime()
                temp = op.nodeDisp(ctrlNode, dispDir)
                # Print the current displacement
                if IOflag >= 2:  # noqa: PLR2004
                    print(  # noqa: T201
                        'Pushed ',
                        ctrlNode,
                        ' in ',
                        dispDir,
                        ' to ',
                        temp,
                        ' with ',
                        loadf,
                        'step',
                        step,
                    )

                # If the analysis fails, try the following changes to achieve convergence  # noqa: E501
                # Analysis will be slower in here though...
                if ok != 0:
                    print('Trying relaxed convergence..')  # noqa: T201
                    op.test(
                        testType, tolInit * 0.01, iterInit * 50
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                    ok = op.analyze(1)
                    op.test(
                        testType, tolInit, iterInit
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                if ok != 0:
                    print('Trying Newton with initial then current .')  # noqa: T201
                    op.test(
                        testType, tolInit * 0.01, iterInit * 50
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                    op.algorithm('Newton', '-initialThenCurrent')
                    ok = op.analyze(1)
                    op.algorithm(algorithmType)
                    op.test(
                        testType, tolInit, iterInit
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                if ok != 0:
                    print('Trying ModifiedNewton with initial ..')  # noqa: T201
                    op.test(
                        testType, tolInit * 0.01, iterInit * 50
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                    op.algorithm('ModifiedNewton', '-initial')
                    ok = op.analyze(1)
                    op.algorithm(algorithmType)
                    op.test(
                        testType, tolInit, iterInit
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                if ok != 0:
                    print('Trying KrylovNewton ..')  # noqa: T201
                    op.test(
                        testType, tolInit * 0.01, iterInit * 50
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                    op.algorithm('KrylovNewton')
                    ok = op.analyze(1)
                    op.algorithm(algorithmType)
                    op.test(
                        testType, tolInit, iterInit
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                if ok != 0:
                    print('Perform a Hail Mary ....')  # noqa: T201
                    op.test(
                        'FixedNumIter', iterInit
                    )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
                    ok = op.analyze(1)

                for nod_ini, nod_end in zip(
                    ListNodesDrift[:-1, 0], ListNodesDrift[1:, 0]
                ):
                    # print('nod_ini ', nod_ini, 'nod_end', nod_end)  # noqa: ERA001
                    nod_ini = int(nod_ini)  # noqa: PLW2901
                    nod_end = int(nod_end)  # noqa: PLW2901
                    pos_i = op.nodeCoord(nod_ini, 2)
                    pos_s = op.nodeCoord(nod_end, 2)
                    hpiso = pos_s - pos_i
                    desp_i = op.nodeDisp(nod_ini, 1)
                    desp_s = op.nodeDisp(nod_end, 1)
                    desp_piso = abs(desp_s - desp_i)
                    drift_piso = desp_piso / hpiso
                    if drift_piso >= maxDriftPiso:
                        maxDriftPiso = drift_piso  # noqa: N806

                VBasal = 0.0  # noqa: N806
                op.reactions()
                for node in ListNodesBasal:
                    # print('ind Basal ', node[0])  # noqa: ERA001
                    VBasal = VBasal + op.nodeReaction(node[0], 1)  # noqa: N806
                VBasal_v = np.append(VBasal_v, VBasal)  # noqa: N806
                DriftTecho = op.nodeDisp(ctrlNode, dispDir) / htot  # noqa: N806
                DriftTecho_v = np.append(DriftTecho_v, DriftTecho)  # noqa: N806
                loadf = op.getTime()
                step += 1
            maxDriftTecho = dU * step / htot  # noqa: N806
            maxDriftTecho2 = op.nodeDisp(ctrlNode, dispDir) / htot  # noqa: N806

            if ok != 0:
                print('DispControl Analysis FAILED')  # noqa: T201
            else:
                print('DispControl Analysis SUCCESSFUL')  # noqa: T201
            if loadf <= 0:
                print('Stopped because of Load factor below zero: ', loadf)  # noqa: T201
            #    if PrintFlag == 0:
            #        os.remove("singlePush.txt")  # noqa: ERA001
            #        print singlePush.txt
            return (
                maxDriftPiso,
                maxDriftTecho,
                maxDriftTecho2,
                VBasal_v,
                DriftTecho_v,
            )

        # Pushover function varying tests and algorithms
        def singlePush(dref, mu, ctrlNode, dispDir, nSteps):  # noqa: ANN001, ANN202, C901, N802, N803, PLR0912, PLR0915
            # --------------------------------------------------
            # Description of Parameters
            # --------------------------------------------------
            # dref:			Reference displacement to which cycles are run. Corresponds to yield or equivalent other, such as 1mm  # noqa: E501
            # mu:			Multiple of dref to which the push is run. So pushover can be run to a specifived ductility or displacement  # noqa: E501
            # ctrlNode:		Node to control with the displacement integrator.
            # dispDir:		DOF the loading is applied.
            # nSteps:		Number of steps.
            # IOflag:		Option to print details on screen. 2 for print of each step, 1 for basic info (default), 0 for off  # noqa: E501
            # ---------------------------------------------------
            test = {
                1: 'NormDispIncr',
                2: 'RelativeEnergyIncr',
                3: 'EnergyIncr',
                4: 'RelativeNormUnbalance',
                5: 'RelativeNormDispIncr',
                6: 'NormUnbalance',
                7: 'FixedNumIter',
            }
            alg = {
                1: 'KrylovNewton',
                2: 'SecantNewton',
                3: 'ModifiedNewton',
                4: 'RaphsonNewton',
                5: 'PeriodicNewton',
                6: 'BFGS',
                7: 'Broyden',
                8: 'NewtonLineSearch',
            }

            # test = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 3:'EnergyIncr'}  # noqa: ERA001
            # alg = {1:'KrylovNewton', 2:'ModifiedNewton'}  # noqa: ERA001

            IOflag = 2  # noqa: N806
            PrintFlag = 0  # noqa: N806, F841
            testType = 'RelativeNormDispIncr'  # Dont use with Penalty constraints  # noqa: N806

            tolInit = 1.0e-7  # the initial Tolerance, so it can be referred back to  # noqa: N806
            iterInit = 50  # the initial Max Number of Iterations  # noqa: N806
            algorithmType = 'KrylovNewton'  # the algorithm type  # noqa: N806
            #      	algorithmType Newton;		#      the algorithm type
            #      	algorithmType Newton;		#      the algorithm type

            # op.constraints('Transformation') # how it handles boundary conditions  # noqa: ERA001
            # op.numberer('RCM')    # renumber dof to minimize band-width (optimization), if you want to  # noqa: ERA001, E501
            # op.system('BandGeneral') # how to store and solve the system of equations in the analysis  # noqa: ERA001, E501

            op.test(
                testType, tolInit, iterInit
            )  # determine if convergence has been achieved at the end of an iteration step  # noqa: E501
            op.algorithm(
                algorithmType
            )  # use Newton solution algorithm: updates tangent stiffness at every iteration  # noqa: E501
            disp = dref * mu
            dU = disp / (1.0 * nSteps)  # noqa: N806
            print(  # noqa: T201
                'dref ', dref, 'mu ', mu, 'dU ', dU, 'disp ', disp, 'nSteps ', nSteps
            )
            op.integrator(
                'DisplacementControl', ctrlNode, dispDir, dU
            )  # determine the next time step for an analysis
            op.analysis('Static')  # defivne type of analysis static or transient

            # Print values
            if IOflag >= 1:
                print('singlePush: Push ', ctrlNode, ' to ', mu)  # noqa: T201

            #      the initial values to start the while loop
            ok = 0
            step = 1
            loadf = 1.0
            # This feature of disabling the possibility of having a negative loading has been included.  # noqa: E501
            # This has been adapted from a similar script by Prof. Garbaggio
            maxDriftPiso = 0.0  # noqa: N806
            htot = op.nodeCoord(ctrlNode, 2)
            VBasal_v = []  # noqa: N806
            DriftTecho_v = []  # noqa: N806
            # factor_v = np.array([1,0.75,0.5,0.25,0.1,2,3,5,10])  # noqa: ERA001
            # fact_v = np.array([50,100,500])  # noqa: ERA001
            # factor = 100  # noqa: ERA001
            # fact = 1.  # noqa: ERA001
            while step <= nSteps and ok == 0 and loadf > 0:
                # self.ui.progressBar.setValue(100 * step / nSteps)  # noqa: ERA001
                ok = op.analyze(1)
                loadf = op.getTime()
                temp = op.nodeDisp(ctrlNode, dispDir)
                if IOflag >= 2:  # noqa: PLR2004
                    print(  # noqa: T201
                        'Pushed ',
                        ctrlNode,
                        ' in ',
                        dispDir,
                        ' to ',
                        temp,
                        ' with ',
                        loadf,
                        'step ',
                        step,
                    )
                # for factor in factor_v:
                # op.integrator('DisplacementControl',ctrlNode,dispDir,factor*dU)  # determine the next time step for an analysis  # noqa: ERA001, E501
                # for fact in fact_v:
                for j in alg:
                    for i in test:
                        for fact in [1, 20, 50]:
                            if ok != 0 and j >= 4 and i != 7:  # noqa: PLR2004
                                # print('Trying ',str(alg[j]))  # noqa: ERA001
                                op.test(test[i], tolInit * 0.01, iterInit * fact)
                                op.algorithm(alg[j])
                                ok = op.analyze(1)
                                op.algorithm(algorithmType)
                                op.test(testType, tolInit, iterInit)
                            elif ok != 0 and j < 4 and i != 7:  # noqa: PLR2004
                                # print('Trying ',str(alg[j]))  # noqa: ERA001
                                op.test(test[i], tolInit, iterInit * fact)
                                op.algorithm(alg[j], '-initial')
                                ok = op.analyze(1)
                                op.algorithm(algorithmType)
                                op.test(testType, tolInit, iterInit)
                            if ok == 0:
                                break
                        if ok != 0 and i == 7:  # noqa: PLR2004
                            op.test(test[i], iterInit)
                            op.algorithm(alg[j])
                            ok = op.analyze(1)
                        if ok == 0:
                            break
                    if ok == 0:
                        break
                    # if ok == 0:
                    #     break  # noqa: ERA001
                    # if ok == 0:
                    #     break  # noqa: ERA001
                # op.integrator('DisplacementControl',ctrlNode,dispDir,dU)  # determine the next time step for an analysis  # noqa: ERA001, E501
                # Calculation of maximum Drift between floors
                for nod_ini, nod_end in zip(
                    ListNodesDrift[:-1, 0], ListNodesDrift[1:, 0]
                ):
                    # print('nod_ini ', nod_ini, 'nod_end', nod_end)  # noqa: ERA001
                    nod_ini = int(nod_ini)  # noqa: PLW2901
                    nod_end = int(nod_end)  # noqa: PLW2901
                    pos_i = op.nodeCoord(nod_ini, 2)
                    pos_s = op.nodeCoord(nod_end, 2)
                    hpiso = pos_s - pos_i
                    desp_i = op.nodeDisp(nod_ini, 1)
                    desp_s = op.nodeDisp(nod_end, 1)
                    desp_piso = abs(desp_s - desp_i)
                    drift_piso = desp_piso / hpiso
                    if drift_piso >= maxDriftPiso:
                        maxDriftPiso = drift_piso  # noqa: N806

                VBasal = 0.0  # noqa: N806
                op.reactions()
                for node in ListNodesBasal:
                    # print('ind Basal ', node[0])  # noqa: ERA001
                    VBasal = VBasal + op.nodeReaction(node[0], 1)  # noqa: N806
                VBasal_v = np.append(VBasal_v, VBasal)  # noqa: N806
                DriftTecho = op.nodeDisp(ctrlNode, dispDir) / htot  # noqa: N806
                DriftTecho_v = np.append(DriftTecho_v, DriftTecho)  # noqa: N806
                loadf = op.getTime()
                step += 1
            maxDriftTecho = dU * step / htot  # noqa: N806
            maxDriftTecho2 = op.nodeDisp(ctrlNode, dispDir) / htot  # noqa: N806

            if ok != 0:
                print('DispControl Analysis FAILED')  # noqa: T201
            else:
                print('DispControl Analysis SUCCESSFUL')  # noqa: T201
            if loadf <= 0:
                print('Stopped because of Load factor below zero: ', loadf)  # noqa: T201
            #    if PrintFlag == 0:
            #        os.remove("singlePush.txt")  # noqa: ERA001
            #        print singlePush.txt
            return (
                maxDriftPiso,
                maxDriftTecho,
                maxDriftTecho2,
                VBasal_v,
                DriftTecho_v,
            )

        ListNodesDrift = ListNodes[np.where(ListNodes[:, 1] == 0.0)]  # noqa: N806
        ListNodesBasal = ListNodes[np.where(ListNodes[:, 2] == 0.0)]  # noqa: N806
        if T1m <= 0.5:  # noqa: PLR2004
            k = 1.0
        elif T1m <= 2.5:  # noqa: PLR2004
            k = 0.75 + 0.5 * T1m
        else:
            k = 2.0

        sumH = np.sum(np.power(Loc_heigth, k))  # noqa: N806
        floors_num = len(Loc_heigth)

        # Match default example
        triangForceDist = True  # noqa: N806

        # Defining the pushover lateral distribution type
        if triangForceDist == True:  # noqa: E712
            Fp = np.power(Loc_heigth, k) / sumH  # noqa: N806
        else:
            Fp = 1.0 / floors_num * np.ones(floors_num + 1)  # noqa: N806
        print('Fp =', Fp)  # noqa: T201
        op.loadConst('-time', 0.0)
        op.timeSeries('Linear', 2)
        op.pattern('Plain', 2, 1)
        for node, fp, ind in zip(ListNodesDrift, Fp, range(floors_num)):  # noqa: B007
            op.load(int(node[0]), fp, 0.0, 0.0)

        Htotal = Loc_heigth[-1]  # noqa: N806
        #        Der_obj = float(self.ui.Der_obj.text())  # noqa: ERA001
        Der_obj = 0.04  # Match default example  # noqa: N806
        Des_obj = Der_obj * Htotal  # Desplazamiento objetivo  # noqa: N806
        #        nSteps = int(self.ui.nSteps.text())  # noqa: ERA001
        nSteps = 110  # Match default example  # noqa: N806
        dref = Des_obj / nSteps
        mu = nSteps
        IDctrlNode = int(ListNodesDrift[-1, 0])  # Node where displacement is read  # noqa: N806
        print('IDctrlNode =', IDctrlNode)  # noqa: T201
        IDctrlDOF = 1  # DOF x=1, y=2  # noqa: N806
        Tol = 1.0e-4  # Tolerance  # noqa: N806, F841

        runFastPushover = True  # noqa: N806
        if runFastPushover == True:  # noqa: E712
            maxDriftPiso, maxDriftTecho, maxDriftTecho2, VBasal_v, DriftTecho_v = (  # noqa: N806
                singlePush1(dref, mu, IDctrlNode, IDctrlDOF, nSteps)
            )
        else:
            maxDriftPiso, maxDriftTecho, maxDriftTecho2, VBasal_v, DriftTecho_v = (  # noqa: N806
                singlePush(dref, mu, IDctrlNode, IDctrlDOF, nSteps)
            )

        op.wipe()

        # Reading of forces and deflections of beams and columns from recorders
        beams_force_1 = np.loadtxt('Pushover/beams_force_1.out')
        beams_def_1 = np.loadtxt('Pushover/beams_def_1.out')
        beams_force_6 = np.loadtxt('Pushover/beams_force_6.out')
        beams_def_6 = np.loadtxt('Pushover/beams_def_6.out')
        cols_force_1 = np.loadtxt('Pushover/cols_force_1.out')
        cols_def_1 = np.loadtxt('Pushover/cols_def_1.out')
        cols_force_6 = np.loadtxt('Pushover/cols_force_6.out')
        cols_def_6 = np.loadtxt('Pushover/cols_def_6.out')
        print('cols_def_1', cols_def_1)  # noqa: T201

        # fy = float(self.ui.fy.text()) * MPa  # noqa: ERA001
        fy = float(rootSIM['FySteel']) * MPa
        print('Fy', fy)  # noqa: T201

        Es = 200.0 * GPa  # noqa: N806
        ey = fy / Es
        num_beams = len(EleBeam)
        num_cols = len(EleCol)
        CD_Beams = np.zeros([num_beams, 2])  # noqa: N806

        # Calculation of curvature ductility of beams and columns
        for ind, DB in zip(range(1, num_beams + 1), DataBeamDesing):  # noqa: N806
            ets_beam_1 = beams_def_1[:, 2 * ind - 1] + beams_def_1[:, 2 * ind] * (
                DB.dt1 - DB.h / 2
            )
            ebs_beam_1 = beams_def_1[:, 2 * ind - 1] + beams_def_1[:, 2 * ind] * (
                DB.h / 2 - DB.db1
            )
            ets_beam_6 = beams_def_6[:, 2 * ind - 1] + beams_def_6[:, 2 * ind] * (
                DB.dt2 - DB.h / 2
            )
            ebs_beam_6 = beams_def_6[:, 2 * ind - 1] + beams_def_6[:, 2 * ind] * (
                DB.h / 2 - DB.db1
            )
            es_beam_1 = np.maximum(np.absolute(ets_beam_1), np.absolute(ebs_beam_1))
            es_beam_6 = np.maximum(np.absolute(ets_beam_6), np.absolute(ebs_beam_6))
            print('es_beam_1', es_beam_1, 'es_beam_6', es_beam_6)  # noqa: T201
            if np.max(es_beam_1) <= ey:
                CD_1 = 0  # noqa: N806
            else:
                fi_1 = np.absolute(beams_def_1[:, 2 * ind])
                M_beam_1 = np.absolute(beams_force_1[:, 2 * ind])  # noqa: N806
                f = interpolate.interp1d(es_beam_1, M_beam_1)
                My_1 = f(ey)  # noqa: N806
                f = interpolate.interp1d(M_beam_1, fi_1)
                fiy_1 = f(My_1)
                CD_1 = fi_1[-1] / fiy_1  # noqa: N806
            if np.max(es_beam_6) <= ey:
                CD_6 = 0  # noqa: N806
            else:
                fi_6 = np.absolute(beams_def_6[:, 2 * ind])
                M_beam_6 = np.absolute(beams_force_6[:, 2 * ind])  # noqa: N806
                f = interpolate.interp1d(es_beam_6, M_beam_6)
                My_6 = f(ey)  # noqa: N806
                f = interpolate.interp1d(M_beam_6, fi_6)
                fiy_6 = f(My_6)
                CD_6 = fi_6[-1] / fiy_6  # noqa: N806
            CD_Beams[ind - 1, :] = [CD_1, CD_6]
            print('CD_Beams =', CD_Beams)  # noqa: T201

        CD_Cols = np.zeros([num_cols, 2])  # noqa: N806
        for ind, DC in zip(range(1, num_cols + 1), DataColDesing):  # noqa: N806
            ets_col_1 = cols_def_1[:, 2 * ind - 1] + cols_def_1[:, 2 * ind] * (
                DC.d - DC.h / 2
            )
            ebs_col_1 = cols_def_1[:, 2 * ind - 1] + cols_def_1[:, 2 * ind] * (
                DC.h / 2 - DC.d
            )
            ets_col_6 = cols_def_6[:, 2 * ind - 1] + cols_def_6[:, 2 * ind] * (
                DC.d - DC.h / 2
            )
            ebs_col_6 = cols_def_6[:, 2 * ind - 1] + cols_def_6[:, 2 * ind] * (
                DC.h / 2 - DC.d
            )
            es_col_1 = np.maximum(np.absolute(ets_col_1), np.absolute(ebs_col_1))
            es_col_6 = np.maximum(np.absolute(ets_col_6), np.absolute(ebs_col_6))
            print('es_col_1', es_col_1, 'es_col_6', es_col_6)  # noqa: T201
            if np.max(es_col_1) <= ey:
                CD_1 = 0  # noqa: N806
            else:
                fi_1 = np.absolute(cols_def_1[:, 2 * ind])
                M_col_1 = np.absolute(cols_force_1[:, 2 * ind])  # noqa: N806
                f = interpolate.interp1d(es_col_1, M_col_1)
                Mfy_1 = f(ey)  # noqa: N806
                f = interpolate.interp1d(M_col_1, fi_1)
                fify_1 = f(Mfy_1)
                My_1 = np.max(M_col_1)  # noqa: N806
                fiy_1 = My_1 / Mfy_1 * fify_1
                CD_1 = fi_1[-1] / fiy_1  # noqa: N806

            if np.max(es_col_6) <= ey:
                CD_6 = 0  # noqa: N806
            else:
                fi_6 = np.absolute(cols_def_6[:, 2 * ind])
                M_col_6 = np.absolute(cols_force_6[:, 2 * ind])  # noqa: N806
                f = interpolate.interp1d(es_col_6, M_col_6)
                Mfy_6 = f(ey)  # noqa: N806
                f = interpolate.interp1d(M_col_6, fi_6)
                fify_6 = f(Mfy_6)
                My_6 = np.max(M_col_6)  # noqa: N806
                fiy_6 = My_6 / Mfy_6 * fify_6
                CD_6 = fi_6[-1] / fiy_6  # noqa: N806
            CD_Cols[ind - 1, :] = [CD_1, CD_6]
            print('CD_Cols =', CD_Cols)  # noqa: T201
        CD_Ele = np.concatenate((CD_Cols, CD_Beams), axis=0)  # noqa: N806

        Desp_x = np.loadtxt('Pushover/HoriNodes.out')  # noqa: N806
        Desp_y = np.loadtxt('Pushover/VertNodes.out')  # noqa: N806
        Nodes_desp_x = ListNodes[:, 1] + 3 * Desp_x[-1, 1:]  # noqa: N806
        Nodes_desp_y = ListNodes[:, 2] + 3 * Desp_y[-1, 1:]  # noqa: N806

        fpos = 0.1
        fsize = 1
        DataDC = []  # noqa: N806
        for Ele in Elements:  # noqa: N806
            xi = Nodes_desp_x[Ele.Nod_ini]
            yi = Nodes_desp_y[Ele.Nod_ini]
            xe = Nodes_desp_x[Ele.Nod_end]
            ye = Nodes_desp_y[Ele.Nod_end]
            x = np.array([xi, xe])  # noqa: F841
            y = np.array([yi, ye])  # noqa: F841
            Delta_x = xe - xi  # noqa: N806
            Delta_y = ye - yi  # noqa: N806
            xi_CD = xi + fpos * Delta_x  # noqa: N806
            yi_CD = yi + fpos * Delta_y  # noqa: N806
            xe_CD = xe - fpos * Delta_x  # noqa: N806
            ye_CD = ye - fpos * Delta_y  # noqa: N806
            CD_i = CD_Ele[Ele.EleTag - 1, 0]  # noqa: N806
            CD_e = CD_Ele[Ele.EleTag - 1, 1]  # noqa: N806
            DataDC.append(
                DuctilityCurve(
                    xi_CD, xe_CD, yi_CD, ye_CD, fsize * CD_i, fsize * CD_e
                )
            )
        DC_x, DC_y, DC_size = [], [], []  # noqa: N806
        for DC in DataDC:  # noqa: N806
            DC_x.append([DC.xi, DC.xe])
            DC_y.append([DC.yi, DC.ye])
            DC_size.append([DC.CD_i, DC.CD_e])
        DC_x = np.array(DC_x)  # noqa: N806
        DC_x = DC_x.flatten()  # noqa: N806
        DC_y = np.array(DC_y)  # noqa: N806
        DC_y = DC_y.flatten()  # noqa: N806
        DC_size = np.array(DC_size)  # noqa: N806
        DC_size = DC_size.flatten()  # noqa: N806
        print('DC_x= ', DC_x)  # noqa: T201
        print('DC_y= ', DC_y)  # noqa: T201
        print('DC_size= ', DC_size)  # noqa: T201


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(
        runBuildingDesign(
            args.filenameAIM, args.filenameEVENT, args.filenameSAM, args.getRV
        )
    )
