from Gauss1D import gauss1D
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import json
import sys


def materialPM4(baseInputs, matTag, fn):
    fn.write("nDMaterial PM4Sand {} {:.3f} {:.2f} {:.3f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \
    {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}  {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \n".format(matTag, baseInputs["Dr"], baseInputs["Go"], baseInputs["hpo"], baseInputs["rho"], baseInputs["P_atm"], baseInputs["h0"], baseInputs["emax"],
            baseInputs["emin" ], baseInputs["nb"], baseInputs["nd"], baseInputs["Ado"], baseInputs["z_max"], baseInputs["cz"], baseInputs["ce"], baseInputs["phic"],
            baseInputs["nu"], baseInputs["cgd"], baseInputs["cdr"], baseInputs["ckaf"], baseInputs["Q"], baseInputs["R"], baseInputs["m"], baseInputs["Fsed_min"], baseInputs["p_sedo"]))


def materialPDMY03(baseInputs, matTag, fn):
    fn.write("nDMaterial PressureDependMultiYield03 {} {} {:.2f} {:.3e} {:.3e} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \
    {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {} {:.3f} {:.3f} {:.3f} {:.3f} \n" \
        .format(matTag, baseInputs["nd"], baseInputs["rho"], baseInputs["refShearModul"], baseInputs["refBulkModul"], baseInputs["frictionAng"],
        baseInputs["peakShearStra"], baseInputs["refPress"], baseInputs["pressDependCoe"], baseInputs["PTAng"], baseInputs["mType"],
        baseInputs["ca"], baseInputs["cb"], baseInputs["cc"], baseInputs["cd"], baseInputs["ce"], baseInputs["da"], baseInputs["db"], baseInputs["dc"],
        baseInputs["noYieldSurf"], baseInputs["liquefac1"], baseInputs["liquefac2"], baseInputs["pa"], baseInputs["s0"]))

def materialElastic(baseInputs, matTag, fn):
    fn.write("nDMaterial ElasticIsotropic  {} {:.3e} {:.3f} {:.2f} \n" \
        .format(matTag, baseInputs["E"], baseInputs["poisson"], baseInputs["density"]))

def calibration(variables, inputParameters, fn):
    # This function contains two parts: call gauss1D to generate 1D random field; generate material based on random field
    # Currently only relative density is supported
    # Calibration of PM4Sand is based on a parametric study that produces hpo = f(Dr, Go, CRR)
    # Calibration of PDMY03 is based on interpolation of pre calibrated parameters for a range of Dr

    if variables["materialType"] == "PM4Sand_Random":
        # PM4Sand
        baseInputs = {"Dr":0.65, "Go":600.0, "hpo":0.08, "rho":2.0, "P_atm":101.3, "h0":-1.0, "emax":0.800,
	    "emin" :0.500, "nb":0.50, "nd":0.10, "Ado": -1.0, "z_max":-1.0, "cz": 250.0, "ce": -1.0, "phic":33.0,
	    "nu":0.30, "cgd": 2.0, "cdr": -1.0, "ckaf": -1.0, "Q": 10.0, "R":1.5, "m": 0.01, "Fsed_min": -1.0,
	    "p_sedo":-1.0}
    elif variables["materialType"] == "PDMY03_Random":
        # PDMY03
        baseInputs = {"nd": 2, "rho" : 1.5, "refShearModul" :4.69e4, "refBulkModul" :1.251e5, "frictionAng": 30.0, "peakShearStra" :0.1, "refPress" : 101.3,
	    "pressDependCoe" : 0.5, "PTAng" : 20.4, "mType": 0, "ca" : 0.03, "cb" : 5, "cc" : 0.2, "cd" : 16.0, "ce" : 2.000000, "da" : 0.150,
	    "db" : 3.0000, "dc" : -0.2, "noYieldSurf" : 20, "liquefac1" : 1.0, "liquefac2" : 0.0, "pa" : 101.3, "s0" : 1.73}
    elif variables["materialType"] == "Elastic_Random":
        # Elastic
        baseInputs = {"E": 168480, "poisson" : 0.3, "density" : 2.0}

    for keys in baseInputs:
        baseInputs[keys] = inputParameters[keys]

    # calcualte random field
    # size of mesh
    thickness = variables["thickness"]
    waveLength = variables["Ly"]
    # Number of wave number increments in y-direction
    Ny = thickness / waveLength
    rd = gauss1D(thickness, Ny)
    rd.calculate()
    F = np.squeeze(rd.f.reshape((-1,1)))
    Y = np.linspace(0, rd.Ly, rd.My)
    f = interp1d(Y, F, kind="cubic")

    # mapping from random field to mesh
    elemID = np.arange(variables["eleStart"], variables["eleEnd"] + 1, 1)
    elementY = np.linspace(variables["elevationStart"], variables["elevationEnd"], len(elemID))

    for matTag in elemID:
        residual = variables["mean"] * f(elementY[matTag - variables["eleStart"]]) * variables["COV"]
        print()
        if variables["name"] == "Dr":
            # bound Dr between 0.2 and 0.95
            Dr = min(max(0.2, variables["mean"] + residual), 0.95)
            if Dr != Dr:
                Dr = 0.2
            if variables["materialType"] == "PM4Sand_Random":
                baseInputs["Dr"] = Dr
                Go = baseInputs["Go"]
                # CPT and SPT Based Liquefaction Triggering Procedures (Boulanger and Idriss 2014)
                Cd = 46.0
                N160 = Dr ** 2 * Cd
                CRR_IB = np.exp(N160 / 14.1 + (N160 / 126) ** 2 - (N160 / 23.6) ** 3 + (N160 / 25.4) ** 4 - 2.8)
                # Implementaion, Verification, and Validation of PM4Sand in OpenSees, Long Chen and Pedro Arduino, PEER Report, 2020
                # Based on a parametric study using quoFEM
                a = -0.06438
                b = 0.079598 + 0.12406 * Dr
                c = 0.12194 - 0.47627 * Dr - 0.000047009 * Go - CRR_IB + 0.00014048 * Dr * Go + 0.71347 * Dr ** 2
                hpo = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if hpo != hpo:
                    hpo = 0.4
                    CRR_prediction = 0.114 - 0.44844 * Dr - (4.2648e-5) * Go + 0.079849 * hpo + (1.2811e-4) * Dr * Go \
                    + 0.12136 * Dr * hpo + 0.69676 * Dr ** 2 - 0.06381 * hpo ** 2
                    # bound hpo between 0.05 and 1.0
                    if CRR_prediction > CRR_IB:
                        hpo = 0.05
                    else:
                        hpo = 1.0
                baseInputs["hpo"] = hpo
                materialPM4(baseInputs, matTag, fn)
            elif variables["materialType"] == "PDMY03_Random":
                Dr = max(min(Dr, 0.87), 0.33)
                baseInputs["Dr"] = Dr
                # interpolation using Khosravifar, A., Elgamal, A., Lu, J., and Li, J. [2018].
                # "A 3D model for earthquake-induced liquefaction triggering and post-liquefaction response."
                # Soil Dynamics and Earthquake Engineering, 110, 43-52
                drs = np.array([0.33, 0.57, 0.74, 0.87])
                df = pd.DataFrame([(46900,125100,25.4,20.4,0.03,5,0.2,16,2,0.15,3,-0.2),
                (7.37e4, 1.968e5, 30.3, 25.3, 0.012, 3.0, 0.4, 9.0, 0.0, 0.3, 3.0, -0.3),
                (9.46e4, 2.526e5, 35.8, 30.8, 0.005, 1.0, 0.6, 4.6, -1.0, 0.45, 3.0, -0.4),
                (1.119e5, 2.983e5, 42.2, 37.2, 0.001, 0.0, 0.8, 2.2, 0.0, 0.6, 3.0, -0.5)],
                columns=("refShearModul", "refBulkModul", "frictionAng", "PTAng", "ca", "cb", "cc", \
                    "cd", "ce", "da", "db", "dc"))
                for (columnName, columnData) in df.iteritems():
                    f_Dr = interp1d(drs, df[columnName], kind="cubic")
                    baseInputs[columnName] = f_Dr(Dr)
                materialPDMY03(baseInputs, matTag, fn)
        elif variables["name"] == "Vs":
            if variables["materialType"] == "Elastic_Random":
                # bound Dr between 50 and 1500
                Vs = min(max(50, variables["mean"] + residual), 1500)
                baseInputs["E"] = 2.0 * baseInputs["density"] * Vs * Vs * (1.0 + baseInputs["poisson"])
                fn.write("#Vs = {:.2f}\n".format(Vs))
                materialElastic(baseInputs, matTag, fn)

def createMaterial(data):

    eleStart = 0
    eleEnd = 0
    elevationStart = 0
    elevationEnd = 0
    numElems = 0
    totalHeight = 0
    randomMaterialList = ["PM4Sand_Random", "PDMY03_Random", "Elastic_Random"]
    fn = open("material.tcl", "w")

    for layer in reversed(data["soilProfile"]["soilLayers"]):
        if  layer["eSize"] != 0:
            eleStart = numElems + 1
            numElemsLayer = round(layer["thickness"] / layer["eSize"])
            numElems += numElemsLayer
            eleSize = layer["thickness"] / numElemsLayer
            elevationStart = eleSize / 2.0
            totalHeight += layer["thickness"]
            eleEnd = numElems
            elevationEnd = layer["thickness"] - eleSize / 2.0
        if data["materials"][layer["material"] - 1]["type"] in randomMaterialList:
            variables = dict(
                materialType = data["materials"][layer["material"] - 1]["type"],
                name = data["materials"][layer["material"] - 1]["Variable"],
                mean = data["materials"][layer["material"] - 1]["mean"],
                COV = data["materials"][layer["material"] - 1]["COV"],
                Ly = data["materials"][layer["material"] - 1]["Ly"],
                thickness = layer["thickness"],
                eleStart = eleStart,
                eleEnd = eleEnd,
                elevationStart = elevationStart,  # location of first Gauss Point respect to layer base
                elevationEnd = elevationEnd  # location of last Gauss Point respect to layer base
            )
            inputParameters =  data["materials"][layer["material"] - 1]
            calibration(variables, inputParameters, fn)

    fn.close()


if __name__ == "__main__":

    srtName = sys.argv[0]

    ## data obtained from user input
    # define the random field
    with open(srtName) as json_file:
        data = json.load(json_file)

    eventData = data["Events"][0]

    createMaterial(eventData)
