# python code to open the .mat file
# and put data into a SimCenter JSON file

import sys
import os
import subprocess
import json
import stat
import shutil
import numpy as np
import scipy.io as sio
from pprint import pprint


def parseWindMatFile(matFileIn, windFileOutName):
    dataDir = os.getcwd()
    scriptDir = os.path.dirname(os.path.realpath(__file__))

    mat_contents = sio.loadmat(matFileIn)

    depth = float(mat_contents['D'][0])
    breadth = float(mat_contents['B'][0])
    height = float(mat_contents['H'][0])
    fs = float(mat_contents['fs'][0])
    vRef = float(mat_contents['Vref'][0])

    if 's_target' in mat_contents:
        case = 'spectra'
        comp_CFmean = np.squeeze(np.array(mat_contents['comp_CFmean']))
        norm_all = np.squeeze(np.array(mat_contents['norm_all']))
        f_target = np.squeeze(np.array(mat_contents['f_target']))
        s_target = np.squeeze(np.array(mat_contents['s_target']))

        createSpectraJson(
            windFileOutName,
            breadth,
            depth,
            height,
            fs,
            vRef,
            f_target,
            s_target,
            comp_CFmean,
            norm_all,
        )

    elif 'Fx' in mat_contents:
        Fx = np.squeeze(np.array(mat_contents['Fx']))
        Fy = np.squeeze(np.array(mat_contents['Fy']))
        Tz = np.squeeze(np.array(mat_contents['Tz']))
        t = np.squeeze(np.array(mat_contents['t']))

        myJson = {}
        myJson['D'] = depth
        myJson['H'] = height
        myJson['B'] = breadth
        myJson['fs'] = fs
        myJson['Vref'] = vRef

        myJson['Fx'] = np.array(Fx).tolist()
        myJson['Fy'] = np.array(Fy).tolist()
        myJson['Tz'] = np.array(Tz).tolist()
        myJson['t'] = np.array(t).tolist()
        with open(windFileOutName, 'w') as f:
            json.dump(myJson, f)

        # file = open(windFileOutName,"w")
        # file.write("{")
        # file.write("\"D\":%f," % depth)
        # file.write("\"H\":%f," % height)
        # file.write("\"B\":%f," % breadth)
        # file.write("\"fs\":%f," % fs)
        # file.write("\"Vref\":%f," % vRef)

        # case = "timeHistory"
        # Fx = mat_contents['Fx']
        # Fy = mat_contents['Fy']
        # Tz = mat_contents['Tz']
        # t = mat_contents['t']

        # dt = t[0][1]-t[0][0]
        # file.write("\"t\": [")
        # for i in range(t.shape[1]-1):
        #     file.write("%f," % t[0][i])
        # file.write("%f]," % t[0][t.shape[1]-1])

        # numFloors = Fx.shape[0]
        # numSteps = Fx.shape[1]

        # print(numFloors, numSteps)

        # file.write("\"numFloors\":%d," % numFloors)
        # file.write("\"numSteps\":%d," % numSteps)

        # file.write("\"Fx\":[" )
        # for i in range(0, numFloors):
        #     file.write("[")
        #     for j in range(0, numSteps-1):
        #         file.write("%f," % Fx[i,j])
        #     file.write("%f]" % Fx[i,numSteps-1])
        #     if i == numFloors-1:
        #         file.write("],")
        #     else:
        #         file.write(",")

        # file.write("\"Fy\":[" )
        # for i in range(0, numFloors):
        #     file.write("[")
        #     for j in range(0, numSteps-1):
        #         file.write("%f," % Fy[i,j])
        #     file.write("%f]" % Fy[i,numSteps-1])
        #     if i == numFloors-1:
        #         file.write("],")
        #     else:
        #         file.write(",")

        # file.write("\"Tz\":[" )
        # for i in range(0, numFloors):
        #     file.write("[")
        #     for j in range(0, numSteps-1):
        #         file.write("%f," % Tz[i,j])
        #     file.write("%f]" % Tz[i,numSteps-1])
        #     if i == numFloors-1:
        #         file.write("]")
        #     else:
        #         file.write(",")

        # file.write("}")
        # file.close()

        # Check valid JSON file,
        validate = True
        if validate:
            with open(windFileOutName, 'r') as infile:
                json_data = infile.read()

            # Try to parse the JSON data
            try:
                json_object = json.loads(json_data)
                print('JSON file is valid')
            except json.decoder.JSONDecodeError:
                print('JSON file is not valid')


def createSpectraJson(
    windFileOutName,
    breadth,
    depth,
    height,
    fs,
    vRef,
    f_target,
    s_target,
    comp_CFmean,
    norm_all,
):
    ncomp = comp_CFmean.shape[0]
    nf = f_target.shape[0]

    myJson = {}
    myJson['D'] = depth
    myJson['H'] = height
    myJson['B'] = breadth
    myJson['fs'] = fs
    myJson['Vref'] = vRef
    myJson['comp_CFmean'] = comp_CFmean.tolist()
    myJson['norm_all'] = norm_all.tolist()
    myJson['f_target'] = f_target.tolist()

    myJson['s_target_real'] = np.real(s_target).tolist()
    myJson['s_target_imag'] = np.imag(s_target).tolist()

    with open(windFileOutName, 'w') as f:
        json.dump(myJson, f)

    # Check valid JSON file
    validate = True
    if validate:
        with open(windFileOutName, 'r') as infile:
            json_data = infile.read()

        # Try to parse the JSON data
        try:
            json_object = json.loads(json_data)
            print('JSON file is valid')
        except json.decoder.JSONDecodeError:
            print('JSON file is not valid')

    # file = open(windFileOutName,"w")
    # file.write("{")
    # file.write("\"D\":%f," % depth)
    # file.write("\"H\":%f," % height)
    # file.write("\"B\":%f," % breadth)
    # file.write("\"fs\":%f," % fs)
    # file.write("\"Vref\":%f," % vRef)
    # file.write("\"units\":{\"length\":\"m\",\"time\":\"sec\"},")

    # ncomp = comp_CFmean.shape[0]
    # nf = f_target.shape[0]

    # file.write("\"comp_CFmean\":[" )
    # for j in range(0, ncomp - 1):
    #     file.write("%f," % comp_CFmean[j])
    # file.write("%f]," % comp_CFmean[ncomp - 1])

    # file.write("\"norm_all\":[" )
    # for j in range(0, ncomp - 1):
    #     file.write("%f," % norm_all[j])
    # file.write("%f]," % norm_all[ncomp - 1])

    # file.write("\"f_target\":[" )
    # for j in range(0, nf - 1):
    #     file.write("%f," % f_target[j])
    # file.write("%f]," % f_target[ncomp - 1])

    # file.write("\"s_target_real\":[" )
    # for i in range(0, ncomp):
    #     file.write("[")
    #     for j in range(0, ncomp):
    #         file.write("[")
    #         for k in range(0, nf-1):
    #             file.write("%f," % s_target[i,j,k].real)
    #         file.write("%f]" % s_target[i,j,nf-1].real)

    #         if j == ncomp-1:
    #             file.write("]")
    #         else:
    #             file.write(",")

    #     if i == ncomp-1:
    #         file.write("],")
    #     else:
    #         file.write(",")

    # file.write("\"s_target_imag\":[" )
    # for i in range(0, ncomp):
    #     file.write("[")
    #     for j in range(0, ncomp):
    #         file.write("[")
    #         for k in range(0, nf-1):
    #             file.write("%f," % s_target[i,j,k].imag)
    #         file.write("%f]" % s_target[i,j,nf-1].imag)

    #         if j == ncomp-1:
    #             file.write("]")
    #         else:
    #             file.write(",")

    #     if i == ncomp-1:
    #         file.write("]")
    #     else:
    #         file.write(",")

    # file.write("}")
    # file.close()


def createPODJson(
    filename, V, D1, SpeN, f_target, norm_all, D, H, B, fs, vRef, comp_CFmean
):
    myJson = {}
    myJson['V_imag'] = np.imag(V).tolist()
    myJson['V_real'] = np.real(V).tolist()
    myJson['D1'] = D1.tolist()
    myJson['SpeN'] = SpeN
    myJson['f_target'] = f_target.tolist()
    myJson['norm_all'] = norm_all.tolist()
    myJson['comp_CFmean'] = comp_CFmean.tolist()
    myJson['D'] = D
    myJson['H'] = H
    myJson['B'] = B
    myJson['fs'] = fs
    myJson['Vref'] = vRef

    with open(filename, 'w') as f:
        json.dump(myJson, f)
