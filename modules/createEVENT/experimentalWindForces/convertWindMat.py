# python code to open the .mat file  # noqa: INP001, D100
# and put data into a SimCenter JSON file

import json
import os

import numpy as np
import scipy.io as sio


def parseWindMatFile(matFileIn, windFileOutName):  # noqa: ANN001, ANN201, N802, N803, D103
    dataDir = os.getcwd()  # noqa: PTH109, N806, F841
    scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806, F841

    mat_contents = sio.loadmat(matFileIn)

    depth = float(mat_contents['D'][0])
    breadth = float(mat_contents['B'][0])
    height = float(mat_contents['H'][0])
    fs = float(mat_contents['fs'][0])
    vRef = float(mat_contents['Vref'][0])  # noqa: N806

    if 's_target' in mat_contents:
        case = 'spectra'  # noqa: F841
        comp_CFmean = np.squeeze(np.array(mat_contents['comp_CFmean']))  # noqa: N806
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
        Fx = np.squeeze(np.array(mat_contents['Fx']))  # noqa: N806
        Fy = np.squeeze(np.array(mat_contents['Fy']))  # noqa: N806
        Tz = np.squeeze(np.array(mat_contents['Tz']))  # noqa: N806
        t = np.squeeze(np.array(mat_contents['t']))

        myJson = {}  # noqa: N806
        myJson['D'] = depth
        myJson['H'] = height
        myJson['B'] = breadth
        myJson['fs'] = fs
        myJson['Vref'] = vRef

        myJson['Fx'] = np.array(Fx).tolist()
        myJson['Fy'] = np.array(Fy).tolist()
        myJson['Tz'] = np.array(Tz).tolist()
        myJson['t'] = np.array(t).tolist()
        with open(windFileOutName, 'w') as f:  # noqa: PTH123
            json.dump(myJson, f)

        # file = open(windFileOutName,"w")  # noqa: ERA001
        # file.write("{")  # noqa: ERA001
        # file.write("\"D\":%f," % depth)  # noqa: ERA001
        # file.write("\"H\":%f," % height)  # noqa: ERA001
        # file.write("\"B\":%f," % breadth)  # noqa: ERA001
        # file.write("\"fs\":%f," % fs)  # noqa: ERA001
        # file.write("\"Vref\":%f," % vRef)  # noqa: ERA001

        # case = "timeHistory"  # noqa: ERA001
        # Fx = mat_contents['Fx']  # noqa: ERA001
        # Fy = mat_contents['Fy']  # noqa: ERA001
        # Tz = mat_contents['Tz']  # noqa: ERA001
        # t = mat_contents['t']  # noqa: ERA001

        # dt = t[0][1]-t[0][0]  # noqa: ERA001
        # file.write("\"t\": [")  # noqa: ERA001
        # for i in range(t.shape[1]-1):
        #     file.write("%f," % t[0][i])  # noqa: ERA001
        # file.write("%f]," % t[0][t.shape[1]-1])  # noqa: ERA001

        # numFloors = Fx.shape[0]  # noqa: ERA001
        # numSteps = Fx.shape[1]  # noqa: ERA001

        # print(numFloors, numSteps)  # noqa: ERA001

        # file.write("\"numFloors\":%d," % numFloors)  # noqa: ERA001
        # file.write("\"numSteps\":%d," % numSteps)  # noqa: ERA001

        # file.write("\"Fx\":[" )  # noqa: ERA001
        # for i in range(0, numFloors):
        #     file.write("[")  # noqa: ERA001
        #     for j in range(0, numSteps-1):
        #         file.write("%f," % Fx[i,j])  # noqa: ERA001
        #     file.write("%f]" % Fx[i,numSteps-1])  # noqa: ERA001
        #     if i == numFloors-1:
        #         file.write("],")  # noqa: ERA001
        #     else:  # noqa: ERA001
        #         file.write(",")  # noqa: ERA001

        # file.write("\"Fy\":[" )  # noqa: ERA001
        # for i in range(0, numFloors):
        #     file.write("[")  # noqa: ERA001
        #     for j in range(0, numSteps-1):
        #         file.write("%f," % Fy[i,j])  # noqa: ERA001
        #     file.write("%f]" % Fy[i,numSteps-1])  # noqa: ERA001
        #     if i == numFloors-1:
        #         file.write("],")  # noqa: ERA001
        #     else:  # noqa: ERA001
        #         file.write(",")  # noqa: ERA001

        # file.write("\"Tz\":[" )  # noqa: ERA001
        # for i in range(0, numFloors):
        #     file.write("[")  # noqa: ERA001
        #     for j in range(0, numSteps-1):
        #         file.write("%f," % Tz[i,j])  # noqa: ERA001
        #     file.write("%f]" % Tz[i,numSteps-1])  # noqa: ERA001
        #     if i == numFloors-1:
        #         file.write("]")  # noqa: ERA001
        #     else:  # noqa: ERA001
        #         file.write(",")  # noqa: ERA001

        # file.write("}")  # noqa: ERA001
        # file.close()  # noqa: ERA001

        # Check valid JSON file,
        validate = True
        if validate:
            with open(windFileOutName) as infile:  # noqa: PTH123
                json_data = infile.read()

            # Try to parse the JSON data
            try:
                json_object = json.loads(json_data)  # noqa: F841
                print('JSON file is valid')  # noqa: T201
            except json.decoder.JSONDecodeError:
                print('JSON file is not valid')  # noqa: T201


def createSpectraJson(  # noqa: ANN201, N802, D103, PLR0913
    windFileOutName,  # noqa: ANN001, N803
    breadth,  # noqa: ANN001
    depth,  # noqa: ANN001
    height,  # noqa: ANN001
    fs,  # noqa: ANN001
    vRef,  # noqa: ANN001, N803
    f_target,  # noqa: ANN001
    s_target,  # noqa: ANN001
    comp_CFmean,  # noqa: ANN001, N803
    norm_all,  # noqa: ANN001
):
    ncomp = comp_CFmean.shape[0]  # noqa: F841
    nf = f_target.shape[0]  # noqa: F841

    myJson = {}  # noqa: N806
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

    with open(windFileOutName, 'w') as f:  # noqa: PTH123
        json.dump(myJson, f)

    # Check valid JSON file
    validate = True
    if validate:
        with open(windFileOutName) as infile:  # noqa: PTH123
            json_data = infile.read()

        # Try to parse the JSON data
        try:
            json_object = json.loads(json_data)  # noqa: F841
            print('JSON file is valid')  # noqa: T201
        except json.decoder.JSONDecodeError:
            print('JSON file is not valid')  # noqa: T201

    # file = open(windFileOutName,"w")  # noqa: ERA001
    # file.write("{")  # noqa: ERA001
    # file.write("\"D\":%f," % depth)  # noqa: ERA001
    # file.write("\"H\":%f," % height)  # noqa: ERA001
    # file.write("\"B\":%f," % breadth)  # noqa: ERA001
    # file.write("\"fs\":%f," % fs)  # noqa: ERA001
    # file.write("\"Vref\":%f," % vRef)  # noqa: ERA001
    # file.write("\"units\":{\"length\":\"m\",\"time\":\"sec\"},")  # noqa: ERA001

    # ncomp = comp_CFmean.shape[0]  # noqa: ERA001
    # nf = f_target.shape[0]  # noqa: ERA001

    # file.write("\"comp_CFmean\":[" )  # noqa: ERA001
    # for j in range(0, ncomp - 1):
    #     file.write("%f," % comp_CFmean[j])  # noqa: ERA001
    # file.write("%f]," % comp_CFmean[ncomp - 1])  # noqa: ERA001

    # file.write("\"norm_all\":[" )  # noqa: ERA001
    # for j in range(0, ncomp - 1):
    #     file.write("%f," % norm_all[j])  # noqa: ERA001
    # file.write("%f]," % norm_all[ncomp - 1])  # noqa: ERA001

    # file.write("\"f_target\":[" )  # noqa: ERA001
    # for j in range(0, nf - 1):
    #     file.write("%f," % f_target[j])  # noqa: ERA001
    # file.write("%f]," % f_target[ncomp - 1])  # noqa: ERA001

    # file.write("\"s_target_real\":[" )  # noqa: ERA001
    # for i in range(0, ncomp):
    #     file.write("[")  # noqa: ERA001
    #     for j in range(0, ncomp):
    #         file.write("[")  # noqa: ERA001
    #         for k in range(0, nf-1):
    #             file.write("%f," % s_target[i,j,k].real)  # noqa: ERA001
    #         file.write("%f]" % s_target[i,j,nf-1].real)  # noqa: ERA001

    #         if j == ncomp-1:
    #             file.write("]")  # noqa: ERA001
    #         else:  # noqa: ERA001
    #             file.write(",")  # noqa: ERA001

    #     if i == ncomp-1:
    #         file.write("],")  # noqa: ERA001
    #     else:  # noqa: ERA001
    #         file.write(",")  # noqa: ERA001

    # file.write("\"s_target_imag\":[" )  # noqa: ERA001
    # for i in range(0, ncomp):
    #     file.write("[")  # noqa: ERA001
    #     for j in range(0, ncomp):
    #         file.write("[")  # noqa: ERA001
    #         for k in range(0, nf-1):
    #             file.write("%f," % s_target[i,j,k].imag)  # noqa: ERA001
    #         file.write("%f]" % s_target[i,j,nf-1].imag)  # noqa: ERA001

    #         if j == ncomp-1:
    #             file.write("]")  # noqa: ERA001
    #         else:  # noqa: ERA001
    #             file.write(",")  # noqa: ERA001

    #     if i == ncomp-1:
    #         file.write("]")  # noqa: ERA001
    #     else:  # noqa: ERA001
    #         file.write(",")  # noqa: ERA001

    # file.write("}")  # noqa: ERA001
    # file.close()  # noqa: ERA001


def createPODJson(  # noqa: ANN201, N802, D103, PLR0913
    filename, V, D1, SpeN, f_target, norm_all, D, H, B, fs, vRef, comp_CFmean  # noqa: ANN001, N803
):
    myJson = {}  # noqa: N806
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

    with open(filename, 'w') as f:  # noqa: PTH123
        json.dump(myJson, f)
