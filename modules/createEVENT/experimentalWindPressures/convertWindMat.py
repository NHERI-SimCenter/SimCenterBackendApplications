# python code to open the .mat file  # noqa: CPY001, D100, INP001
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
        with open(windFileOutName, 'w') as f:  # noqa: PLW1514, PTH123
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
            with open(windFileOutName) as infile:  # noqa: FURB101, PLW1514, PTH123
                json_data = infile.read()

            # Try to parse the JSON data
            try:
                json_object = json.loads(json_data)  # noqa: F841
                print('JSON file is valid')  # noqa: T201
            except json.decoder.JSONDecodeError:
                print('JSON file is not valid')  # noqa: T201


def createSpectraJson(  # noqa: ANN201, N802, D103
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

    with open(windFileOutName, 'w') as f:  # noqa: PLW1514, PTH123
        json.dump(myJson, f)

    # Check valid JSON file
    validate = True
    if validate:
        with open(windFileOutName) as infile:  # noqa: FURB101, PLW1514, PTH123
            json_data = infile.read()

        # Try to parse the JSON data
        try:
            json_object = json.loads(json_data)  # noqa: F841
            print('JSON file is valid')  # noqa: T201
        except json.decoder.JSONDecodeError:
            print('JSON file is not valid')  # noqa: T201

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


def createPODJson(  # noqa: ANN201, N802, D103
    filename,  # noqa: ANN001
    V,  # noqa: ANN001, N803
    D1,  # noqa: ANN001, N803
    SpeN,  # noqa: ANN001, N803
    f_target,  # noqa: ANN001
    norm_all,  # noqa: ANN001
    D,  # noqa: ANN001, N803
    H,  # noqa: ANN001, N803
    B,  # noqa: ANN001, N803
    fs,  # noqa: ANN001
    vRef,  # noqa: ANN001, N803
    comp_CFmean,  # noqa: ANN001, N803
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

    with open(filename, 'w') as f:  # noqa: PLW1514, PTH123
        json.dump(myJson, f)
