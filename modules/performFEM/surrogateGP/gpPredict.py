import time
import pickle as pickle
import numpy as np
import os
import sys
import json as json
import shutil
from scipy.stats import lognorm, norm
import subprocess

try:
    moduleName = "emukit"
    from emukit.multi_fidelity.convert_lists_to_array import  convert_x_list_to_array, convert_xy_lists_to_arrays
    moduleName = "GPy"
    import GPy as GPy
    error_tag = False  # global variable
except:
    error_tag = True
    print("Failed to import module:" + moduleName)

# from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

def main(params_dir,surrogate_dir,json_dir,result_file, dakota_path):
    global error_file


    os_type=sys.platform.lower()
    run_type ='runninglocal'

    #
    # create a log file
    #

    msg0 = os.path.basename(os.getcwd()) + " : "
    file_object = open('../surrogateLog.log', 'a')

    folderName = os.path.basename(os.getcwd())
    sampNum = folderName.split(".")[-1]

    #
    # read json -- current input file
    #

    def error_exit(msg):
        error_file.write(msg) # local
        error_file.close()
        file_object.write(msg0 + msg) # global file
        file_object.close()
        print(msg)
        exit(-1)

    def error_warning(msg):
        error_file.write(msg)
        file_object.write(msg0 + msg)
        print(msg)

    with open(dakota_path) as f: # current input file
        try:
            inp_tmp = json.load(f)
            inp_fem = inp_tmp["Applications"]["FEM"]
        except ValueError:
            msg = 'invalid json format - dakota.json'
            error_exit(msg)

    norm_var_thr = inp_fem["varThres"]
    when_inaccurate = inp_fem["femOption"]
    do_mf = inp_tmp

    np.random.seed(int(inp_fem["gpSeed"])+int(sampNum))

    # sampNum=0

    # if no g and rv,


    #
    # read json -- original input for training surrogate
    #
    f = open(json_dir)
    try:
        sur = json.load(f)
    except ValueError:
        msg = 'invalid json format'
        error_exit(msg)

    f.close()
    did_logtransform = sur["doLogtransform"]
    kernel = sur["kernName"]
    if kernel == 'Radial Basis':
        kern_name = 'rbf'
    elif kernel == 'Exponential':
        kern_name = 'Exponential'
    elif kernel == 'Matern 3/2':
        kern_name = 'Mat32'
    elif kernel == 'Matern 5/2':
        kern_name = 'Mat52'
    did_mf = sur["doMultiFidelity"]

    # from json
    g_name_sur = list()
    ng_sur = 0
    Y=np.zeros((sur['highFidelityInfo']['valSamp'],sur['ydim']))
    for g in sur['ylabels']:
        g_name_sur += [g]
        Y[:,ng_sur]=np.array(sur['yExact'][g])
        ng_sur += 1

    rv_name_sur = list()
    nrv_sur = 0
    X=np.zeros((sur['highFidelityInfo']['valSamp'],sur['xdim']))
    for rv in sur['xlabels']:
        rv_name_sur += [rv]
        X[:,nrv_sur]=np.array(sur['xExact'][rv])
        nrv_sur += 1

    try:
        constIdx = sur['highFidelityInfo']["constIdx"]
        constVal = sur['highFidelityInfo']["constVal"]
    except:
        constIdx = []
        constVal = []

        # Read pickles
    # todo: fix for different nys

    if kernel == 'Radial Basis':
        kr = GPy.kern.RBF(input_dim=nrv_sur, ARD=True)
    elif kernel == 'Exponential':
        kr = GPy.kern.Exponential(input_dim=nrv_sur, ARD=True)
    elif kernel == 'Matern 3/2':
        kr = GPy.kern.Matern32(input_dim=nrv_sur, ARD=True)
    elif kernel == 'Matern 5/2':
        kr = GPy.kern.Matern52(input_dim=nrv_sur, ARD=True)

    if sur['doLinear']:
        kr = kr + GPy.kern.Linear(input_dim=nrv_sur, ARD=True)

    if did_logtransform:
        Y = np.log(Y)

    if not did_mf:
        kg = kr
        m_list = list()
        for ny in range(ng_sur):
            m_list = m_list + [GPy.models.GPRegression(X, Y[:, ny][np.newaxis].transpose(), kernel=kg.copy(),normalizer=True)]
            for key, val in sur["modelInfo"][g_name_sur[ny]].items():
                exec('m_list[ny].' + key + '= np.array(val)')

    else:
        with open(surrogate_dir, "rb") as file:
            m_list=pickle.load(file)

    # to read:::
    # kern_name='Mat52'
    #did_logtransform=True

    # at ui



    # REQUIRED: rv_name, y_var

    t_total = time.process_time()

    with open(params_dir, "r") as x_file:
        data = x_file.readlines()
        nrv = int(data[0])
        if nrv != nrv_sur:
            msg = 'Error importing input data: Number of dimension inconsistent: surrogate model requires {} RV(s) but input has {} RV(s).\n'.format(nrv_sur, nrv)
            error_exit(msg)

        #rv_name = list()

        for i in range(nrv):
            name_values = data[i + 1].split()
            name = name_values[0]
            samples = [float(vals) for vals in name_values[1:]]
            ns = len(samples)
            try:
                id_map = rv_name_sur.index(name)
            except ValueError:
                msg = 'Error importing input data: variable "{}" not identified.'.format(name)
                error_exit(msg)

            if i==0:
                nsamp = ns
                rv_val = np.zeros((ns, nrv))

            if ns!=nsamp:
                msg = 'Error importing input data: sample size in params.in is not consistent.'
                error_exit(msg)

            rv_val[:, id_map] = samples


        g_idx = []
        for edp in (inp_tmp["EDP"]):
            edp_names = []
            if edp["length"]==1:
                edp_names += [edp["name"]]
            else:
                for i in range(0,edp["length"]):
                    edp_names += [edp["name"] + "_" + str(i+1)]
            try:
                for i in range(0,edp["length"]):
                    id_map = g_name_sur.index(edp_names[i])
                    g_idx += [id_map]
            except ValueError:
                msg = 'Error importing input data: qoi "{}" not identified.'.format(edp["name"])
                error_exit(msg)




    # f = open(work_dir + '/templatedir/dakota.json')
    # inp = json.load(f)
    # f.close()


    # try:
    #     f = open(surrogate_dir, 'rb')
    # except OSError:
    #     msg = 'Could not open/read surrogate model from: ' + surrogate_dir + '\n'
    #     print(msg)
    #     error_file.write(msg)
    #     error_file.close()
    #     file_object.write(msg0+msg)
    #     file_object.close()
    #     exit(-1)
    # with f:
    #     m_list = pickle.load(f)



    # read param in file and sort input

    y_dim = len(m_list)

    y_pred_median = np.zeros([nsamp, y_dim])
    y_pred_var_tmp=np.zeros([nsamp, y_dim])
    y_pred_var=np.zeros([nsamp, y_dim])
    y_data_var=np.zeros([nsamp, y_dim])
    y_samp = np.zeros([nsamp, y_dim])
    y_q1 = np.zeros([nsamp, y_dim])
    y_q3 = np.zeros([nsamp, y_dim])
    for ny in range(y_dim):
        y_data_var[:,ny] = np.var(m_list[ny].Y)
        #y_pred_tmp, y_pred_var_tmp  = m_list[ny].predict(rv_val)
        if ny in constIdx:
            y_pred_median_tmp, y_pred_var_tmp[ny] = np.ones([nsamp])*constVal[constIdx.index(ny)], np.zeros([nsamp])
        else:
            y_pred_median_tmp, y_pred_var_tmp_tmp  = predict(m_list[ny],rv_val,did_mf)
            y_pred_median_tmp= np.squeeze(y_pred_median_tmp)
            y_pred_var_tmp_tmp= np.squeeze(y_pred_var_tmp_tmp)
        y_pred_var_tmp[:, ny] = y_pred_var_tmp_tmp
        y_samp_tmp = np.random.normal(y_pred_median_tmp,np.sqrt(y_pred_var_tmp[:, ny] ))
        if did_logtransform:
            y_pred_median[:,ny]= np.exp(y_pred_median_tmp)
            # y_var_val = np.var(np.log(m_list[ny].Y))
            y_pred_var[:,ny] = np.exp(2 * y_pred_median_tmp + y_pred_var_tmp[:, ny] ) * (np.exp(y_pred_var_tmp[:, ny]) - 1)
            y_samp[:,ny] = np.exp(y_samp_tmp)

            #mu = np.log(y_pred_median_tmp)
            #sig = np.sqrt(np.log(y_pred_var_tmp[ny]/ pow(y_pred_median_tmp, 2) + 1))

            y_q1[:,ny] = lognorm.ppf(0.05, s=np.sqrt(y_pred_var_tmp[:, ny] ), scale=np.exp(y_pred_median_tmp))
            y_q3[:,ny]= lognorm.ppf(0.95, s=np.sqrt(y_pred_var_tmp[:, ny] ), scale=np.exp(y_pred_median_tmp))

        else:
            
            y_pred_median[:,ny]=y_pred_median_tmp
            y_pred_var[:,ny]= y_pred_var_tmp[:, ny]
            y_samp[:,ny] = y_samp_tmp
            y_q1[:,ny] = norm.ppf(0.05, loc=y_pred_median_tmp, scale=np.sqrt(y_pred_var_tmp[:, ny] ))
            y_q3[:,ny] = norm.ppf(0.95, loc=y_pred_median_tmp, scale=np.sqrt(y_pred_var_tmp[:, ny] ))

        if np.isnan(y_samp[:,ny]).any():
            y_samp[:,ny] = np.nan_to_num(y_samp[:,ny])
        if np.isnan(y_pred_var[:,ny]).any():
            y_pred_var[:,ny] = np.nan_to_num(y_pred_var[:,ny])


        #for parname in m_list[ny].parameter_names():
        #    if (kern_name in parname) and parname.endswith('variance'):
        #        exec('y_pred_prior_var[ny]=m_list[ny].' + parname)

    #error_ratio1 = y_pred_var.T / y_pred_prior_var
    error_ratio2 = y_pred_var_tmp / y_data_var
    idx = np.argmax(error_ratio2,axis=1) + 1

    '''
    if np.max(error_ratio1) > norm_var_thr:

        is_accurate = False
        idx = np.argmax(error_ratio1) + 1

        msg = 'Prediction error of output {} is {:.2f}%, which is greater than threshold={:.2f}%  '.format(idx, np.max(
            error_ratio1)*100, norm_var_thr*100)
    '''

    is_accurate_array = (np.max(error_ratio2,axis=1) < norm_var_thr)

    y_pred_subset = np.zeros([nsamp, len(g_idx)])
    msg1 = []
    for ns in range(nsamp):
        if not is_accurate_array[ns]:
            msg1 += ['Prediction error of output {} is {:.2f}%, which is greater than threshold={:.2f}%  '.format(idx[ns], np.max(
            error_ratio2[ns]) * 100, norm_var_thr * 100)]
        else:
            msg1 += ['']

        if not is_accurate_array[ns]:


            if when_inaccurate == 'doSimulation':

                #
                # (1) create "workdir.idx " folder :need C++17 to use the files system namespace
                #
                templatedirFolder = os.path.join(os.getcwd(), 'templatedir_SIM')

                current_dir_i = os.path.join(os.getcwd(), 'subworkdir.{}'.format(1+ns))
                try:
                    shutil.copytree(templatedirFolder, current_dir_i)
                except Exception as ex:
                    try:
                        shutil.copytree(templatedirFolder, current_dir_i)
                    except Exception as ex:
                        msg = "Error running FEM: " + str(ex)

                # change directory, create params.in
                #shutil.copyfile(os.path.join(os.getcwd(), 'params.in'), os.path.join(current_dir_i, 'params.in'))
                outF = open(current_dir_i + "/params.in", "w")
                outF.write("{}\n".format(nrv))
                for i in range(nrv):
                    outF.write("{} {}\n".format(rv_name_sur[i], rv_val[ns, i]))
                outF.close()

                os.chdir(current_dir_i)

                # run workflowDriver
                if os_type.lower().startswith('win') and run_type.lower() == 'runninglocal':
                    workflowDriver = "driver.bat"
                else:
                    workflowDriver = "driver"

                workflow_run_command = '{}/{}'.format(current_dir_i, workflowDriver)
                subprocess.Popen(workflow_run_command, shell=True).wait()

                # back to directory, copy result.out
                #shutil.copyfile(os.path.join(sim_dir, 'results.out'), os.path.join(os.getcwd(), 'results.out'))

                with open('results.out', 'r') as f:
                    y_pred = np.array([np.loadtxt(f)]).flatten()
                    y_pred_subset[ns,:] = y_pred[g_idx]

                os.chdir("../")

                msg2 = msg0+msg1[ns]+'- RUN original model\n'
                error_warning(msg2)
                #exit(-1)
                
            elif when_inaccurate == 'giveError':
                msg2 = msg0+msg1[ns]+'- EXIT\n'
                error_exit(msg2)

            elif when_inaccurate == 'continue':
                msg2 = msg0+msg1[ns]+'- CONTINUE [Warning: results may not be accurate]\n'
                error_warning(msg2)

                if inp_fem["predictionOption"].lower().startswith("median"):
                    y_pred_subset[ns,:]  = y_pred_median[ns,g_idx]
                elif inp_fem["predictionOption"].lower().startswith("rand"):
                    y_pred_subset[ns,:]  = y_samp[ns,g_idx]

        else:
            msg3 = msg0+'Prediction error of output {} is {:.2f}%\n'.format(idx[ns], np.max(error_ratio2[ns])*100)
            error_warning(msg3)

            if inp_fem["predictionOption"].lower().startswith("median"):
                y_pred_subset[ns,:]  = y_pred_median[ns,g_idx]
            elif inp_fem["predictionOption"].lower().startswith("rand"):
                y_pred_subset[ns,:]  = y_samp[ns,g_idx]

    np.savetxt(result_file, y_pred_subset, fmt='%.5e')

    y_pred_median_subset=y_pred_median[:,g_idx]
    y_q1_subset=y_q1[:,g_idx]
    y_q3_subset=y_q3[:,g_idx]
    y_pred_var_subset=y_pred_var[:,g_idx]

    #
    # tab file
    #
    
    g_name_subset = [g_name_sur[i] for i in g_idx]

    with open('../surrogateTab.out', 'a') as tab_file:
        # write header
        if os.path.getsize('../surrogateTab.out') == 0:
            tab_file.write("%eval_id interface "+ " ".join(rv_name_sur) + " "+ " ".join(g_name_subset) + " " + ".median ".join(g_name_subset) + ".median "+ ".q5 ".join(g_name_subset) + ".q5 "+ ".q95 ".join(g_name_subset) + ".q95 " +".var ".join(g_name_subset) + ".var \n")
        # write values

        for ns in range(nsamp):
            rv_list = " ".join("{:e}".format(rv)  for rv in rv_val[ns,:])
            ypred_list = " ".join("{:e}".format(yp) for yp in y_pred_subset[ns,:])
            ymedian_list = " ".join("{:e}".format(ym) for ym in y_pred_median_subset[ns,:])
            yQ1_list = " ".join("{:e}".format(yq1)  for yq1 in y_q1_subset[ns,:])
            yQ3_list = " ".join("{:e}".format(yq3) for yq3 in y_q3_subset[ns,:])
            ypredvar_list=" ".join("{:e}".format(ypv)  for ypv in y_pred_var_subset[ns,:])

            tab_file.write(str(sampNum)+" NO_ID "+ rv_list + " "+ ypred_list + " " + ymedian_list+ " "+ yQ1_list + " "+ yQ3_list +" "+ ypredvar_list + " \n")

    error_file.close()
    file_object.close()

def predict(m, X, did_mf):

    if not did_mf:
        return m.predict(X)
    else:
        X_list = convert_x_list_to_array([X, X])
        X_list_l = X_list[:X.shape[0]]
        X_list_h = X_list[X.shape[0]:]
        return m.predict(X_list_h)


if __name__ == "__main__":
    error_file = open('../surrogate.err', "w")
    inputArgs = sys.argv

    if not inputArgs[2].endswith('.json'):
        msg = 'ERROR: surrogte information file (.json) not set'
        print(msg); error_file.write(msg); exit(-1)

    elif not inputArgs[3].endswith('.pkl'):
        msg = 'ERROR: surrogte model file (.pkl) not set'
        print(msg); error_file.write(msg); exit(-1)

    elif len(inputArgs) < 4 or len(inputArgs) > 4:
        msg = 'ERROR: put right number of argv'
        print(msg); error_file.write(msg); exit(-1)

    '''
    params_dir = 'params.in'
    surrogate_dir = 'C:/Users/yisan/Desktop/quoFEMexamples/surrogates/SimGpModel_2_better.pkl'
    result_file = 'results_GP.out'
    '''
    '''
    try:
        opts, args = getopt.getopt(argv)
    except getopt.GetoptError:
        print
        'surrogate_pred.py -i <dir_params.in> -o <dir_SimGpModel.pkl>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'surrogate_pred.py -i <dir_params.in> -o <dir_SimGpModel.pkl>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
   '''



    params_dir = inputArgs[1]
    surrogate_dir = inputArgs[3]
    surrogate_meta_dir = inputArgs[2]
    result_file="results.out"

    sys.exit(main(params_dir,surrogate_dir,surrogate_meta_dir,result_file,'scInput.json'))

