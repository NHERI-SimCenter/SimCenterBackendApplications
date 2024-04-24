
import json
import math
import time
import os
try:
    moduleName = "numpy"
    import numpy as np

    moduleName = "scipy"
    from scipy.signal import csd, windows, butter, lfilter
    from scipy.interpolate import interp1d
    from scipy.stats import gaussian_kde, genpareto, norm
    from scipy import interpolate
    import os
    error_tag = False  # global variable
except:
    error_tag = True

from convertWindMat import *

errPath = "./workflow.err"          # error file name
sys.stderr = open(errPath,'w')      # redirecting stderr (this way we can capture all sorts of python errors)
def err_exit(msg):
    print("Failed in wind load generator: "+ msg)                   # display in stdout
    print("Failed in wind load generator: "+ msg, file=sys.stderr)  # display in stderr
    exit(-1)                                                        # exit with non-zero exit code


def main(aimName,evtName, getRV):

    with open(aimName, 'r', encoding='utf-8') as f:
        aim_data = json.load(f)

    evt_data = aim_data["Events"][0]

    filename = evt_data["filename"]

    #
    # User-defined variables
    #

    V_H = evt_data["windSpeed"]            # 4*Vref, wind speed at full scale (m/s)
    T_full = evt_data["fullScaleDuration"]      # 1600, Duration of wind pressure realization at full scale (s)
    # TODO check if there is recommended modes
    perc_mod = evt_data["modePercent"]/100 # percentage of modes to include in the simulation
    seed = evt_data["seed"]                 # Set seeds for reproducibility
    Tw = evt_data["windowSize"]             # 4, window size/duration (sec) - smaller window leads to more smoothing - model scale
    overlap = evt_data["overlapPerc"] / 100  # 0.5   , 50% overlap - user defined
    gg = evt_data["cpsdGroupSize"]             # 5 , User defined: variable that subdivides the CPSD matrix into ggxgg "groups" in order to avoid running out of memory

    ms = evt_data.get("modelScale",0)           # model scale

    selected_taps = np.array(evt_data["selectedTaps"]) # np.arange(91,150+1) - 1 , to start from zero   # selected taps for simulation (1:510 if all taps are included)
    tap = len(selected_taps)
    filtHz = 100           # if applied - filtering high-frequency noise - model scale
                            # set equal to 0 if not applied

    l_mo = int(np.ceil(tap*perc_mod))      # number of modes included in the simulation
    if l_mo>100 or l_mo<0:
        err_exit('Number of modes should be equal or less than the number of components')

    print('Number of modes = '+ str(l_mo))


    #
    # Parameters
    #
    tailTrshd = 5  # Percentage of tail threshold on both tails - Fixed value for all times series
    nl = tailTrshd / 100  # Lower Tail Threshold
    nu = 1 - nl  # Upper Tail Threshold

    if getRV:
        print("Running Get RV")
        do_parallel = True
        runType = aim_data["runType"]

        if do_parallel:

            #
            # Define pool
            #

            if runType == "runningLocal":
                from multiprocessing import Pool
                n_processor = os.cpu_count()
                print("Starting pool")
                tmp = time.time()
                pool = Pool(n_processor)
                print(" - Elapsed time: {:.3f} seconds.\n".format(time.time() - tmp))
            else:
                from mpi4py import MPI
                from mpi4py.futures import MPIPoolExecutor
                world = MPI.COMM_WORLD
                n_processor = world.Get_size()
                pool = MPIPoolExecutor()
                
        if filename.endswith('.mat'):
            pressure_data = sio.loadmat(filename)
            for key in pressure_data:
                #print(key)
                if not key.startswith("__"):
                    pressure_data[key] = pressure_data[key][0]

        elif filename.endswith('.json'):
            with open(filename,'r', encoding='utf-8') as jsonFile:
                pressure_data = json.load(jsonFile)

        fs = np.squeeze(pressure_data["frequency"])
        Vref = np.squeeze(pressure_data["windSpeed"])
        Td = np.squeeze(pressure_data["period"])
        pressure_json = pressure_data["pressureCoefficients"]

        dt = 1/fs
        tvec = np.arange(0,Td,dt) +dt
        Cp_pf = np.zeros((len(tvec),len(pressure_json)))
        id_list = set()
        for tap_info in pressure_json:
            id = np.squeeze(tap_info["id"])
            data = np.squeeze(tap_info["data"])
            Cp_pf[:,id-1] = data
            id_list.add(int(id))
        ang = pressure_data["incidenceAngle"]          # Wind Direction (angle)
        
        '''
        
        import matplotlib.pyplot as plt
        myx = np.array([np.squeeze(a).item() for a in np.squeeze(pressure_data["tapLocations"]["xLoc"])])[selected_taps-1]
        myy = np.array([np.squeeze(a).item() for a in np.squeeze(pressure_data["tapLocations"]["yLoc"])])[selected_taps-1]
        myf = np.array([np.squeeze(a).item() for a in np.squeeze(pressure_data["tapLocations"]["face"])])[selected_taps-1]
        myMean = np.mean(Cp_pf,axis=0)[selected_taps-1]
        id = np.where(np.array(myf)==1)[0]
        plt.scatter(myx[id],myy[id],20,c=myMean[id])
        plt.show()
        
        '''

        if ms==0: # when mat file is imported, model scale is not precalculated
                print("Model scale not found. Calculating the unified model scale..")
                D = np.squeeze(pressure_data["depth"])
                H = np.squeeze(pressure_data["height"])
                B = np.squeeze(pressure_data["breadth"])
                D_full = aim_data["GeneralInformation"]["depth"]
                H_full = aim_data["GeneralInformation"]["height"]
                B_full = aim_data["GeneralInformation"]["width"]
                ms = H_full/H
                print("Model scaling factor of {:.2f} is used".format(ms))
                if (((not ms == D_full/D ) or (not ms == B_full/B )) and getRV):
                    print("Warning: target-data geometry scaling ratio is inconsistent: H={:.2}, B={:.2}, D={:.2}".format(H_full/H,B_full/B,D_full/D))

        if len(set(selected_taps.flatten()).difference(id_list)) > 0:
            msg = "The selected taps are not a subset of your original set: following tabs are not found"
            msg += set(selected_taps.flatten()).difference(set(id_list))
            err_exit(msg)

        # Values for paretotails function

        N = np.size(Cp_pf,1)          # total number of data points
        fc = fs/2               # Nyquist Frequency (Hz)  wind tunnel
        fp = fs/ms              # scaled frequency
        fcut = fc/ms            # scaled Nyquist frequency
        #T = N/fs                # duration of simulation in model scale (s)
        #rng_gen = 'twister'     # random number generator
        #air_dens = 1.225        # (kg/m3) at 15 oC, sea level
        # dtm = 1/fs              #

        # filtering added
        if filtHz > 0:
            n = 2
            Hz = filtHz
            Wn = Hz / (fs / 2)
            [b, a] = butter(n, Wn)
            x = Cp_pf - np.mean(Cp_pf,axis=0)
            #y = filter(b, a, x)
            y = lfilter(b, a, x, axis=0)
            Cp = (y + np.mean(Cp_pf,axis=0))

        #######################################################################################################################
        # Standardization of wind records
        # This step is necessary when doing the proper orthogonal decomposition (POD).
        # Cp have different order of magnitude, and the energy is more evenly distributed
        # when standardized, requiring less modes in the simulation.
        # Pressure Coefficients Time historites

        Cp_std = np.std(Cp,axis=0)                   # std of time series for later use
        Cp_mean = np.mean(Cp,axis=0)                 # mean of time series for later use
        #Cp_norm = np.normalize(Cp)            # standardize Cp time series such that mean = 0 and std = 1 for all taps.
        row_sums = Cp.sum(axis=1)
        Cp_norm = (Cp - Cp_mean)/Cp_std

        # Smoothed target CPSD
        wind_size = fs*Tw
        nover = np.round(overlap*wind_size)
        fcut_sc = (V_H/Vref)*fcut          # scaled cut-off frequency
        dt = 1/(2*fcut_sc)                 # max. time increment to avoid aliasing (s)
        N_t = int(np.round(T_full/dt))             # number of time points
        nfft = N_t




        #
        # Learning CPSD only if needed
        #

        out= np.mod(tap,gg)
        c = (tap-out)/gg
        d = np.arange(0,c+1, dtype=int) * int(gg)
        if out>0:
            d = np.concatenate([d, np.array([d[-1] + out])])

        # TODO: dealing with gpuArray, gather
        nSampPoints = int(nfft/2+1)
        s_target = np.zeros((len(selected_taps),len(selected_taps),nSampPoints),dtype = 'complex_')
        startTime = time.time()
        # TODO: works only if the selected taps are is continuous
        selected_taps_tmp = np.concatenate([selected_taps,[selected_taps[-1]+1]]) # zero is dummy that will not appear in the analysis

        print("Training cross power spectrum density..");
        t_init = time.time()

        nloop = (len(d)-1)*(len(d)-1)
        for i in range(1,len(d)):
            for j in range(1,len(d)):

                if np.mod((i-1)*(len(d)-1) + j,round(nloop/10))==0:
                    print("{:.0f} % completed".format(((i-1)*(len(d)-1) + j)/nloop*100))

                kk=np.arange(d[i-1],d[i])
                ll=np.arange(d[j-1],d[j])
                #ii = np.arange(selected_taps_tmp[d[i-1]],selected_taps_tmp[d[i]])
                #jj = np.arange(selected_taps_tmp[d[j-1]],selected_taps_tmp[d[j]])
                ii = selected_taps_tmp[kk]
                jj = selected_taps_tmp[ll]

                [s,f_target]=cpsd_matlab(Cp_norm[:,ii-1],Cp_norm[:,jj-1],wind_size,nover,nfft,fp) # -1 becuase tab1 is at column 0
                # cpsd_all[kk,ll] = s
                s_target[d[i-1]:d[i],d[j-1]:d[j]] = s

        print(" - Elapsed time: {:.1f} seconds.\n".format(time.time() - t_init))

        unitLength = aim_data["GeneralInformation"]["units"]["length"]
        unitTime = aim_data["GeneralInformation"]["units"]["time"]

        print("Performing POD..");
        t_init = time.time()
        # Spectral Proper Orthogonal Decomposition
        V, D1, SpeN = perform_POD(s_target, f_target, tap, l_mo, pool)
        print(" - Elapsed time: {:.1f} seconds.\n".format(time.time() - t_init))

        #
        # Computing nonGaussian CDFs
        #

        if do_parallel:
            print("Computing nonGaussian CDF in parallel")
            tmp = time.time()
            iterables = ((Cp_norm[:, selected_taps[i]-1],) for i in range(tap))
            try:
                result_objs = list(pool.starmap(getCDF, iterables))
                print(" - Elapsed time: {:.3f} seconds.\n".format(time.time() - tmp))
            except KeyboardInterrupt:
                print("Ctrl+c received, terminating and joining pool.")
                try:
                    self.pool.shutdown()
                except Exception:
                    sys.exit()

            my_cdf_vects = np.zeros((1000, tap))
            my_cdf_x_range = np.zeros((2, tap))
            for i in range(len(result_objs)):
                my_cdf_vects[:,i] = result_objs[i][0]
                my_cdf_x_range[:,i] = result_objs[i][1]
                

        else:
            print("Computing nonGaussian CDF")
            tmp = time.time()
            my_cdf_vects = np.zeros((1000, tap))
            my_cdf_x_range = np.zeros((2, tap))
            for i in range(tap):
                '''
                Cp_temp = Cp_norm[:, selected_taps[i]]
                kernel = gaussian_kde(Cp_temp)
                kernel_cdf = np.vectorize(lambda x: kernel.integrate_box_1d(-np.inf, x))
                my_cdf_x = np.linspace(min(Cp_temp), max(Cp_temp), 1000)  # TODO is 1000 enough?
    
                my_cdf_vects[:, i] = kernel_cdf(my_cdf_x)  # Takes too long to evaluate
                my_cdf_x_range[:, i] = [min(Cp_temp), max(Cp_temp)]
                '''
                my_cdf_vects[:, i], my_cdf_x_range[:, i] = getCDF(Cp_norm[:, selected_taps[i]-1])

            print(" - Elapsed time: {:.1f} seconds.\n".format(time.time() - t_init))

        # Simulation of Gaussian Stochastic wind force coefficients

        #------------------------------------------------------
        # Scale up to wind speed of interest, V_H
        #------------------------------------------------------

        iterm_json = {}
        iterm_json["selected_taps"] = selected_taps
        iterm_json["ms"] = ms
        iterm_json["V_H"] = V_H
        iterm_json["T_full"] = T_full
        iterm_json["Cp_norm"] = Cp_norm
        # iterm_json["Tw"] = Tw
        # iterm_json["overlap"] = overlap
        # iterm_json["nover"] = nover
        iterm_json["dt"] = dt
        # iterm_json["fs"] = fs
        # iterm_json["N_t"] = N_t
        iterm_json["fcut_sc"] = fcut_sc
        iterm_json["Vref"] = Vref
        iterm_json["Cp_std"] = Cp_std
        iterm_json["Cp_mean"] = Cp_mean
        # iterm_json["s_target"] = s_target
        iterm_json["f_target"] = f_target
        iterm_json["pressureData"] = pressure_data
        iterm_json["length"] = unitLength
        iterm_json["time"] = unitTime
        iterm_json["V"] = V
        iterm_json["D1"] = D1
        iterm_json["SpeN"] = SpeN
        iterm_json["my_cdf_vects"] = my_cdf_vects
        iterm_json["my_cdf_x_range"] = my_cdf_x_range
        #
        # save into a file
        #

        if not os.path.exists("../input_File"):
            os.makedirs("../input_File")
        sio.savemat('../input_File/POD_Cp.mat', iterm_json)

        file_loaded = False

    else:
        iterm_json =  sio.loadmat('../input_File/POD_Cp.mat')
        selected_taps =np.squeeze(iterm_json["selected_taps"])
        ms =np.squeeze(iterm_json["ms"])
        V_H =np.squeeze(iterm_json["V_H"])
        T_full =np.squeeze(iterm_json["T_full"])
        Cp_norm =np.squeeze(iterm_json["Cp_norm"])
        #Tw =np.squeeze(iterm_json["Tw"])
        #overlap =np.squeeze(iterm_json["overlap"])
        #nover =np.squeeze(iterm_json["nover"])
        dt =np.squeeze(iterm_json["dt"])
        #fs =np.squeeze(iterm_json["fs"])
        #N_t =np.squeeze(iterm_json["N_t"])
        fcut_sc =np.squeeze(iterm_json["fcut_sc"])
        #s_target =np.squeeze(iterm_json["s_target"])
        f_target =np.squeeze(iterm_json["f_target"])
        Vref =np.squeeze(iterm_json["Vref"])
        Cp_std=np.squeeze(iterm_json["Cp_std"])
        Cp_mean=np.squeeze(iterm_json["Cp_mean"])
        unitLength = np.squeeze(iterm_json["length"])
        unitTime = np.squeeze(iterm_json["time"] )
        V = np.squeeze(iterm_json["V"] )
        D1 = iterm_json["D1"]
        SpeN = np.squeeze(iterm_json["SpeN"])
        my_cdf_vects = np.squeeze(iterm_json["my_cdf_vects"])
        my_cdf_x_range = np.squeeze(iterm_json["my_cdf_x_range"])

        do_parallel = False
        file_loaded = True


    f_full = f_target[0:]  # don't exclude freq = 0 Hz
    f_vH = (V_H / Vref) * f_full  # scaledfreq.(Hz)
    V_vH = V  # scaled eigenmodes
    D_vH = (V_H / Vref) ** 3 * D1  # scaled eigenvalues
    theta_vH = np.arctan2(np.imag(V_vH), np.real(V_vH))  # scaled theta

    f_inc = 1 / T_full  # freq.increment(Hz)
    N_f = round(T_full * np.squeeze(fcut_sc)) + 1  # number of time points

    N_t = round(T_full / dt)  # number of time points
    fvec = np.arange(0, f_inc * (N_f), f_inc)  # frequency line
    t_vec_sc = np.linspace(0,dt*N_t,N_t)  # time line
    f = f_vH[0:SpeN]  # frequencies from the decomposition upto SpeN points(Hz)
    nf_dir = np.arange(tap)  # vector number of components


    Nsim = 1  # Number of realizations to be generated
    seeds = np.arange(seed, Nsim + seed)  # Set seeds for reproducibility

    print("Creating Gaussian Realizations");
    t_init = time.time()

    CP_sim = np.zeros((len(seeds), tap, N_t))
    for seed_num in range(len(seeds)):
        t_init = time.time()
        F_jzm = simulation_gaussian(tap, N_t, V_vH, D_vH, theta_vH, nf_dir, N_f, f_inc, f, l_mo, t_vec_sc, SpeN, V_H, Vref, seeds, seed_num)
        CP_sim[seed_num, :, :] = F_jzm  # zero-mean force coefficient time series (simulation)

    print(" - Elapsed time: {:.1f} seconds.\n".format(time.time() - t_init))

    # TODO: parfor

    print("Creating NonGaussian Realizations");
    if do_parallel:
        Cp_nongauss_kernel = np.zeros((tap,CP_sim.shape[2],len(seeds)))
        print("Running {} simulations in parallel".format(tap))

        tmp = time.time()
        iterables = ((Cp_norm[:, selected_taps[i]-1], CP_sim[seed_num, i, :],nl,nu,my_cdf_vects[:,i],my_cdf_x_range[:,i]) for i in range(tap))
        try:
            result_objs = list(pool.starmap(genCP, iterables))
            print(" - Elapsed time: {:.3f} seconds.\n".format(time.time() - tmp))
        except KeyboardInterrupt:
            print("Ctrl+c received, terminating and joining pool.")
            try:
                self.pool.shutdown()
            except Exception:
                sys.exit()

        Cp_nongauss_kernel = np.zeros((tap, CP_sim.shape[2], len(seeds)))
        Cp_nongauss_kernel[:,:,0] = np.array(result_objs)
                
        
    else:
        Cp_nongauss_kernel = np.zeros((tap,CP_sim.shape[2],len(seeds)))

        print("Running {} simulations in series".format(tap))
        tmp = time.time()
        for seed_num in range(len(seeds)): # always 1
            for i in range(tap):
                Cp_nongauss_kernel[i, :, seed_num] = genCP(Cp_norm[:, selected_taps[i]-1], CP_sim[seed_num, i, :],nl,nu,my_cdf_vects[:,i],my_cdf_x_range[:,i])

        print(" - Elapsed time: {:.3f} seconds.\n".format(time.time() - tmp))

    Cp_std_tmp = Cp_std[selected_taps-1][:,np.newaxis,np.newaxis]
    Cp_mean_tmp = Cp_mean[selected_taps-1][:,np.newaxis,np.newaxis]
    Cp_nongauss = np.transpose(Cp_nongauss_kernel,(0, 2, 1))*np.tile(Cp_std_tmp,(1,len(seeds),N_t))+ np.tile(Cp_mean_tmp,(1,len(seeds),N_t))  # destandardize the time series

    # Convert to Full Scale Pressure time series
    #P_full=Cp_nongauss*(1/2)*air_dens*V_H**2  # Net Pressure values in full scale (Pa)

    # Obs: Ps is the static pressure. Some wind tunnel datasets do not provide this
    # value. Not included in this calculation.



    #
    # Save Results
    #
    print("Saving results")

    pressure_data = iterm_json["pressureData"]

    new_json = {}
    #new_json["period"] = Td*ms*Vref/V_H
    new_json["period"] = t_vec_sc[-1]
    new_json["frequency"] = 1/(t_vec_sc[1]-t_vec_sc[0])

    #new_json["windSpeed"] =float(pressure_data["windSpeed"])
    new_json["windSpeed"] = float(evt_data["windSpeed"])

    new_json["units"]={}
    new_json["units"]["length"] =str(np.squeeze(unitLength))
    new_json["units"]["time"] =str(np.squeeze(unitTime))

    if file_loaded:
        new_json["roofType"] =str(pressure_data["roofType"][0][0][0])
        new_json["breadth"] = float(pressure_data["breadth"][0][0][0][0] * ms)
        new_json["depth"] = float(pressure_data["depth"][0][0][0][0] * ms)
        new_json["height"] = float(pressure_data["height"][0][0][0][0] * ms)
        new_json["pitch"] = float(pressure_data["pitch"][0][0][0][0])
        new_json["incidenceAngle"] =float(pressure_data["incidenceAngle"][0][0][0][0])
        new_taps=[]
        for taps in pressure_data["tapLocations"][0][0][0]:
           if taps["id"][0][0] in selected_taps:
                tmp = {}
                tmp["id"] = int(taps["id"][0][0])
                tmp["xLoc"] = float(taps["xLoc"][0][0])*ms
                tmp["yLoc"] = float(taps["yLoc"][0][0])*ms
                tmp["face"] = int(taps["face"][0][0])
                new_taps += [tmp]

    else:
        new_json["roofType"] =str(pressure_data["roofType"])
        new_json["breadth"] = float(pressure_data["breadth"][0] * ms)
        new_json["depth"] = float(pressure_data["depth"][0] * ms)
        new_json["height"] = float(pressure_data["height"][0] * ms)
        new_json["pitch"] = float(pressure_data["pitch"][0])
        new_json["incidenceAngle"] =float(pressure_data["incidenceAngle"][0])
        new_taps=[]
        for taps in pressure_data["tapLocations"]:
           if taps["id"] in selected_taps:
                tmp = {}
                tmp["id"] = int(taps["id"][0][0])
                tmp["xLoc"] = float(taps["xLoc"][0][0])*ms
                tmp["yLoc"] = float(taps["yLoc"][0][0])*ms
                tmp["face"] = int(taps["face"][0][0])
                new_taps += [tmp];


    new_pressures=[]
    for i in range(len(selected_taps)):
        tmp={}
        tmp["id"] = int(selected_taps[i])
        tmp["data"] = Cp_nongauss[i,0,:].tolist()
        new_pressures += [tmp]

    new_json["pressureCoefficients"] = new_pressures
    new_json["tapLocations"] = new_taps
    # %% Plots for verification of code
    #

    with open('tmpSimCenterLowRiseTPU.json', 'w', encoding='utf-8') as f:
        json.dump(new_json, f)

    #curScriptPath = abspath(getsourcefile(lambda:0))
    curScriptPath = os.path.realpath(__file__)
    creatEVENTDir = os.path.dirname(os.path.dirname(curScriptPath))

    siteFile = os.path.join(creatEVENTDir, 'LowRiseTPU', 'LowRiseTPU')

    command_line = f"{siteFile} \"--filenameAIM\" {aimName} \"--filenameEVENT\" {evtName}"
    print("Processing pressure->force:")
    print(command_line)
    # run command

    try:
        os.system(command_line)
    except:
        err_exit('Failed to convert pressure to force.')


    #t_sc = ms*(Vref/V_H);   #scale wind tunnel time series to compare

    # #
    # # Pressure coefficients (selected tap 10)
    #
    '''
    t_sc = ms*(Vref/V_H); 
    
    import matplotlib.pyplot as plt

    plt.plot(t_vec_sc,np.squeeze(Cp_nongauss[9,0,:]))
    plt.plot(t_sc*tvec,Cp[:,selected_taps[9]])
    plt.xlabel('t(s)')
    plt.ylabel('Cp')
    plt.title('Cp - Tap 10 - Full Scale')
    plt.legend(['Simulated signal - '+str(T_full)+ 's','Wind Tunnel Data'])
    plt.ylim([-1, 1.6]); plt.xlim([0, 1600])
    plt.show()


    plt.hist(np.squeeze(Cp_nongauss[9,0,:]),bins=100, density=True, fc=(0, 0, 1, 0.5), log=True)
    plt.hist(Cp[:,selected_taps[9]],bins=100,density=True, fc=(1, 0, 0, 0.5), log=True)
    plt.legend(['Sim','Wind Tunnel Data'])
    plt.xlabel('Cp')
    plt.ylabel('PDF')
    #plt.ylim([10**-3,10])
    plt.show()

    # plt.plot(t_vec_sc, np.squeeze(P_full[9, 0,:]))
    # plt.xlabel('Wind Pressure (Pa)')
    # plt.ylabel('t(s)')
    # plt.ylim([-1400,2000])
    # plt.xlim([0,1000])
    # plt.show()
    '''


def genCP(Cp_temp, Cp_sim_temp, nl, nu, my_cdf_vect, my_cdf_x_range):
    #
    # combining the loops to directly send temp instead of dist_kde
    #

    # TODO; why double?

    meanCp = np.mean(Cp_sim_temp)
    stdCp = np.std(Cp_sim_temp)
    F_vvv = (Cp_sim_temp - meanCp) / stdCp

    # cdf points from gaussian distribuition
    cdf_vvv = norm.cdf(F_vvv, 0, 1)
    # force the data being bounded in due to numerical errors that can happen in Matlab when CDF ~0 or ~1;

    cdf_vvv[cdf_vvv < 0.00001] = 0.00001
    cdf_vvv[cdf_vvv > 0.99999] = 0.99999
    # map F_vvv into F_nongauss through inverse cdf of the mix distribution
    # TODO why single precision for cdf_vv?

    return paretotails_icdf(cdf_vvv, nl, nu, Cp_temp, my_cdf_vect, my_cdf_x_range)

def getCDF(Cp_temp):
    kernel = gaussian_kde(Cp_temp)
    kernel_cdf = np.vectorize(lambda x: kernel.integrate_box_1d(-np.inf, x))
    my_cdf_x = np.linspace(min(Cp_temp), max(Cp_temp), 1000)  # TODO is 1000 enough?

    my_cdf_vects = kernel_cdf(my_cdf_x)  # Takes too long to evaluate
    my_cdf_x_range = [min(Cp_temp), max(Cp_temp)]

    return my_cdf_vects, my_cdf_x_range

def paretotails_icdf(pf, nl,nu, temp, my_cdf_vect, my_cdf_x):

    #
    # Learning KDE
    #
    #kernel = gaussian_kde(temp)
    #
    #
    # pareto parts - quick stuff
    #
    lower_temp = temp[temp<np.quantile(temp,nl)]
    upper_temp = temp[temp>np.quantile(temp,nu)]
    #gpareto_param_lower = genpareto.fit(-lower_temp, loc=np.min(-lower_temp))
    #gpareto_param_upper = genpareto.fit(upper_temp, loc=np.min(upper_temp))

    #
    # Estimating CDFs
    #

    icdf_vals = np.zeros((len(pf),))

    #
    # lower pareto
    #
    idx1 = np.where(pf<nl)
    myX = -lower_temp
    c, loc, scal = genpareto.fit(myX, loc=np.min(myX))
    mydist = genpareto(c=c, loc=loc, scale=scal)

    icdf_vals[idx1] = -mydist.ppf(1-(pf[idx1])/(nl))

    #
    # middle kernel
    #

    my_cdf_x = np.linspace(my_cdf_x[0],my_cdf_x[1],my_cdf_vect.shape[0])
    idx2 = np.where((pf>=nl)* (pf<nu))
    # not to have duplicates in x

    unique_val, unique_id = np.unique(my_cdf_vect, return_index=True)

    kernel_icdf = interpolate.interp1d(my_cdf_vect[unique_id], my_cdf_x[unique_id], kind='cubic', bounds_error=False)

    
    icdf_vals[idx2] = kernel_icdf(pf[idx2])

    #
    # upper pareto
    #

    idx3 = np.where(pf>nu)
    myX = upper_temp
    c, loc, scal = genpareto.fit(myX, loc=np.min(myX))
    mydist = genpareto(c=c, loc=loc, scale=scal)

    icdf_vals[idx3] = mydist.ppf(1-(1-pf[idx3])/(1-nu))

    return icdf_vals



    '''
    def paretotails(temp,nl,nu):
    
        # KDE part
        kernel = gaussian_kde(temp)
        #cdf = np.vectorize(lambda x: kernel.integrate_box_1d(-np.inf, x))
    
        # pareto parts
        lower_temp = temp[temp<np.quantile(temp,nl)]
        upper_temp = temp[temp>np.quantile(temp,nu)]
        gpareto_param_lower = genpareto.fit(-lower_temp, loc=np.min(-lower_temp))
        gpareto_param_upper = genpareto.fit(upper_temp, loc=np.min(upper_temp))
        #f_low=genpareto.cdf(x_target,gpareto_param_lower[0],gpareto_param_lower[1],gpareto_param_lower[2])
        #f_upp=genpareto.cdf(x_target,gpareto_param_upper[0],gpareto_param_upper[1],gpareto_param_upper[2])
        #tt=genpareto.isf(1-0.025,gpareto_param_lower[0],gpareto_param_lower[1],gpareto_param_lower[2])
        #genpareto.cdf(tt, gpareto_param_lower[0], gpareto_param_lower[1], gpareto_param_lower[2])
        #f_upp=genpareto.cdf(x_target,gpareto_param_upper[0],gpareto_param_upper[1],gpareto_param_upper[2])
    
    
        c = 0.1
        r = genpareto.rvs(c, size=1000)
        
        
        # r = lower_temp
        # c, loc, scal  =  genpareto.fit(r)
        # mydist = genpareto(c=c,loc=loc,scale=scal)
        # plt.scatter(r,mydist.pdf(r))
        # plt.show()
        # plt.hist(r)
        # plt.show()
    
        myX = upper_temp
        c, loc, scal = genpareto.fit(myX, loc=np.min(myX))
        mydist = genpareto(c=c, loc=loc, scale=scal)
        plt.scatter(myX, mydist.pdf(myX))
        plt.show()
        plt.hist(myX)
        plt.show()
    
        myX = -lower_temp
        c, loc, scal = genpareto.fit(myX, loc=np.min(myX))
        mydist = genpareto(c=c, loc=loc, scale=scal)
        plt.scatter(-myX, mydist.pdf(myX))
        plt.show()
        plt.hist(-myX)
        plt.show()
    '''

    return kernel, gpareto_param_lower, gpareto_param_upper


def cpsd_matlab(Components1,Components2,wind_size,nover,nfft,fp):

    window = windows.hann(int(wind_size))

    ncombs1 = Components1.shape[1]
    ncombs2 = Components2.shape[1]
    nSampPoints = int(nfft/2+1)

    if nfft<2500:
        print("ERROR: time series is too short. Please put a longer duration")
        exit(-1)

    s_target = np.zeros((ncombs1,ncombs2,nSampPoints),dtype = 'complex_')

    for nc2 in range(ncombs2):
        for nc1 in range(ncombs1):
            [f_target,s_tmp] = csd(Components1[:,nc1],Components2[:,nc2],window=window,noverlap = nover,nfft = nfft,fs = fp)
            s_target[nc1,nc2,:] = s_tmp #*4/np.pi

    return s_target, f_target

def perform_POD(s_target, f_target, ncomp, l_mo, pool):
    S_F = s_target[:, :, 0:]  # do not exclude freq = 0 Hz
    f_full = f_target[0:]  # do not exclude freq = 0 Hz

    SpeN = f_full.shape[0]  # exclude freq = 0 Hz

    Vs = np.zeros((ncomp, ncomp, SpeN), dtype='complex_')
    Ds = np.zeros((ncomp, ncomp, SpeN))

    #
    # eigenvalue analysis in parallel
    #
    iterables = ((S_F[:, :, ii],) for ii in range(SpeN))
    try:
        result_objs = list(pool.starmap( np.linalg.eig, iterables))
    except MemoryError:
        err_exit("Low memory performing POD")

    except KeyboardInterrupt:
        print("Ctrl+c received, terminating and joining pool.")
        try:
            self.pool.shutdown()
        except Exception:
            sys.exit()


    for ii in range(SpeN):
        D_all = result_objs[ii][0]
        V_all = result_objs[ii][1]
        ind = np.argsort(D_all)
        Ds[:, :, ii] = np.real(np.diag(D_all[ind]))
        Vs[:, :, ii] = V_all[:, ind]


    # Truncation
    V = np.zeros((ncomp, l_mo, SpeN), dtype='complex_')
    D0 = np.zeros((l_mo, l_mo, SpeN))
    for tt in range(l_mo):
        V[:, tt, :] = Vs[:, ncomp - 1 - tt, :]
        D0[tt, tt, :] = Ds[ncomp - 1 - tt, ncomp - 1 - tt, :]

    D1 = np.zeros((l_mo, 1, SpeN))
    for ii in range(SpeN):
        D1[:, 0, ii] = np.diag(D0[:, :, ii])

    return V, D1, SpeN


def simulation_gaussian(ncomp , N_t, V_vH, D_vH, theta_vH, nf_dir, N_f, f_inc, f, l_mo, tvec, SpeN, V_H, vRef, seed,
                        seed_num):
    #
    # Set Seed
    #

    folderName = os.path.basename(os.getcwd())  # Lets get n from workdir.n and add this to the seed
    sampNum = folderName.split(".")[-1]

    if not sampNum.isnumeric():
        np.random.seed(seed[seed_num])
    else:
        np.random.seed(seed[seed_num] + int(sampNum))

    F_jzm = np.zeros((ncomp, N_t))  # force coefficients initialize matrix
    f_tmp = np.linspace(0, (N_f - 1) * f_inc, N_f)

    for m in range(l_mo):
        mo = m  # current        mode  #
        Vmo = V_vH[nf_dir, mo, :]  # eigenvector for mode mo
        # Dmo = D_vH[mo, 0,:] # eigenvalue for mode mo
        Dmo = D_vH[mo, 0, :] + 1j * 0  # To avoid nan when calculating VDmo

        thetmo = theta_vH[nf_dir, mo, :]  # theta for mode mo
        VDmo = np.sqrt((V_H / vRef) ** 3) * np.abs(Vmo) * (
                    np.ones((ncomp, 1)) * np.sqrt(Dmo))  # product of eigenvector X

        # Generate  random phase  angle for each frequency SpeN
        varth = (2 * np.pi) * np.random.random(size=(1, N_f))

        # Loop over floors
        # g_jm = np.zeros((N_t, ncomp),dtype = 'complex_')
        F_jm = np.zeros((ncomp, N_t))

        coef = np.sqrt(2) * np.sqrt(f_inc) * np.exp(1j * varth)
        coef2 = np.exp(1j * ((mo + 1) / l_mo * f_inc) * tvec)

        fVDmo = interp1d(f, VDmo, kind='linear', fill_value="extrapolate")
        fthetmo = interp1d(f, thetmo, kind='linear', fill_value="extrapolate")
        fV_interp = np.abs(fVDmo(f_tmp))
        fthet_interp = np.exp((1j) * (fthetmo(f_tmp)))

        for j in range(ncomp):
            # l denotes a particular freq. point
            # m denotes a particular mode
            # j denotes a particular floor
            fVDmo = interp1d(f, VDmo[j, :], kind='linear', fill_value="extrapolate")
            fthetmo = interp1d(f, thetmo[j, :], kind='linear', fill_value="extrapolate")

            B_jm = np.zeros((N_t,), dtype='complex_')
            B_jm[0:N_f] = coef * fV_interp[j, :] * fthet_interp[j, :]

            g_jm = np.fft.ifft(B_jm) * N_t
            F_jm[j, :] = np.real(g_jm * coef2)

        #TODO it is hard to tell whether they are similar or not
        F_jzm = F_jzm + F_jm  # sum up F from different modes (zero - mean)

    return F_jzm



    #with open(errPath, "w") as f:
    #    f.write("Failed in wind load generator: "+ msg)


if __name__ == '__main__':

    inputArgs = sys.argv

    # set filenames
    aimName = sys.argv[2]
    evtName = sys.argv[4]

    getRV = False;
    for myarg in sys.argv:
        if (myarg == "--getRV") or (myarg == "getRV"):
            getRV = True;


    if error_tag and getRV:
        err_exit("Failed to import module " + moduleName + " for wind load generator. Please check the python path in the preference")

    # if getRV:
    #     aimName = aimName + ".sc"

    try:
        main(aimName, evtName, getRV)
    except Exception as err:
        import traceback
        if getRV:
            err_exit(str(err) + "..." + str(traceback.format_exc()))

        else:
            err_exit(str(err) + "..." + str(traceback.format_exc()))
