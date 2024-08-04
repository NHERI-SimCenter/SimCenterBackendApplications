import json  # noqa: CPY001, D100, INP001
import os
import time

try:
    moduleName = 'numpy'  # noqa: N816
    import numpy as np

    moduleName = 'scipy'  # noqa: N816
    from scipy.interpolate import interp1d
    from scipy.signal import csd, windows

    error_tag = False  # global variable
except:  # noqa: E722
    error_tag = True

from convertWindMat import *  # noqa: F403


def main(aimName, evtName, getRV):  # noqa: ANN001, ANN201, C901, D103, N803, PLR0914, PLR0915
    # THIS IS PERFORMED ONLY ONCE with open(aimName, 'r', encoding='utf-8') as f:
    with open(aimName) as f:  # noqa: PLW1514, PTH123
        aim_data = json.load(f)

    evt_data = aim_data['Events'][0]

    filename = evt_data['filename']
    # from UI

    V_H = evt_data['windSpeed']  # wind speed at full scale (vel)  # noqa: N806
    T_full = evt_data[  # noqa: N806
        'fullScaleDuration'
    ]  # Duration of wind load at full scale (time)
    perc_mod = evt_data[
        'modePercent'
    ]  # percentage of modes to include in the simulation
    seed = evt_data['seed']
    # ^ Choose percentage of modes to include in the simulation (%). We suggest between 25% and 30% for higher accuracy

    #
    # Parsing the json file
    #

    if filename.endswith('.mat'):
        mat_filenmae = filename
        base = os.path.splitext(mat_filenmae)[0]  # noqa: PTH122
        json_filename = base + '.json'

        if getRV:
            parseWindMatFile(mat_filenmae, json_filename)  # noqa: F405
            os.remove(mat_filenmae)  # noqa: PTH107

        filename = json_filename

    with open(filename, encoding='utf-8') as jsonFile:  # noqa: PTH123, N806
        data = json.load(jsonFile)

    if not getRV:
        case = 'PODmodes'

    elif evt_data['type'] == 'WindForceSpectrum':  # creates {forceSpectra}.json
        if ('s_target_real' not in data) or ('s_target_imag' not in data):
            raise Exception(  # noqa: TRY002
                'Target Spectrum info not found in ' + evt_data['filename'] + '.'
            )

        case = 'spectra'

    elif (
        evt_data['type'] == 'ExperimentalWindForces'
    ):  # creates {forceTimehistory}.json here and later overwrites it with {forceSpectra}.json
        if ('Fx' not in data) or ('Fy' not in data) or ('Tz' not in data):
            raise Exception(  # noqa: TRY002
                'Force time histories not found in ' + evt_data['filename'] + '.'
            )

        case = 'timeHistory'

    # elif not getRV:

    #    # read {forceSpectra}.json

    #    case = "spectra"

    else:
        raise Exception('Event type [' + evt_data['type'] + '] not found.')  # noqa: TRY002

    D = data['D']  # noqa: N806
    H = data['H']  # noqa: N806
    B = data['B']  # noqa: N806
    fs = data['fs']
    vRef = data['Vref']  # noqa: N806

    #
    # check if model scale is found in the key
    #

    ms = evt_data.get('modelScale', 0)  # model scale
    if ms == 0:  # when mat file is imported, model scale is not precalculated
        print('Model scale not found. Calculating the unified model scale..')  # noqa: T201
        D_full = aim_data['GeneralInformation']['depth']  # noqa: N806
        H_full = aim_data['GeneralInformation']['height']  # noqa: N806
        B_full = aim_data['GeneralInformation']['width']  # noqa: N806
        ms = H_full / H
        print(f'Model scaling factor of {ms:.2} is used')  # noqa: T201
        if ((ms != D_full / D) or (ms != B_full / B)) and getRV:
            print(  # noqa: T201
                f'Warning: target-data geometry scaling ratio is inconsistent: H={H_full / H:.2}, B={B_full / B:.2}, D={D_full / D:.2}'
            )

    if case == 'timeHistory':
        # Tw = 4          # duration of window (sec) - user defined - smaller window leads to more smoothing
        # overlap = 0.5   # 50% overlap - user defined
        Tw = evt_data['windowSize']  # noqa: N806
        overlap = evt_data['overlapPerc'] / 100

        Fx = np.array(data['Fx'])  # noqa: N806
        Fy = np.array(data['Fy'])  # noqa: N806
        Tz = np.array(data['Tz'])  # noqa: N806

        t = data['t']  # noqa: F841
        N = Fx.shape[1]  # noqa: N806
        nfloors = Fx.shape[0]
        nfloors_GI = aim_data['GeneralInformation']['NumberOfStories']  # noqa: N806

        if nfloors != nfloors_GI:
            err_exit(
                f'Number of floors does not match - input file has {nfloors} floors, GI tab defines {nfloors_GI} floors'
            )

    elif case == 'spectra':
        s_target_real = np.array(data['s_target_real'])
        s_target_imag = np.array(data['s_target_imag'])
        s_target = s_target_real + 1j * s_target_imag
        f_target = np.array(data['f_target'])
        norm_all = np.array(data['norm_all'])
        comp_CFmean = np.array(data['comp_CFmean'])  # noqa: N806

    elif case == 'PODmodes':
        V_imag = np.array(data['V_imag'])  # noqa: N806
        V_real = np.array(data['V_real'])  # noqa: N806
        V = V_real + 1j * V_imag  # noqa: N806
        D1 = np.array(data['D1'])  # noqa: N806
        SpeN = data['SpeN']  # noqa: N806
        f_target = np.array(data['f_target'])
        norm_all = np.array(data['norm_all'])
        comp_CFmean = np.array(data['comp_CFmean'])  # noqa: N806
    #
    # Below here is fully parameterized
    #

    #
    # Compute the basic quantities
    #

    dtm = 1 / fs  # time step model scale  # noqa: F841
    fc = fs / 2  # Nyquist Frequency (Hz)  wind tunnel
    fp = fs / ms  # scaled frequency
    fcut = fc / ms  # scaled Nyquist frequency
    air_dens = 1.225  # (kg/m3) at 15 oC, sea level
    ndir = 3  # number of coordinate axes (X,Y,Z)

    if case == 'timeHistory':  # Experimental wind forces
        T = N / fs  # duration of simulation in model scale (s)  # noqa: N806, F841
        ncomp = nfloors * ndir  # total number of force components

    elif case == 'spectra' or case == 'PODmodes':  # noqa: PLR1714
        ncomp = comp_CFmean.shape[0]
        nfloors = int(ncomp / ndir)

    #
    # Number of modes to be included
    #

    l_mo = int(
        np.round(ncomp * ((perc_mod) / 100) + 1.0e-10)
    )  # small value added to make .5 round up
    if l_mo > 100 or l_mo < 0:  # noqa: PLR2004
        msg = 'Error: Number of modes should be equal or less than the number of components'  # noqa: F841

    print('Number of modes = ' + str(l_mo))  # noqa: T201

    #
    # Scaling building geometry
    #

    B_full = B * ms  # full scale  # noqa: N806
    D_full = D * ms  # full scale  # noqa: N806
    H_full = H * ms  # full scale  # noqa: N806
    MaxD_full = max(D_full, B_full)  # full scale  # noqa: N806

    #
    # Get CPSD
    #

    if case == 'timeHistory':
        [s_target, f_target, norm_all, comp_CFmean, Fx_full, Fy_full, Tz_full] = (  # noqa: F841, N806
            learn_CPSD(
                Fx,
                Fy,
                Tz,
                ms,
                air_dens,
                vRef,
                H_full,
                B_full,
                D_full,
                MaxD_full,
                fs,
                Tw,
                overlap,
                fp,
                V_H,
                fcut,
                T_full,
            )
        )

    #
    # Eigen decomposition
    #

    if (case == 'timeHistory') or (case == 'spectra'):  # noqa: PLR1714
        V, D1, SpeN = perform_POD(s_target, f_target, ncomp, l_mo)  # noqa: N806

        if getRV:
            #    # let us overwrite the json file.
            createPODJson(  # noqa: F405
                filename,
                V,
                D1,
                SpeN,
                f_target,
                norm_all,
                D,
                H,
                B,
                fs,
                vRef,
                comp_CFmean,
            )

    #
    # Simulation of Gaussian Stochastic wind force coefficients
    #

    f_full = f_target[0:]  # don't exclude freq = 0 Hz
    f_vH = (V_H / vRef) * f_full  # scaledfreq.(Hz)  # noqa: N806
    V_vH = V  # scaled eigenmodes  # noqa: N806
    D_vH = (V_H / vRef) ** 3 * D1  # scaled eigenvalues  # noqa: N806
    theta_vH = np.arctan2(np.imag(V_vH), np.real(V_vH))  # scaled theta  # noqa: N806
    fcut_sc = (V_H / vRef) * fcut
    f_inc = 1 / T_full  # freq.incremen
    # number of freq.points consideredt(Hz)
    N_f = round(T_full * fcut_sc) + 1  # noqa: N806
    dt = 1 / (2 * fcut_sc)  # max.time incremen to avoid aliasing(s)
    N_t = round(T_full / dt)  # number of time points  # noqa: N806
    fvec = np.arange(0, f_inc * (N_f), f_inc)  # frequency line  # noqa: F841
    tvec = np.arange(0, dt * (N_t), dt)  # time line
    f = f_vH[0:SpeN]  # frequencies from the decomposition upto SpeN points(Hz)
    nf_dir = np.arange(ncomp)  # vector number of components

    Nsim = 1  # Number of realizations to be generated  # noqa: N806
    seeds = np.arange(seed, Nsim + seed)  # Set seeds for reproducibility

    CF_sim0 = np.zeros((len(seeds), ncomp, N_t))  # noqa: N806
    for seed_num in range(len(seeds)):
        print(f'Creating Realization # {seed_num + 1} among {len(seeds)} ')  # noqa: T201
        t_init = time.time()

        F_jzm = simulation_gaussian(  # noqa: N806
            ncomp,
            N_t,
            V_vH,
            D_vH,
            theta_vH,
            nf_dir,
            N_f,
            f_inc,
            f,
            l_mo,
            tvec,
            SpeN,
            V_H,
            vRef,
            seeds,
            seed_num,
        )
        CF_sim0[seed_num, :, :] = (
            F_jzm  # zero-mean force coefficient time series (simulation)
        )

        print(f' - Elapsed time: {time.time() - t_init:.3} seconds.\n')  # noqa: T201

    #
    # Destandardize force coefficients
    #
    #

    CF_sim1 = (  # noqa: N806
        np.transpose(CF_sim0, (1, 2, 0)) / (V_H / vRef) ** 3 / np.sqrt(V_H / vRef)
    )  # rescale Force Coefficients

    CF_sim = CF_sim1 * np.transpose(  # noqa: N806
        norm_all[np.newaxis, np.newaxis], (2, 1, 0)
    ) + np.transpose(
        comp_CFmean[np.newaxis, np.newaxis], (2, 1, 0)
    )  # force coefficients
    # Transform back the Force Coefficients into Forces (N)
    static_pres = np.vstack(
        (
            np.ones((nfloors, 1, 1)) * (0.5 * air_dens * vRef**2 * H_full * B_full),
            np.ones((nfloors, 1, 1)) * (0.5 * air_dens * vRef**2 * H_full * D_full),
            np.ones((nfloors, 1, 1))
            * (0.5 * air_dens * vRef**2 * H_full * MaxD_full**2 / 2),
        )
    )
    F_sim = (  # noqa: N806
        (V_H / vRef) ** 2 * CF_sim * static_pres
    )  # simulated forces at full scale wind speed

    # return F_sim

    #
    # Writing results to an event file
    #

    if getRV:
        F_sim = np.zeros(F_sim.shape)  # noqa: N806

    evtInfo = {}  # noqa: N806

    evtInfo['dT'] = tvec[1] - tvec[0]
    evtInfo['numSteps'] = tvec.shape[0]

    patterns = []
    id_timeseries = 0
    ts_dof_info = []
    ts_floor_info = []
    for nd in range(ndir):
        for nf in range(nfloors):
            id_timeseries += 1
            my_pattern = {}
            if nd == 0 or nd == 1:  # noqa: PLR1714
                my_pattern['dof'] = nd + 1  # x and y dir
            elif nd == 2:  # noqa: PLR2004
                my_pattern['dof'] = 6  # moments

            my_pattern['floor'] = str(nf + 1)
            my_pattern['name'] = str(id_timeseries)
            my_pattern['staticWindLoad'] = 0.0
            my_pattern['timeSeries'] = str(id_timeseries)
            my_pattern['type'] = 'WindFloorLoad'
            patterns += [my_pattern]
            ts_dof_info += [nd]
            ts_floor_info += [nf]

    evtInfo['pattern'] = patterns
    evtInfo['subtype'] = 'ExperimentalWindForces'
    evtInfo['type'] = 'Wind'

    timeSeries = []  # noqa: N806
    for id in range(id_timeseries):  # noqa: A001
        my_ts = {}
        my_ts['dT'] = tvec[1] - tvec[0]
        my_ts['name'] = str(id + 1)
        my_ts['type'] = 'Value'

        cur_dof = ts_dof_info[id]
        cur_floor = ts_floor_info[id]
        my_ts['data'] = F_sim[(cur_dof) * nfloors + cur_floor, :, 0].tolist()

        timeSeries += [my_ts]  # noqa: N806

    evtInfo['timeSeries'] = timeSeries

    with open(evtName, 'w', encoding='utf-8') as fp:  # noqa: PTH123
        json.dump({'Events': [evtInfo]}, fp)

    """
    # plotting
    import matplotlib.pyplot as plt
    # Plots of time series at different floors

    plt.plot(tvec, F_sim[9 , :, 1] / 1000,lw=1)
    if case == "timeHistory":
        t_sc = t * ms * (vRef / V_H) # scale wind tunnel time series to compare
        plt.plot(t_sc, (V_H / vRef) ** 2 * Fx_full[9 ,: ] / 1000,lw=1)
    plt.xlabel('t(s)')
    plt.ylabel('Fx (kN)')
    plt.title('Force - Floor = 10 - Full Scale')
    plt.show()

    plt.plot(tvec, F_sim[34,:, 1] / 1000)
    if case == "timeHistory":
        plt.plot(t_sc, (V_H / vRef) ** 2 * Fy_full[9,:] / 1000) # scaled wind tunnel record
    plt.xlabel('t(s)')
    plt.ylabel('Fy (kN)')
    plt.title('Force - Floor = 10 - Full Scale')
    plt.show()

    plt.plot(tvec,  F_sim[59,:, 1]/ 1000)
    if case == "timeHistory":
        plt.plot(t_sc, (V_H / vRef) ** 2 * Tz_full[9,:] / 1000)
    plt.xlabel('t(s)')
    plt.ylabel('Tz (kN)')
    plt.title('Force - Floor = 10 - Full Scale')
    plt.show()

    plt.plot(tvec, F_sim[19,:, 1] / 1000)
    if case == "timeHistory":
        plt.plot(t_sc, (V_H / vRef) ** 2 * Fx_full[19,:] / 1000)
    plt.xlabel('t(s)')
    plt.ylabel('Fx (kN)')
    plt.title('Force - Floor = 20 - Full Scale')
    plt.show()

    plt.plot(tvec, F_sim[45,:, 1] / 1000)
    if case == "timeHistory":
        plt.plot(t_sc, (V_H / vRef) ** 2 * Fy_full[19,:] / 1000)  # scaled wind tunnel record
    plt.xlabel('t(s)')
    plt.ylabel('Fx (kN)')
    plt.title('Force - Floor = 20 - Full Scale')
    plt.show()

    plt.plot(tvec, F_sim[69,:, 1] / 1000)
    if case == "timeHistory":
        plt.plot(t_sc, (V_H / vRef) ** 2 * Tz_full[19,:] / 1000)
    plt.xlabel('t(s)')
    plt.ylabel('Tz (kN)')
    plt.title('Force - Floor = 20 - Full Scale')
    plt.show()

    """


def perform_POD(s_target, f_target, ncomp, l_mo):  # noqa: ANN001, ANN201, N802, D103
    S_F = s_target[:, :, 0:]  # do not exclude freq = 0 Hz  # noqa: N806
    f_full = f_target[0:]  # do not exclude freq = 0 Hz

    SpeN = f_full.shape[0]  # exclude freq = 0 Hz  # noqa: N806

    Vs = np.zeros((ncomp, ncomp, SpeN), dtype='complex_')  # noqa: N806
    Ds = np.zeros((ncomp, ncomp, SpeN))  # noqa: N806

    for ii in range(
        SpeN
    ):  # eigen - decomposition at every frequency of CPSD matrix and sort them
        [D_all, V_all] = np.linalg.eig(S_F[:, :, ii])  # noqa: N806
        ind = np.argsort(D_all)
        Ds[:, :, ii] = np.real(np.diag(D_all[ind]))
        Vs[:, :, ii] = V_all[:, ind]

    # Truncation
    V = np.zeros((ncomp, l_mo, SpeN), dtype='complex_')  # noqa: N806
    D0 = np.zeros((l_mo, l_mo, SpeN))  # noqa: N806
    for tt in range(l_mo):
        V[:, tt, :] = Vs[:, ncomp - 1 - tt, :]
        D0[tt, tt, :] = Ds[ncomp - 1 - tt, ncomp - 1 - tt, :]

    D1 = np.zeros((l_mo, 1, SpeN))  # noqa: N806
    for ii in range(SpeN):
        D1[:, 0, ii] = np.diag(D0[:, :, ii])

    return V, D1, SpeN


def learn_CPSD(  # noqa: ANN201, D103, N802, PLR0913, PLR0917
    Fx,  # noqa: ANN001, N803
    Fy,  # noqa: ANN001, N803
    Tz,  # noqa: ANN001, N803
    ms,  # noqa: ANN001
    air_dens,  # noqa: ANN001
    vRef,  # noqa: ANN001, N803
    H_full,  # noqa: ANN001, N803
    B_full,  # noqa: ANN001, N803
    D_full,  # noqa: ANN001, N803
    MaxD_full,  # noqa: ANN001, N803
    fs,  # noqa: ANN001
    Tw,  # noqa: ANN001, N803
    overlap,  # noqa: ANN001
    fp,  # noqa: ANN001
    V_H,  # noqa: ANN001, N803
    fcut,  # noqa: ANN001
    T_full,  # noqa: ANN001, N803
):
    Fx_full = ms**2 * Fx  # full scale Fx(N)  # noqa: N806
    Fy_full = ms**2 * Fy  # full scale  Fy(N)  # noqa: N806
    Tz_full = ms**3 * Tz  # full scale Tz(N.m)  # noqa: N806

    # Force Coefficients (unitless)
    CFx = Fx_full / (0.5 * air_dens * vRef**2 * H_full * B_full)  # noqa: N806
    CFy = Fy_full / (0.5 * air_dens * vRef**2 * H_full * D_full)  # noqa: N806
    CTz = Tz_full / (0.5 * air_dens * vRef**2 * H_full * MaxD_full**2 / 2)  # noqa: N806

    # Mean Force Coefficients
    CFx_mean = np.mean(CFx, axis=1)  # noqa: N806
    CFy_mean = np.mean(CFy, axis=1)  # noqa: N806
    CTz_mean = np.mean(CTz, axis=1)  # noqa: N806

    comp_CFmean = np.concatenate([CFx_mean, CFy_mean, CTz_mean])  # noqa: N806

    RF = 3.5  # Reduction Factor  # noqa: N806

    # Normalization factor
    xnorm = np.std(CFx - CFx_mean[np.newaxis].T, axis=1) * RF
    ynorm = np.std(CFy - CFy_mean[np.newaxis].T, axis=1) * RF
    tornorm = np.std(CTz - CTz_mean[np.newaxis].T, axis=1) * RF
    norm_all = np.concatenate([xnorm, ynorm, tornorm])

    # Standardazation of Forces (force coeff have about the same range)
    CFx_norm = (CFx - np.mean(CFx, axis=1)[np.newaxis].T) / xnorm[np.newaxis].T  # noqa: N806
    CFy_norm = (CFy - np.mean(CFy, axis=1)[np.newaxis].T) / ynorm[np.newaxis].T  # noqa: N806
    CTz_norm = (CTz - np.mean(CTz, axis=1)[np.newaxis].T) / tornorm[np.newaxis].T  # noqa: N806
    Components = np.vstack([CFx_norm, CFy_norm, CTz_norm]).T  # noqa: N806

    # Smoothed target CPSD
    wind_size = fs * Tw
    nover = round(overlap * wind_size)

    # nfft = int(wind_size)
    fcut_sc = (V_H / vRef) * fcut
    dt = 1 / (2 * fcut_sc)  # max.time incremen to avoid aliasing(s)
    N_t = round(T_full / dt)  # number of time points  # noqa: N806
    nfft = N_t

    t_init = time.time()
    # [s_target,f_target] = cpsd(Components,Components,hanning(wind_size),nover,nfft,fp,'mimo');
    s_target, f_target = cpsd_matlab(
        Components, Components, wind_size, nover, nfft, fp
    )

    print(f' - Elapsed time: {time.time() - t_init:.3} seconds.\n')  # noqa: T201

    return s_target, f_target, norm_all, comp_CFmean, Fx_full, Fy_full, Tz_full


def cpsd_matlab(Components1, Components2, wind_size, nover, nfft, fp):  # noqa: ANN001, ANN201, N803, D103
    window = windows.hann(int(wind_size))

    ncombs1 = Components1.shape[1]
    ncombs2 = Components2.shape[1]
    nSampPoints = int(nfft / 2 + 1)  # noqa: N806
    s_target = np.zeros((ncombs1, ncombs2, nSampPoints), dtype='complex_')

    print('Training cross power spectrum density..')  # noqa: T201
    for nc2 in range(ncombs2):
        for nc1 in range(ncombs1):
            [f_target, s_tmp] = csd(
                Components1[:, nc1],
                Components2[:, nc2],
                window=window,
                noverlap=nover,
                nfft=nfft,
                fs=fp,
            )
            s_target[nc1, nc2, :] = s_tmp  # *4/np.pi

    return s_target, f_target


def simulation_gaussian(  # noqa: ANN201, D103, PLR0913, PLR0917
    ncomp,  # noqa: ANN001
    N_t,  # noqa: ANN001, N803
    V_vH,  # noqa: ANN001, N803
    D_vH,  # noqa: ANN001, N803
    theta_vH,  # noqa: ANN001, N803
    nf_dir,  # noqa: ANN001
    N_f,  # noqa: ANN001, N803
    f_inc,  # noqa: ANN001
    f,  # noqa: ANN001
    l_mo,  # noqa: ANN001
    tvec,  # noqa: ANN001
    SpeN,  # noqa: ANN001, ARG001, N803
    V_H,  # noqa: ANN001, N803
    vRef,  # noqa: ANN001, N803
    seed,  # noqa: ANN001
    seed_num,  # noqa: ANN001
):
    #
    # Set Seed
    #

    folderName = os.path.basename(  # noqa: PTH119, N806
        os.getcwd()  # noqa: PTH109
    )  # Lets get n from workdir.n and add this to the seed
    sampNum = folderName.split('.')[-1]  # noqa: N806

    if sampNum == 'templatedir':
        np.random.seed(seed[seed_num])
    else:
        np.random.seed(seed[seed_num] + int(sampNum))

    # force coefficients initialize matrix
    F_jzm = np.zeros((ncomp, N_t))  # noqa: N806
    f_tmp = np.linspace(0, (N_f - 1) * f_inc, N_f)

    for m in range(l_mo):
        mo = m  # current        mode  #
        Vmo = V_vH[nf_dir, mo, :]  # eigenvector for mode mo  # noqa: N806
        # Dmo = D_vH[mo, 0,:] # eigenvalue for mode mo
        # To avoid nan when calculating VDmo
        Dmo = D_vH[mo, 0, :] + 1j * 0  # noqa: N806

        thetmo = theta_vH[nf_dir, mo, :]  # theta for mode mo
        VDmo = (  # noqa: N806
            np.sqrt((V_H / vRef) ** 3)
            * np.abs(Vmo)
            * (np.ones((ncomp, 1)) * np.sqrt(Dmo))
        )  # product of eigenvector X

        # Generate  random phase  angle for each frequency SpeN
        varth = (2 * np.pi) * np.random.random(size=(1, N_f))

        # Loop over floors
        # g_jm = np.zeros((N_t, ncomp),dtype = 'complex_')
        F_jm = np.zeros((ncomp, N_t))  # noqa: N806

        coef = np.sqrt(2) * np.sqrt(f_inc) * np.exp(1j * varth)
        coef2 = np.exp(1j * ((mo + 1) / l_mo * f_inc) * tvec)

        fVDmo = interp1d(f, VDmo, kind='linear', fill_value='extrapolate')  # noqa: N806
        fthetmo = interp1d(f, thetmo, kind='linear', fill_value='extrapolate')
        fV_interp = np.abs(fVDmo(f_tmp))  # noqa: N806
        fthet_interp = np.exp((1j) * (fthetmo(f_tmp)))

        for j in range(ncomp):
            # l denotes a particular freq. point
            # m denotes a particular mode
            # j denotes a particular floor
            fVDmo = interp1d(f, VDmo[j, :], kind='linear', fill_value='extrapolate')  # noqa: N806
            fthetmo = interp1d(
                f, thetmo[j, :], kind='linear', fill_value='extrapolate'
            )

            B_jm = np.zeros((N_t,), dtype='complex_')  # noqa: N806
            B_jm[0:N_f] = coef * fV_interp[j, :] * fthet_interp[j, :]

            g_jm = np.fft.ifft(B_jm) * N_t
            F_jm[j, :] = np.real(g_jm * coef2)

        # sum up F from different modes (zero - mean)
        F_jzm = F_jzm + F_jm  # noqa: N806, PLR6104

    return F_jzm


def err_exit(msg):  # noqa: ANN001, ANN201, D103
    print(msg)  # noqa: T201
    with open('../workflow.err', 'w') as f:  # noqa: FURB103, PLW1514, PTH123
        f.write(msg)
    exit(-1)  # noqa: PLR1722


if __name__ == '__main__':
    # parseWindMatFile("Forces_ANG000_phase1.mat", "Forces_ANG000_phase1.json")
    # parseWindMatFile("TargetSpectra_ANG000_phase1.mat", "TargetSpectra_ANG000_phase1.json")

    inputArgs = sys.argv  # noqa: N816, F405

    # set filenames
    aimName = sys.argv[2]  # noqa: N816, F405
    evtName = sys.argv[4]  # noqa: N816, F405

    getRV = False  # noqa: N816
    for myarg in sys.argv:  # noqa: F405
        if myarg == '--getRV':
            getRV = True  # noqa: N816

    if error_tag and getRV:
        with open('../workflow.err', 'w') as f:  # noqa: FURB103, PLW1514, PTH123
            print('Failed to import module ' + moduleName)  # noqa: T201
            f.write(
                'Failed to import module '
                + moduleName
                + '. Please check the python path in the preference'
            )
        exit(-1)  # noqa: PLR1722

    # if getRV:
    #     aimName = aimName + ".sc"

    try:
        main(aimName, evtName, getRV)
    except Exception as err:  # noqa: BLE001
        import traceback

        if getRV:
            with open('../workflow.err', 'w') as f:  # noqa: FURB103, PLW1514, PTH123
                f.write(
                    'Failed in wind load generator preprocessor:'
                    + str(err)
                    + '...'
                    + str(traceback.format_exc())
                )
                print(  # noqa: T201
                    'Failed in wind load generator preprocessor:'
                    + str(err)
                    + '...'
                    + str(traceback.format_exc())
                )
            exit(-1)  # noqa: PLR1722
        else:
            with open('../dakota.err', 'w') as f:  # noqa: FURB103, PLW1514, PTH123
                f.write(
                    'Failed to generate wind load: '
                    + str(err)
                    + '...'
                    + str(traceback.format_exc())
                )
                print(  # noqa: T201
                    'Failed to generate wind load:'
                    + str(err)
                    + '...'
                    + str(traceback.format_exc())
                )
            exit(-1)  # noqa: PLR1722
