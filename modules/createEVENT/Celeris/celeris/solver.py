import os  # noqa: INP001, D100

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham
import sys
import taichi as ti
import taichi.math as tm
from celeris.utils import (
    CalcUV,
    CalcUV_Sed,
    FrictionCalc,
    NumericalFlux,
    Reconstruct,
    cosh,
    sineWave,
)
from scipy.interpolate import griddata


def checjson(variable, data):  # noqa: D103
    # To check if variables exist in Celeris configuration file
    R = 0  # noqa: N806
    if variable in data:
        R = 1  # noqa: N806
    return R


@ti.data_oriented
class SedClass:  # noqa: D101
    def __init__(
        self,
        d50=0.004,
        p=0.4,
        psi=0.0005,
        CriticalShields=0.045,  # noqa: N803
        rhorat=2.65,
    ):
        self.d50 = d50
        self.p = p
        self.psi = psi
        self.CriticalShields = CriticalShields
        self.rhorat = rhorat
        self.Shields = 1.0 / ((self.rhorat - 1.0) * 9.81 * self.d50 / 1000.0)
        self.C_erosion = self.psi * ti.pow(self.d50 / 1000.0, -0.2)
        self.C_settling = ti.sqrt(
            4.0 / 3.0 * 9.81 * self.d50 / 1000.0 / 0.2 * (self.rhorat - 1.0)
        )
        self.C_rho_sat = (self.p + self.rhorat * (1 - self.p)) / 1000.0


sediment_default = SedClass()  # Check this implementation


@ti.data_oriented
class Solver:  # noqa: D101
    def __init__(  # noqa: C901, PLR0913, PLR0915
        self,
        domain=None,
        boundary_conditions=None,
        dissipation_threshold=0.3,
        theta=2.0,
        timeScheme=2,  # noqa: N803
        pred_or_corrector=1,
        show_window=True,  # noqa: FBT002
        maxsteps=1000,
        Bcoef=1.0 / 15.0,  # noqa: N803
        outdir=None,
        model='SWE',
        useBreakingModel=False,  # noqa: FBT002, N803
        whiteWaterDecayRate=0.01,  # noqa: N803
        whiteWaterDispersion=0.1,  # noqa: N803
        useSedTransModel=False,  # noqa: FBT002, N803
        sediment=sediment_default,
        infiltrationRate=0.001,  # noqa: N803
        clearCon=1,  # noqa: N803
        showBreaking=0,  # noqa: N803
        delta_breaking=2.0,
        T_star_coef=5.0,  # noqa: N803
        dzdt_I_coef=0.50,  # noqa: N803
        dzdt_F_coef=0.15,  # noqa: N803
    ):
        self.domain = domain
        self.bc = boundary_conditions
        self.useSedTransModel = useSedTransModel
        self.clearConc = clearCon
        self.Bottom = self.domain.bottom()  # Bottom [BN, BE, B, Dry/Wet] # B->bottom
        self.precision = self.domain.precision
        # self.temp_bottom = self.domain.bottom()
        # Physical "state" values, eval at center of cell
        self.State = self.domain.states()  # stores the state[eta, P, Q, hc] at t = n
        self.NewState = (
            self.domain.states()
        )  # stores the state[eta, P, Q, hc] at t = n + 1
        self.BottomFriction = self.domain.states()
        self.stateUVstar = (
            self.domain.states()
        )  # the values of the current (n) bous-grouped state, or eta, U, V, and c
        self.current_stateUVstar = self.domain.states()  # next bous-grouped state
        # State variables, texture gives values along cell edges
        self.Hnear = self.domain.states()
        self.H = self.domain.states()  # water depth
        self.U = self.domain.states()  # E-W velocity
        self.V = self.domain.states()  # N-S velocity
        self.C = self.domain.states()  # Scalar concentration
        # Advective flux vectors F(U)-x - G(U)-y
        self.XFlux = self.domain.states()  # stores x-flux values along cell edges
        self.YFlux = self.domain.states()  # stores y-flux values along cell edges
        self.oldGradients = (
            self.domain.states()
        )  # stores d(state)/dt values at previous time step
        self.oldOldGradients = (
            self.domain.states()
        )  # stores d(state)/dt values from two time steps ago
        self.predictedGradients = (
            self.domain.states()
        )  # stores d(state)/dt values found in the predictor step
        self.dU_by_dt = self.domain.states()  # stores d(state)/dt values output from Pass3 calls - Possible can be replaced by self.predictedGradients

        # Bouss
        self.predictedF_G_star = (
            self.domain.states()
        )  # stores F*, G* (bous only) found in predictor step
        self.F_G_star_oldGradients = (
            self.domain.states()
        )  # stores F*, G* (bous only) found at previous time step
        self.F_G_star_oldOldGradients = (
            self.domain.states()
        )  # stores F*, G* (bous only) found from two time steps ago
        # Parallel Cyclic Reduction
        self.coefMatx = (
            self.domain.states()
        )  # tridiagonal coefficients for x-dir (bous only)
        self.coefMaty = (
            self.domain.states()
        )  # tridiagonal coefficients for y-dir (bous only)
        self.temp_PCRx1 = self.domain.states()  # temp storage for PCR x-dir PCR reduced tridiagonal coefficients for x-dir (bous only)
        self.temp_PCRy1 = self.domain.states()  # temp storage for PCR y-dir PCR reduced tridiagonal coefficients for y-dir (bous only)
        self.temp_PCRx2 = self.domain.states()  # temp storage for PCR x-dir
        self.temp_PCRy2 = self.domain.states()  # temp storage for PCR y-dir
        self.temp2_PCRx = self.domain.states()  # temp storage for P solution
        self.temp2_PCRy = self.domain.states()  # temp storage for Q solution

        # Sediment Transport Vectors
        self.sediment = sediment  # Sediment parameters
        self.State_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]
        self.Sed_C = (
            self.domain.states()
        )  # State of sediment class at edges [4,nx,ny]
        self.XFlux_Sed = self.domain.states_one()  # Flux sediment class [1,nx,ny]
        self.YFlux_Sed = self.domain.states_one()  # Flux sediment class [1,nx,ny]
        self.NewState_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]
        self.oldGradients_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]
        self.oldOldGradients_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]
        self.predictedGradients_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]
        self.dU_by_dt_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]
        self.erosion_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]
        self.deposition_Sed = (
            self.domain.states_one()
        )  # State of sediment class [1,nx,ny]

        self.infiltrationRate = infiltrationRate
        self.showBreaking = showBreaking
        self.Auxiliary = self.domain.states()

        self.ShipPressure = self.domain.states_one()
        self.DissipationFlux = self.domain.states()
        self.ContSource = self.domain.states_one()
        self.Breaking = self.domain.states()

        self.R_x = self.domain.reflect_x()
        self.R_y = self.domain.reflect_y()
        self.BCShift = self.domain.Boundary_shift
        self.BoundaryNy = self.domain.Ny - 1
        self.BoundaryNx = self.domain.Nx - 1
        self.dx = self.domain.dx
        self.dy = self.domain.dy
        self.nSL = self.domain.north_sl
        self.sSL = self.domain.south_sl
        self.eSL = self.domain.east_sl
        self.wSL = self.domain.west_sl
        self.seaLevel = self.domain.seaLevel
        self.base_depth = self.domain.maxdepth()
        self.maxtopo = self.domain.maxtopo()
        self.WaveData = self.bc.get_data()
        self.Nwaves = self.bc.N_data
        self.init_eta = self.bc.init_eta
        self.useBreakingModel = useBreakingModel
        self.dissipation_threshold = dissipation_threshold
        self.whiteWaterDecayRate = whiteWaterDecayRate
        self.whiteWaterDispersion = whiteWaterDispersion
        self.delta_breaking = delta_breaking
        self.theta = theta  # Mixing parameter between upwind and centered (1-upwind, 2-centered), typically 1.5
        self.delta = ti.min(0.005, self.base_depth / 5000.0)
        self.epsilon = ti.pow(self.delta, 2)
        self.g = self.domain.g
        self.water_density = 1000.0
        self.dt = self.domain.dt()
        self.isManning = self.domain.isManning
        self.friction = self.domain.friction
        self.timeScheme = timeScheme
        self.pred_or_corrector = pred_or_corrector
        self.pixel = self.domain.pixels
        self.nx = self.domain.Nx
        self.ny = self.domain.Ny
        self.pi = np.pi
        self.show_window = show_window
        self.maxsteps = maxsteps
        self.Bcoef = Bcoef
        # Model parameters
        self.outdir = outdir
        self.model = model
        # Boundary conditions
        self.bcNorth = self.bc.North
        self.bcSouth = self.bc.South
        self.bcEast = self.bc.East
        self.bcWest = self.bc.West
        # Breaking parameters
        self.T_star_coef = T_star_coef
        self.dzdt_F_coef = dzdt_F_coef
        self.dzdt_I_coef = dzdt_I_coef

        # Similitude scaling parameters
        self.similitude = "Froude"
        self.length_scale = 1.0 
        self.time_scale = 1.0
        
        # Force sensor parameters
        self.force_sensor_begin_scaled = [0.0, 0.0]
        self.force_sensor_end_scaled = [1.0, 1.0]
        self.force_sensor_begin_pixel = [0.0, 0.0]
        self.force_sensor_end_pixel = [1.0, 1.0]
        self.hydrostatic_forces = ti.field(dtype=ti.f32, shape=(self.maxsteps))
        self.hydrostatic_forces_from_numpy = ti.field(dtype=ti.f32, shape=(self.maxsteps))
        self.hydrostatic_forces.fill(-1.0)
        self.hydrostatic_forces_numpy = np.zeros((self.maxsteps))
        self.hydrostatic_forces_list = []
        self.hydrostatic_force = ti.field(dtype=ti.f32, shape=(1))
        self.current_U_force = ti.field(dtype=ti.f32, shape=(1))
        self.current_V_force = ti.field(dtype=ti.f32, shape=(1))
        self.current_force = ti.field(dtype=ti.f32, shape=(1))

        # self.hydrostatic_forces_numpy = np.zeros((1, self.maxsteps))
        self.step = 0
        if self.bc.celeris == True:  # noqa: E712
            if checjson('force_sensor_begin', self.bc.configfile) == 1:
                self.force_sensor_begin = self.bc.configfile['force_sensor_begin']
                self.force_sensor_begin_scaled[0] = self.force_sensor_begin[0] / self.bc.configfile['WIDTH']
                self.force_sensor_begin_scaled[1] = self.force_sensor_begin[1] / self.bc.configfile['HEIGHT']
            else:
                self.force_sensor_begin = [0.0, 0.0]
                self.force_sensor_begin_scaled = [0.0, 0.0]
                
            if checjson('force_sensor_end', self.bc.configfile) == 1:
                self.force_sensor_end = self.bc.configfile['force_sensor_end']
                self.force_sensor_end_scaled[0] = self.force_sensor_end[0] / self.bc.configfile['WIDTH']
                self.force_sensor_end_scaled[1] = self.force_sensor_end[1] / self.bc.configfile['HEIGHT']
            else:
                self.force_sensor_end = [self.bc.configfile['WIDTH'], self.bc.configfile['HEIGHT']]
                self.force_sensor_end_scaled = [1.0, 1.0]
                
            if checjson('whiteWaterDecayRate', self.bc.configfile) == 1:
                self.whiteWaterDecayRate = float(
                    self.bc.configfile['whiteWaterDecayRate']
                )
            else:
                self.whiteWaterDecayRate = whiteWaterDecayRate

            if checjson('whiteWaterDispersion', self.bc.configfile) == 1:
                self.whiteWaterDispersion = float(
                    self.bc.configfile['whiteWaterDispersion']
                )
            else:
                self.whiteWaterDispersion = whiteWaterDispersion

            if checjson('dissipation_threshold', self.bc.configfile) == 1:
                self.dissipation_threshold = float(
                    self.bc.configfile['dissipation_threshold']
                )
            else:
                self.dissipation_threshold = dissipation_threshold

            if checjson('delta_breaking', self.bc.configfile) == 1:
                self.delta_breaking = float(self.bc.configfile['delta_breaking'])
            else:
                self.delta_breaking = delta_breaking

            if checjson('Theta', self.bc.configfile) == 1:
                self.theta = float(self.bc.configfile['Theta'])
            else:
                self.theta = theta

            if checjson('Bcoef', self.bc.configfile) == 1:
                self.Bcoef = float(self.bc.configfile['Bcoef'])
            else:
                self.Bcoef = Bcoef

            if checjson('dt', self.bc.configfile) == 1:
                self.dt = float(self.bc.configfile['dt'])
            else:
                self.dt = self.domain.dt()

            if checjson('timeScheme', self.bc.configfile) == 1:
                self.timeScheme = int(float(self.bc.configfile['timeScheme']))
            else:
                self.timeScheme = self.timeScheme

            if checjson('delta', self.bc.configfile) == 1:
                self.delta = float(self.bc.configfile['delta'])
            else:
                self.delta = ti.min(0.005, self.base_depth / 5000.0)

            if checjson('epsilon', self.bc.configfile) == 1:
                self.epsilon = float(self.bc.configfile['epsilon'])
            else:
                self.epsilon = ti.pow(self.delta, 2)

            if checjson('Px', self.bc.configfile) == 1:
                self.Px = int(self.bc.configfile['Px'])
            else:
                self.Px = int(ti.ceil(ti.log(self.nx) / ti.log(2)))

            if checjson('Py', self.bc.configfile) == 1:
                self.Py = int(self.bc.configfile['Py'])
            else:
                self.Py = int(ti.ceil(ti.log(self.ny) / ti.log(2)))

            if checjson('friction', self.bc.configfile) == 1:
                self.friction = int(self.bc.configfile['friction'])
            else:
                self.friction = self.bc.friction

            if checjson('useBreakingModel', self.bc.configfile) == 1:
                self.useBreakingModel = bool(
                    int(self.bc.configfile['useBreakingModel'])
                )
            else:
                self.useBreakingModel = useBreakingModel

            if checjson('showBreaking', self.bc.configfile) == 1:
                self.showBreaking = int(self.bc.configfile['showBreaking'])

            if not self.useBreakingModel:
                self.showBreaking = 0

            if checjson('T_star_coef', self.bc.configfile) == 1:
                # defines length of time until breaking becomes fully developed
                self.T_star_coef = float(self.bc.configfile['T_star_coef'])
            else:
                self.T_star_coef = T_star_coef

            if checjson('dzdt_I_coef', self.bc.configfile) == 1:
                # start breaking parameter
                self.dzdt_I_coef = float(self.bc.configfile['dzdt_I_coef'])
            else:
                self.dzdt_I_coef = dzdt_I_coef

            if checjson('dzdt_F_coef', self.bc.configfile) == 1:
                # end breaking parameter
                self.dzdt_F_coef = float(self.bc.configfile['dzdt_F_coef'])
            else:
                self.dzdt_F_coef = dzdt_F_coef

            if checjson('useSedTransModel', self.bc.configfile) == 1:
                self.useSedTransModel = bool(
                    int(self.bc.configfile['useSedTransModel'])
                )
            else:
                self.useSedTransModel = useSedTransModel

            if checjson('NLSW_or_Bous', self.bc.configfile) == 1:
                # end breaking parameter
                modelo = int(float(self.bc.configfile['NLSW_or_Bous']))
                if modelo == 1:
                    self.model = 'Bouss'
                else:
                    self.model = 'SWE'
            if self.useSedTransModel:
                if checjson('sedC1_d50', self.bc.configfile) == 1:
                    self.sediment.d50 = float(self.bc.configfile['sedC1_d50'])
                if checjson('sedC1_n', self.bc.configfile) == 1:
                    self.sediment.p = float(self.bc.configfile['sedC1_n'])
                if checjson('sedC1_psi', self.bc.configfile) == 1:
                    self.sediment.psi = float(self.bc.configfile['sedC1_psi'])
                if checjson('sedC1_criticalshields', self.bc.configfile) == 1:
                    self.sediment.CriticalShields = float(
                        self.bc.configfile['sedC1_criticalshields']
                    )
                if checjson('sedC1_denrat', self.bc.configfile) == 1:
                    self.sediment.rhorat = float(self.bc.configfile['sedC1_denrat'])
                # Sediment parameters must be computed
                self.sediment.Shields = 1.0 / (
                    (self.sediment.rhorat - 1.0)
                    * self.g
                    * self.sediment.d50
                    / 1000.0
                )
                self.sediment.C_erosion = self.sediment.psi * ti.pow(
                    self.sediment.d50 / 1000.0, -0.2
                )
                self.sediment.C_settling = ti.sqrt(
                    4.0
                    / 3.0
                    * self.g
                    * self.sediment.d50
                    / 1000.0
                    / 0.2
                    * (self.sediment.rhorat - 1.0)
                )
                self.sediment.C_rho_sat = (
                    self.sediment.p + self.sediment.rhorat * (1 - self.sediment.p)
                ) / 1000.0

        # COMPUTATION OF PARAMETERS
        self.one_over_dx = 1 / self.dx
        self.one_over_dy = 1 / self.dy
        self.double_dx = 2.0 * self.dx
        self.double_dy = 2.0 * self.dy
        self.one_over_d2x = self.one_over_dx * self.one_over_dx
        self.one_over_d2y = self.one_over_dy * self.one_over_dy
        self.one_over_dxdy = self.one_over_dx * self.one_over_dy
        self.one_over_d3x = self.one_over_d2x * self.one_over_dx
        self.one_over_d3y = self.one_over_d2y * self.one_over_dy
        self.two_theta = 2 * self.theta
        self.half_g = 0.5 * self.g
        self.g_over_dx = self.g / self.dx
        self.g_over_dy = self.g / self.dy
        self.boundary_epsilon = self.epsilon
        self.boundary_g = self.g
        self.Bcoef_g = self.Bcoef * self.g

    @ti.kernel
    def InitStates(self):  # noqa: N802, D102
        for i, j in self.State:
            self.State[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            self.stateUVstar[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            self.Auxiliary[i, j] = ti.Vector([0.0, 50000.0, 0.0, 0.0], self.precision)

    @ti.kernel
    def fill_bottom_field(self):  # noqa: D102
        lengthCheck = 3  # noqa: N806
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            self.Bottom[0, i, j] = (
                0.5 * self.Bottom[2, i, j]
                + 0.5 * self.Bottom[2, i, ti.min(self.ny - 1, j + 1)]
            )  # BN
            self.Bottom[1, i, j] = (
                0.5 * self.Bottom[2, i, j]
                + 0.5 * self.Bottom[2, ti.min(self.nx - 1, i + 1), j]
            )  # BE
            self.Bottom[3, i, j] = 99.0
            for xx in range(i - lengthCheck, i + lengthCheck + 1):
                ti.loop_config(serialize=True)
                for yy in range(j - lengthCheck, j + lengthCheck + 1):
                    xC = ti.min(self.nx - 1, ti.max(0, xx))  # noqa: N806
                    yC = ti.min(self.ny - 1, ti.max(0, yy))  # noqa: N806

                    if self.Bottom[2, xC, yC] >= 0:
                        self.Bottom[3, i, j] = -99.0

    @ti.func
    def BoundSineWaves(self, NumWaves, Waves, x, y, t, d_here, grav):  # noqa: N802, N803, D102
        result = ti.Vector([0.0, 0.0, 0.0], self.precision)
        x = self.precision(x)
        y = self.precision(y)
        t = self.precision(t)
        grav = self.precision(grav)
        if d_here > 0.0001:  # noqa: PLR2004
            for w in range(NumWaves):
                result += sineWave(
                    x,
                    y,
                    t,
                    d_here,
                    self.precision(Waves[w, 0]),
                    self.precision(Waves[w, 1]),
                    self.precision(Waves[w, 2]),
                    self.precision(Waves[w, 3]),
                    grav,
                    self.bc.WaveType,
                )
        return result

    @ti.func
    def SolitaryWave(self, x0, y0, theta, x, y, t, d_here, amp=0.0):  # noqa: N802, D102
        if amp == 0.0:
            amp = self.bc.amplitude

        # if T == 0.0:
        #     T = self.bc.period
        #     if self.bc.period <= 0.0:
        #         T = 2.0 * self.pi * ti.sqrt(ti.abs(amp) / (self.g * d_here))

        xloc = x - x0
        yloc = y - y0
        k = ti.sqrt(0.75 * ti.abs(amp) / ti.pow(d_here, 3.0))
        c = ti.sqrt(self.g * (amp + d_here))
        eta = amp / ti.pow(
            cosh(k * (xloc * ti.cos(theta) + yloc * ti.sin(theta) - c * t)), 2.0
        )
        hu = ti.sqrt(1.0 + 0.5 * amp / d_here) * eta * c * ti.cos(theta)
        hv = ti.sqrt(1.0 + 0.5 * amp / d_here) * eta * c * ti.sin(theta)
        return eta, hu, hv

    @ti.kernel
    def BoundaryPass(self, time: ti.f32, txState: ti.template()):  # noqa: C901, N802, N803, D102, PLR0915
        # for i,j in txState:
        # make enum for bc type
        BOUNDARY_TYPE_SOLID = 0  # noqa: N806
        BOUNDARY_TYPE_SPONGE = 1  # noqa: N806
        BOUNDARY_TYPE_SINE = 2  # noqa: N806
        BOUNDARY_TYPE_DAM = 3  # noqa: N806, F841

        WAVE_TYPE_SINE = 2  # noqa: N806
        WAVE_TYPE_SOLITARY = 3  # noqa: N806

        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            BCState = txState[i, j]  # noqa: N806
            BCState_Sed = self.State_Sed[i, j].x  # noqa: N806
            BCState_Sed = ti.max(BCState_Sed, 0.0)  # noqa: N806
            ### SPONGE LAYERS
            if (
                self.bcWest == BOUNDARY_TYPE_SPONGE
                and i <= 2 + self.bc.BoundaryWidth
            ):
                gamma = ti.pow(
                    0.5
                    * (
                        0.5
                        + 0.5
                        * ti.cos(
                            self.pi
                            * (self.precision(self.bc.BoundaryWidth - i) + 2.0)
                            / float(self.bc.BoundaryWidth - 1)
                        )
                    ),
                    0.005,
                )
                BCState = txState[i, j] * self.precision(gamma)  # noqa: N806
                BCState_Sed = 0.0  # noqa: N806
            if (
                self.bcEast == BOUNDARY_TYPE_SPONGE
                and i >= self.nx - (self.bc.BoundaryWidth) - 1
            ):
                gamma = ti.pow(
                    0.5
                    * (
                        0.5
                        + 0.5
                        * ti.cos(
                            self.pi
                            * self.precision(
                                self.bc.BoundaryWidth - self.BoundaryNx - i
                            )
                            / float(self.bc.BoundaryWidth - 1)
                        )
                    ),
                    0.005,
                )
                BCState = txState[i, j] * self.precision(gamma)  # noqa: N806
                BCState_Sed = 0.0  # noqa: N806
            if (
                self.bcSouth == BOUNDARY_TYPE_SPONGE
                and j <= 2 + self.bc.BoundaryWidth
            ):
                gamma = ti.pow(
                    0.5
                    * (
                        0.5
                        + 0.5
                        * ti.cos(
                            self.pi
                            * self.precision(self.bc.BoundaryWidth - j + 2.0)
                            / float(self.bc.BoundaryWidth - 1)
                        )
                    ),
                    0.005,
                )
                BCState = txState[i, j] * self.precision(gamma)  # noqa: N806
                BCState_Sed = 0.0  # noqa: N806
            if (
                self.bcNorth == BOUNDARY_TYPE_SPONGE
                and j >= self.ny - self.bc.BoundaryWidth - 1
            ):
                gamma = ti.pow(
                    0.5
                    * (
                        0.5
                        + 0.5
                        * ti.cos(
                            self.pi
                            * self.precision(
                                self.bc.BoundaryWidth - (self.BoundaryNy - j)
                            )
                            / (self.bc.BoundaryWidth - 1)
                        )
                    ),
                    0.005,
                )
                BCState = txState[i, j] * self.precision(gamma)  # noqa: N806
                BCState_Sed = 0.0  # noqa: N806
            ### SOLID WALLS
            if (
                self.bcWest == BOUNDARY_TYPE_SOLID  # noqa: PLR1714
                or self.bcWest == BOUNDARY_TYPE_SPONGE
            ):
                if i <= 1:
                    BCState[0] = txState[self.BCShift - i, j][0]
                    BCState[1] = -txState[self.BCShift - i, j][1]
                    BCState[2] = txState[self.BCShift - i, j][2]
                    BCState[3] = txState[self.BCShift - i, j][3]
                    BCState_Sed = 0.0  # noqa: N806
                elif i == 2:  # noqa: PLR2004
                    BCState[1] = 0.0
                    BCState_Sed = 0.0  # noqa: N806
            if (
                self.bcEast == BOUNDARY_TYPE_SOLID  # noqa: PLR1714
                or self.bcEast == BOUNDARY_TYPE_SPONGE
            ):
                if i >= self.nx - 2:
                    BCState[0] = txState[self.R_x - i, j][0]
                    BCState[1] = -txState[self.R_x - i, j][1]
                    BCState[2] = txState[self.R_x - i, j][2]
                    BCState[3] = txState[self.R_x - i, j][3]
                    BCState_Sed = 0.0  # noqa: N806
                elif i == self.nx - 3:
                    BCState[1] = 0.0
                    BCState_Sed = 0.0  # noqa: N806
            if (
                self.bcSouth == BOUNDARY_TYPE_SOLID  # noqa: PLR1714
                or self.bcSouth == BOUNDARY_TYPE_SPONGE
            ):
                if j <= 1:
                    BCState[0] = txState[i, self.BCShift - j][0]
                    BCState[1] = txState[i, self.BCShift - j][1]
                    BCState[2] = -txState[i, self.BCShift - j][2]
                    BCState[3] = txState[i, self.BCShift - j][3]
                    BCState_Sed = 0.0  # noqa: N806
                elif j == 2:  # noqa: PLR2004
                    BCState[2] = 0.0
                    BCState_Sed = 0.0  # noqa: N806
            if (
                self.bcNorth == BOUNDARY_TYPE_SOLID  # noqa: PLR1714
                or self.bcNorth == BOUNDARY_TYPE_SPONGE
            ):
                if j >= self.ny - 2:
                    BCState[0] = txState[i, self.R_y - j][0]
                    BCState[1] = txState[i, self.R_y - j][1]
                    BCState[2] = -txState[i, self.R_y - j][2]
                    BCState[3] = txState[i, self.R_y - j][3]
                    BCState_Sed = 0.0  # noqa: N806
                elif j == self.ny - 3:
                    BCState[2] = 0.0
                    BCState_Sed = 0.0  # noqa: N806
            ### INCOMING WALLS
            if self.bcWest == BOUNDARY_TYPE_SINE and i <= 2:  # noqa: PLR2004
                if self.bc.WaveType <= WAVE_TYPE_SINE:
                    B_here = -self.base_depth  # noqa: N806
                    d_here = ti.max(0, -B_here)
                    x = i * self.dx
                    y = j * self.dy
                    bcwave = self.BoundSineWaves(
                        self.Nwaves,
                        self.WaveData,
                        x,
                        y,
                        time,
                        d_here,
                        self.boundary_g,
                    )
                    BCState = ti.Vector(  # noqa: N806
                        [bcwave[0] + self.wSL, bcwave[1], bcwave[2], 0.0],
                        self.precision,
                    )
                    BCState_Sed = 0.0  # noqa: N806
                elif self.bc.WaveType == WAVE_TYPE_SOLITARY:
                    # d_here = max(0, self.wSL - self.Bottom[2, i, j])
                    # celerity = ti.sqrt(self.g * (self.bc.amplitude + abs(d_here)))
                    # development_length = celerity * self.bc.period / 1.5
                    # development_length = max(
                    #     development_length,
                    #     ti.sqrt(self.g * abs(self.base_depth))
                    #     * self.bc.period
                    #     / 1.5,
                    # )
                    # x0 = -1.0 * abs(development_length)
                    # y0 = 0.0
                    # eta, hu, hv = self.SolitaryWave(
                    #     x0, y0, 0.0, i * self.dx, j * self.dy, time, d_here
                    # )
                    # BCState = ti.Vector(  # noqa: N806
                    #     [eta + self.wSL, hu, hv, 0.0], self.precision
                    # )
                    d_here = max( 0 , self.wSL -self.Bottom[2,i,j])
                    x0 = -10.0 * self.base_depth # Shift in X
                    y0 =   0.0
                    theta = 0.0
                    eta,hu,hv = self.SolitaryWave(x0, y0, theta, i*self.dx, j*self.dy, time, d_here)
                    BCState = ti.Vector([eta,hu,hv,0.0],self.precision)
                    BCState_Sed = 0.0  # noqa: N806

            if self.bcEast == BOUNDARY_TYPE_SINE and i >= self.nx - 3:
                if self.bc.WaveType <= WAVE_TYPE_SINE:
                    B_here = -self.base_depth  # noqa: N806
                    d_here = ti.max(0, -B_here)
                    x = i * self.dx
                    y = j * self.dy
                    bcwave = self.BoundSineWaves(
                        self.Nwaves,
                        self.WaveData,
                        x,
                        y,
                        time,
                        d_here,
                        self.boundary_g,
                    )
                    BCState = ti.Vector(  # noqa: N806
                        [bcwave[0] + self.eSL, bcwave[1], bcwave[2], 0.0],
                        self.precision,
                    )
                    BCState_Sed = 0.0  # noqa: N806
                elif self.bc.WaveType == WAVE_TYPE_SOLITARY:
                    # d_here = max(0, self.eSL - self.Bottom[2, i, j])
                    # celerity = ti.sqrt(self.g * (self.bc.amplitude + abs(d_here)))
                    # development_length = celerity * self.bc.period / 1.5
                    # development_length = max(
                    #     development_length,
                    #     ti.sqrt(self.g * abs(self.base_depth))
                    #     * self.bc.period
                    #     / 1.5,
                    # )
                    # x0 = self.nx * self.dx + abs(development_length)
                    d_here = max(0, self.eSL - self.Bottom[2,i,j])
                    x0 = self.precision(self.nx*self.dx) + 10.0 * self.base_depth
                    y0 = 0.0
                    theta = -3.1415
                    eta, hu, hv = self.SolitaryWave(
                        x0, y0, theta, i * self.dx, j * self.dy, time, d_here
                    )
                    # BCState = ti.Vector(  # noqa: N806
                    #     [eta + self.eSL, hu, hv, 0.0], self.precision
                    # )
                    BCState = ti.Vector([eta,hu,hv,0.0],self.precision)
                    BCState_Sed = 0.0  # noqa: N806

            if self.bcSouth == BOUNDARY_TYPE_SINE and j <= 2:  # noqa: PLR2004
                if self.bc.WaveType <= WAVE_TYPE_SINE:
                    B_here = -self.base_depth  # noqa: N806
                    d_here = ti.max(0, -B_here)
                    x = i * self.dx
                    y = j * self.dy
                    bcwave = self.BoundSineWaves(
                        self.Nwaves,
                        self.WaveData,
                        x,
                        y,
                        time,
                        d_here,
                        self.boundary_g,
                    )
                    BCState = ti.Vector(  # noqa: N806
                        [bcwave[0] + self.sSL, bcwave[1], bcwave[2], 0.0],
                        self.precision,
                    )
                    BCState_Sed = 0.0  # noqa: N806
                elif self.bc.WaveType == WAVE_TYPE_SOLITARY:
                    # d_here = max(0, self.sSL - self.Bottom[2, i, j])
                    # celerity = ti.sqrt(self.g * (self.bc.amplitude + abs(d_here)))
                    # development_length = celerity * self.bc.period / 1.5
                    # development_length = max(
                    #     development_length,
                    #     ti.sqrt(self.g * abs(self.base_depth))
                    #     * self.bc.period
                    #     / 1.5,
                    # )
                    # x0 = 0.0
                    # y0 = -1.0 * abs(development_length)
                    # theta = 3.1415 / 2.0
                    # eta, hu, hv = self.SolitaryWave(
                    #     x0, y0, theta, i * self.dx, j * self.dy, time, d_here
                    # )
                    # BCState = ti.Vector(  # noqa: N806
                    #     [eta + self.sSL, hu, hv, 0.0], self.precision
                    # )
                    d_here = max( 0 , self.nSL -self.Bottom[2,i,j])
                    x0 = 0.0
                    y0 = -10.0 * self.base_depth
                    theta = 3.1415 / 2.0
                    eta,hu,hv = self.SolitaryWave(x0, y0, theta, i*self.dx, j*self.dy, time, d_here)
                    BCState = ti.Vector([eta,hu,hv,0.0],self.precision)
                    BCState_Sed = 0.0  # noqa: N806

            if self.bcNorth == BOUNDARY_TYPE_SINE and j >= self.ny - 3:
                if self.bc.WaveType <= WAVE_TYPE_SINE:
                    B_here = -self.base_depth  # noqa: N806
                    d_here = ti.max(0, -B_here)
                    x = i * self.dx
                    y = j * self.dy
                    bcwave = self.BoundSineWaves(
                        self.Nwaves,
                        self.WaveData,
                        x,
                        y,
                        time,
                        d_here,
                        self.boundary_g,
                    )
                    BCState = ti.Vector(  # noqa: N806
                        [bcwave[0] + self.nSL, bcwave[1], bcwave[1], 0.0],
                        self.precision,
                    )
                    BCState_Sed = 0.0  # noqa: N806
                # Solitary Waves
                elif self.bc.WaveType == WAVE_TYPE_SOLITARY:
                    # d_here = max(0, self.nSL - self.Bottom[2, i, j])
                    # celerity = ti.sqrt(self.g * (self.bc.amplitude + abs(d_here)))
                    # development_length = celerity * self.bc.period / 1.5
                    # development_length = max(
                    #     development_length,
                    #     ti.sqrt(self.g * abs(self.base_depth))
                    #     * self.bc.period
                    #     / 1.5,
                    # )
                    # x0 = 0.0
                    # y0 = self.ny * self.dy + abs(development_length)
                    # theta = -3.1415 / 2.0
                    # eta, hu, hv = self.SolitaryWave(
                    #     x0, y0, theta, i * self.dx, j * self.dy, time, d_here
                    # )
                    # BCState = ti.Vector(  # noqa: N806
                    #     [eta + self.nSL, hu, hv, 0.0], self.precision
                    # )
                    d_here = max(0, self.nSL - self.Bottom[2, i, j])
                    x0=0.0
                    y0 = self.precision(self.ny*self.dy) + 10.0*self.base_depth
                    theta = -3.1415 / 2.0
                    eta,hu,hv = self.SolitaryWave(x0, y0, theta, i*self.dx, j*self.dy, time, d_here)
                    BCState = ti.Vector([eta,hu,hv,0.0],self.precision)
                    BCState_Sed = 0.0  # noqa: N806
            # Compute the coordinates of the neighbors
            rightIdx = ti.min(i + 1, self.nx - 1)  # noqa: N806
            upIdx = ti.min(j + 1, self.ny - 1)  # noqa: N806
            leftIdx = ti.max(i - 1, 0)  # noqa: N806
            downIdx = ti.max(j - 1, 0)  # noqa: N806

            B_here = self.Bottom[2, i, j]  # noqa: N806
            B_south = self.Bottom[2, i, downIdx]  # noqa: N806
            B_north = self.Bottom[2, i, upIdx]  # noqa: N806
            B_west = self.Bottom[2, leftIdx, j]  # noqa: N806
            B_east = self.Bottom[2, rightIdx, j]  # noqa: N806

            state_south = txState[i, downIdx]
            state_north = txState[i, upIdx]
            state_west = txState[leftIdx, j]
            state_east = txState[rightIdx, j]

            eta_here = BCState.x
            eta_west = state_west.x
            eta_east = state_east.x
            eta_south = state_south.x
            eta_north = state_north.x

            h_here = eta_here - B_here  # CHANGE
            h_south = eta_south - B_south
            h_north = eta_north - B_north
            h_west = eta_west - B_west
            h_east = eta_east - B_east

            h_cut = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            h_cut[0] = ti.max(self.delta, ti.abs(B_here - B_north))
            h_cut[1] = ti.max(self.delta, ti.abs(B_here - B_east))
            h_cut[2] = ti.max(self.delta, ti.abs(B_here - B_south))
            h_cut[3] = ti.max(self.delta, ti.abs(B_here - B_west))

            dry_here = 1
            dry_west = 1
            dry_east = 1
            dry_south = 1
            dry_north = 1

            if h_here <= self.delta:
                dry_here = 0
            if h_west <= h_cut.w:
                dry_west = 0
            if h_east <= h_cut.y:
                dry_east = 0
            if h_south <= h_cut.z:
                dry_south = 0
            if h_north <= h_cut.x:
                dry_north = 0

            sum_dry = dry_west + dry_east + dry_south + dry_north

            h_min = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            h_min[0] = ti.min(h_here, h_north)
            h_min[1] = ti.min(h_here, h_east)
            h_min[2] = ti.min(h_here, h_south)
            h_min[3] = ti.min(h_here, h_west)

            # wetdry = ti.min(h_here, ti.min(h_south, ti.min(h_north, ti.min(h_west, h_east))))
            # nearshore = ti.min(B_here, ti.min(B_south, ti.min(B_north, ti.min(B_west, B_east))))

            # Remove islands
            if dry_here == 1:
                if sum_dry == 0:
                    if B_here <= 0.0:
                        BCState = ti.Vector(  # noqa: N806
                            [ti.max(BCState.x, B_here), 0.0, 0.0, 0.0],
                            self.precision,
                        )
                        BCState_Sed = 0.0  # noqa: N806
                    else:
                        BCState = ti.Vector([B_here, 0.0, 0.0, 0.0], self.precision)  # noqa: N806
                        BCState_Sed = 0.0  # noqa: N806
                elif sum_dry == 1:
                    wet_eta = (
                        float(dry_west) * eta_west
                        + float(dry_east) * eta_east
                        + float(dry_south) * eta_south
                        + float(dry_north) * eta_north
                    ) / float(sum_dry)
                    BCState = ti.Vector([wet_eta, 0.0, 0.0, 0.0], self.precision)  # noqa: N806
                    BCState_Sed = 0.0  # noqa: N806
            # Check for negative depths
            h_here = BCState.x - B_here  # To change
            if h_here <= self.delta:
                if B_here <= 0.0:
                    BCState = ti.Vector(  # noqa: N806
                        [ti.max(BCState.x, B_here), 0.0, 0.0, 0.0], self.precision
                    )
                else:
                    BCState = ti.Vector([B_here, 0.0, 0.0, 0.0], self.precision)  # noqa: N806

                BCState_Sed = 0.0  # noqa: N806

            txState[i, j] = BCState
            self.NewState_Sed[i, j].x = BCState_Sed

    @ti.kernel
    def copy_states(self, src: ti.template(), dst: ti.template()):  # noqa: D102
        ti.static_assert(
            dst.shape == src.shape,
            'copy() needs src and dst fields to be same shape',
        )
        for I in ti.grouped(dst):  # noqa: N806, E741
            dst[I] = src[I]

    @ti.kernel
    def tridiag_coeffs_X(self):  # noqa: N802, D102
        # to calculate the flux terms - Tridiagonal system pag. 12
        Bottom, dx = ti.static(self.Bottom, self.dx)  # noqa: N806
        # ti.loop_config(serialize=True)
        for i in range(self.nx):
            # ti.loop_config(serialize=True)
            for j in range(self.ny):
                a, b, c = 0.0, 0.0, 0.0
                neardry = self.Bottom[3, i, j]
                if i <= 2 or i >= self.nx - 3 or neardry < 0.0:  # noqa: PLR2004
                    a = 0.0
                    b = 1.0
                    c = 0.0
                else:
                    depth_here = -Bottom[2, i, j]
                    depth_plus = -Bottom[2, i + 1, j]
                    depth_minus = -Bottom[2, i - 1, j]
                    # Calculate the first derivative
                    d_dx = (depth_plus - depth_minus) / (2.0 * dx)
                    # Calculate coefficients
                    a = depth_here * d_dx / (6.0 * dx) - (self.Bcoef + 1.0 / 3.0) * (
                        depth_here * depth_here
                    ) / (dx * dx)
                    b = 1.0 + 2.0 * (self.Bcoef + 1.0 / 3.0) * (
                        depth_here * depth_here
                    ) / (dx * dx)
                    c = -depth_here * d_dx / (6.0 * dx) - (
                        self.Bcoef + 1.0 / 3.0
                    ) * (depth_here * depth_here) / (dx * dx)
                self.coefMatx[i, j] = ti.Vector([a, b, c, 0.0], self.precision)

    @ti.kernel
    def tridiag_coeffs_Y(self):  # noqa: N802, D102
        # to calculate the flux terms - Tridiagonal system pag. 12
        Bottom, dy = ti.static(self.Bottom, self.dy)  # noqa: N806
        # ti.loop_config(serialize=True)
        for i in range(self.nx):
            # ti.loop_config(serialize=True)
            for j in range(self.ny):
                a, b, c = 0.0, 0.0, 0.0
                neardry = Bottom[3, i, j]
                if j <= 2 or j >= self.ny - 3 or neardry < 0.0:  # noqa: PLR2004
                    a = 0.0
                    b = 1.0
                    c = 0.0
                else:
                    depth_here = -Bottom[2, i, j]
                    depth_plus = -Bottom[2, i, j + 1]
                    depth_minus = -Bottom[2, i, j - 1]
                    # Calculate the first derivative
                    d_dy = (depth_plus - depth_minus) / (2.0 * dy)
                    # Calculate coefficients
                    a = depth_here * d_dy / (6.0 * dy) - (
                        self.Bcoef + 1.0 / 3.0
                    ) * depth_here * depth_here / (dy * dy)
                    b = 1.0 + 2.0 * (
                        self.Bcoef + 1.0 / 3.0
                    ) * depth_here * depth_here / (dy * dy)
                    c = -depth_here * d_dy / (6.0 * dy) - (
                        self.Bcoef + 1.0 / 3.0
                    ) * depth_here * depth_here / (dy * dy)
                # Store the coefficients in the texture field
                self.coefMaty[i, j] = ti.Vector([a, b, c, 0.0], self.precision)

    @ti.kernel
    def Pass1(self, step: ti.i32):  # noqa: N802, D102
        """
        Reconstruction step (Pass1):
          - Builds left/right (or N/E/S/W) interface values of eta, momentum, and 
            scalar concentration using a generalized minmod limiter.
          - Applies near-dry checks to skip processing cells that are effectively dry.
          - Computes velocity components and partial Froude-limiter logic.

        For 1D (ny=1), a simpler logic is used. For 2D, reconstruction is in both 
        x- and y-directions.

        Args:
            step (int): Counter to compute statistics.
        """
        zro = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)

        # Bresenham's line algorithm
        # bresenham_list = list(
        #     bresenham(
        #         self.force_sensor_begin[0],
        #         self.force_sensor_begin[1],
        #         self.force_sensor_end[0],
        #         self.force_sensor_end[1],
        #     )
        # )
        # print("Bresenham List: ", bresenham_list)
        hydrostatic_force = 0.0
        current_U_force = 0.0
        current_V_force = 0.0
        current_force = 0.0
        
        for i, j in self.State:
            # Compute the coordinates of the neighbors
            rightIdx = ti.min(i + 1, self.nx - 1)  # noqa: N806
            upIdx = ti.min(j + 1, self.ny - 1)  # noqa: N806
            leftIdx = ti.max(i - 1, 0)  # noqa: N806
            downIdx = ti.max(j - 1, 0)  # noqa: N806

            # Read in the state of the water at this pixel and its neighbors [eta,hu,hv,c]
            in_here = self.State[i, j]
            in_S = self.State[i, downIdx]  # noqa: N806
            in_N = self.State[i, upIdx]  # noqa: N806
            in_W = self.State[leftIdx, j]  # noqa: N806
            in_E = self.State[rightIdx, j]  # noqa: N806

            B_here = self.Bottom[2, i, j]  # noqa: N806
            B_south = self.Bottom[2, i, downIdx]  # noqa: N806
            B_north = self.Bottom[2, i, upIdx]  # noqa: N806
            B_west = self.Bottom[2, leftIdx, j]  # noqa: N806
            B_east = self.Bottom[2, rightIdx, j]  # noqa: N806

            # h = eta - B
            h_here = in_here[0] - B_here
            h_south = in_S[0] - B_south
            h_north = in_N[0] - B_north
            h_west = in_W[0] - B_west
            h_east = in_E[0] - B_east

            # Define h_near = eta_near - B_near
            # HNear_vec neighbours [hN,hE,hS,hW]
            self.Hnear[i, j] = [h_north, h_east, h_south, h_west]

            # To avoid unnecessary computations
            h_cut = self.delta
            if h_here <= h_cut:
                if (
                    h_north <= h_cut
                    and h_east <= h_cut
                    and h_south <= h_cut
                    and h_west <= h_cut
                ):
                    self.H[i, j] = zro
                    self.U[i, j] = zro
                    self.V[i, j] = zro
                    self.C[i, j] = zro
                    self.Auxiliary[i, j] = zro
                    continue

            ########################################################
            # Pass 1
            # Load bed elevation data for this cell's edges.
            # B neighbours [BN,BE,BS,BW]
            B = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)  # noqa: N806
            B[0] = self.Bottom[0, i, j]
            B[1] = self.Bottom[1, i, j]
            B[2] = self.Bottom[0, i, downIdx]
            B[3] = self.Bottom[1, leftIdx, j]

            dB_max = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)  # noqa: N806

            dB_west = ti.abs(B_here - B_west)  # noqa: N806
            dB_east = ti.abs(B_here - B_east)  # noqa: N806
            dB_south = ti.abs(B_here - B_south)  # noqa: N806
            dB_north = ti.abs(B_here - B_north)  # noqa: N806

            # Initialize variables for water height, momentum components, and standard deviation
            h = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            w = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            hu = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            hv = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            hc = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)

            # modify limiters based on whether near the inundation limit
            wetdry = ti.min(
                h_here, ti.min(h_south, ti.min(h_north, ti.min(h_west, h_east)))
            )
            rampcoef = ti.min(ti.max(0.0, wetdry / (0.02 * self.base_depth)), 1.0)
            # transition to full upwinding near the shoreline / inundation limit, start transition with a total water depth of base_depth/50
            TWO_THETAc = self.two_theta * rampcoef + 2.0 * (1.0 - rampcoef)  # noqa: N806

            if wetdry <= self.epsilon:
                dB_max = 0.5 * ti.Vector(  # noqa: N806
                    [dB_north, dB_east, dB_south, dB_west], self.precision
                )

            # Reconstruction eta
            wwy = Reconstruct(in_W[0], in_here[0], in_E[0], TWO_THETAc)
            wzx = Reconstruct(in_S[0], in_here[0], in_N[0], TWO_THETAc)
            w = ti.Vector([wzx.y, wwy.y, wzx.x, wwy.x])

            # Reconstruct h from (corrected) w
            h = w - B
            h = ti.max(h, ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision))

            # Reconstruction hu ~P
            huwy = Reconstruct(in_W[1], in_here[1], in_E[1], TWO_THETAc)
            huzx = Reconstruct(in_S[1], in_here[1], in_N[1], TWO_THETAc)
            hu = ti.Vector([huzx.y, huwy.y, huzx.x, huwy.x])

            # Reconstruction hv - Q
            hvwy = Reconstruct(in_W[2], in_here[2], in_E[2], TWO_THETAc)
            hvzx = Reconstruct(in_S[2], in_here[2], in_N[2], TWO_THETAc)
            hv = ti.Vector([hvzx.y, hvwy.y, hvzx.x, hvwy.x])

            # Reconstruction hc - scalar - sediment concentration or contaminent
            hcwy = Reconstruct(in_W[3], in_here[3], in_E[3], TWO_THETAc)
            hczx = Reconstruct(in_S[3], in_here[3], in_N[3], TWO_THETAc)
            hc = ti.Vector([hczx.y, hcwy.y, hczx.x, hcwy.x])

            output_u, output_v, output_c = CalcUV(
                h, hu, hv, hc, self.epsilon, dB_max
            )

            # Froude number limiter
            epsilon_c = ti.max(self.epsilon, dB_max)
            divide_by_h = 2.0 * h / (h * h + max(h * h, epsilon_c))

            Fr = ti.sqrt(output_u * output_u + output_v * output_v) / (  # noqa: N806
                ti.sqrt(9.81 / divide_by_h)
            )
            Frumax = ti.max(Fr.x, ti.max(Fr.y, ti.max(Fr.z, Fr.w)))  # noqa: N806
            dBdx = ti.abs(B_east - B_west) / self.double_dx  # noqa: N806
            dBdy = ti.abs(B_north - B_south) / self.double_dy  # noqa: N806
            dBds_max = ti.max(dBdx, dBdy)  # noqa: N806
            # max Fr allowed on slopes less than 45 degrees is 3;
            # for very steep slopes, artificially slow velocity - physics are just completely wrong here anyhow
            Fr_maxallowed = 3.0 / ti.max(1.0, dBds_max)  # noqa: N806
            if Frumax > Fr_maxallowed:
                Fr_red = Fr_maxallowed / Frumax  # noqa: N806
                output_u = output_u * Fr_red
                output_v = output_v * Fr_red

            # compute statistics of flow depth
            flow_depth = (h[0] + h[1] + h[2] + h[3]) / 4
            maxInundatedDepth = max(flow_depth, self.Auxiliary[i,j][0])
            minInundatedDepth = min(flow_depth, self.Auxiliary[i,j][1])
            meanFlow_depth = (self.Auxiliary[i,j][2]*step + flow_depth)/(step+1)
            
            maxVelocityU = max(  # noqa: N806
                (output_u[0] + output_u[1] + output_u[2] + output_u[3]) / 4,
                self.Auxiliary[i, j][1],
            )
            maxVelocityV = max(  # noqa: N806
                (output_v[0] + output_v[1] + output_v[2] + output_v[3]) / 4,
                self.Auxiliary[i, j][2],
            )
            maxVelocityC = max(  # noqa: N806
                (output_c[0] + output_c[1] + output_c[2] + output_c[3]) / 4,
                self.Auxiliary[i, j][3],
            )

            # Write H, U, V, C vector fields
            self.H[i, j] = h
            self.U[i, j] = output_u
            self.V[i, j] = output_v
            self.C[i, j] = output_c
            self.Auxiliary[i, j] = ti.Vector(
                [maxInundatedDepth, minInundatedDepth, meanFlow_depth, 0.0],
                self.precision,
            )
            
            if (j >= self.force_sensor_begin[1] and j <= self.force_sensor_end[1]) and (i >= self.force_sensor_begin[0] and i <= self.force_sensor_end[0]):
                # print("Calculating hydrostatic force at: ", i, j)
                # Calculate the hydrostatic force
                # Assuming a simple hydrostatic force calculation
                # hydrostatic_force = 1/2 density * g * w * h
                # where density is assumed to be 1 for simplicity
                # and h is the water height at (i, j)
                hsqr = float(self.H[i, j][0]) * float(self.H[i, j][0])
                hydrostatic_force += 0.5 * 1000.0 * float(self.g) * float(hsqr) * float(self.dy) * self.length_scale**3
            
                # Assume uniform current velocity vertically for simplicity
                # volume = dx * dy * Z_water
                # mass = density * volume
                # acceleration = velocity / dt  assuming boundary condition
                # force = acceleration * mass
                cell_volume = float(self.dx) * float(self.dy) * float(self.H[i, j][0])
                cell_mass = 1000.0 * cell_volume  # Assuming density of water is 1000 kg/m^3
                current_U_acceleration = float(self.U[i, j][0]) / float(self.dt)
                current_V_acceleration = float(self.V[i, j][0]) / float(self.dt)     
                current_U_force += current_U_acceleration * cell_mass * self.length_scale**3
                current_V_force += current_V_acceleration * cell_mass * self.length_scale**3
                # Dot product with normal of force sensor
                force_sensor_normal = [self.force_sensor_begin[1] - self.force_sensor_end[1], self.force_sensor_begin[0] - self.force_sensor_end[0]]
                force_sensor_normal_length = ti.sqrt(force_sensor_normal[0]**2 + force_sensor_normal[1]**2)
                force_sensor_normal[0] /= force_sensor_normal_length
                force_sensor_normal[1] /= force_sensor_normal_length
                current_force_dotted = -1.0 * (current_U_force * force_sensor_normal[0] + current_V_force * force_sensor_normal[1])
                current_force += ti.max(current_force_dotted, 0.0) 

                print("Hydrostatic force: ", hydrostatic_force)
                print("Current force: ", current_force)

        self.current_U_force[int(0)] = float(current_U_force)
        self.current_V_force[int(0)] = float(current_V_force)
        self.current_force[int(0)] = float(current_force)
        self.hydrostatic_forces[int(0)] = float(hydrostatic_force)
        self.hydrostatic_forces[int(step)] = float(hydrostatic_force)


    def overwrite_force(self):
        # open forces.evt to overwrite
        print("Overwriting forces.evt")
        with open("forces.evt", "w") as f:
            max_steps = 10000 # self.hydrostatic_forces.shape[0]
            for i in range(max_steps):
                if (i < max_steps - 1):
                    f.write(f"{0.0} ")
                else:
                    f.write(f"{0.0}\n")
                

    def write_force(self):
        # open forces.evt to append        
        with open("forces.evt", "a") as f:
            f.write(f"{self.hydrostatic_forces[int(0)] + self.current_force[int(0)]} ")

    def update_step(self):
        self.step += 1


    @ti.kernel
    def Pass1_SedTrans(self):  # noqa: N802, D102
        # Pass1 - edge value construction of hc
        # using Generalized minmod limiter
        for i, j in self.State_Sed:
            # Compute the coordinates of the neighbors
            rightIdx = ti.min(i + 1, self.nx - 1)  # noqa: N806
            upIdx = ti.min(j + 1, self.ny - 1)  # noqa: N806
            leftIdx = ti.max(i - 1, 0)  # noqa: N806
            downIdx = ti.max(j - 1, 0)  # noqa: N806

            in_here = self.State_Sed[i, j].x
            in_S = self.State_Sed[i, downIdx].x  # noqa: N806
            in_N = self.State_Sed[i, upIdx].x  # noqa: N806
            in_W = self.State_Sed[leftIdx, j].x  # noqa: N806
            in_E = self.State_Sed[rightIdx, j].x  # noqa: N806

            B_here = self.Bottom[2, i, j]  # noqa: N806
            B_south = self.Bottom[2, i, downIdx]  # noqa: N806
            B_north = self.Bottom[2, i, upIdx]  # noqa: N806
            B_west = self.Bottom[2, leftIdx, j]  # noqa: N806
            B_east = self.Bottom[2, rightIdx, j]  # noqa: N806

            dB_west = ti.abs(B_here - B_west)  # noqa: N806
            dB_east = ti.abs(B_here - B_east)  # noqa: N806
            dB_south = ti.abs(B_here - B_south)  # noqa: N806
            dB_north = ti.abs(B_here - B_north)  # noqa: N806

            dB_max = 0.5 * ti.Vector(  # noqa: N806
                [dB_north, dB_east, dB_south, dB_west], self.precision
            )

            h_here = in_here - B_here
            h_south = in_S - B_south
            h_north = in_N - B_north
            h_west = in_W - B_west
            h_east = in_E - B_east

            # Pass 1
            #  Initialize local variables (thread)
            hc = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)

            # modify limiters based on whether near the inundation limit
            wetdry = ti.min(
                h_here, ti.min(h_south, ti.min(h_north, ti.min(h_west, h_east)))
            )
            rampcoef = ti.min(ti.max(0.0, wetdry / (0.02 * self.base_depth)), 1.0)

            # transition to full upwinding near the shoreline / inundation limit, start transition with a total water depth of base_depth/50
            TWO_THETAc = self.two_theta * rampcoef + 2.0 * (1.0 - rampcoef)  # noqa: N806

            # Reconstruction class1
            hcwy = Reconstruct(in_W, in_here, in_E, TWO_THETAc)
            hczx = Reconstruct(in_S, in_here, in_N, TWO_THETAc)

            hc = ti.Vector([hczx.y, hcwy.y, hczx.x, hcwy.x], self.precision)

            h = self.H[i, j]

            epsilon_c = ti.max(self.epsilon, dB_max)
            divide_by_h = 2.0 * h / (h * h + max(h * h, epsilon_c))

            c_sed = divide_by_h * hc

            # Write Sediment concentration by edge
            self.Sed_C[i, j] = c_sed

    @ti.kernel
    def Pass2(self):  # noqa: N802, D102
        # PASS 2 - Calculus of fluxes
        for i, j in self.Hnear:
            rightIdx = ti.min(i + 1, self.nx - 1)  # noqa: N806
            upIdx = ti.min(j + 1, self.ny - 1)  # noqa: N806
            leftIdx = ti.max(i - 1, 0)  # noqa: N806
            downIdx = ti.max(j - 1, 0)  # noqa: N806

            h_vec = self.Hnear[i, j]

            h_here = ti.Vector([0.0, 0.0], self.precision)
            h_here[0] = self.H[i, j][0]
            h_here[1] = self.H[i, j][1]

            hW_east = self.H[rightIdx, j][3]  # noqa: N806
            hS_north = self.H[i, upIdx][2]  # noqa: N806

            u_here = ti.Vector([0.0, 0.0], self.precision)
            u_here[0] = self.U[i, j][0]
            u_here[1] = self.U[i, j][1]

            uW_east = self.U[rightIdx, j][3]  # noqa: N806
            uS_north = self.U[i, upIdx][2]  # noqa: N806

            v_here = ti.Vector([0.0, 0.0], self.precision)
            v_here[0] = self.V[i, j][0]
            v_here[1] = self.V[i, j][1]

            vW_east = self.V[rightIdx, j][3]  # noqa: N806
            vS_north = self.V[i, upIdx][2]  # noqa: N806

            # Compute wave speeds
            cNE = ti.sqrt(self.g * h_here)  # noqa: N806
            cW = ti.sqrt(self.g * hW_east)  # cW evaluated at (j+1, k)  # noqa: N806
            cS = ti.sqrt(self.g * hS_north)  # cS evaluated at (j, k+1)  # noqa: N806

            # Compute propagation speeds
            aplus = ti.max(ti.max(u_here[1] + cNE[1], uW_east + cW), 0.0)
            aminus = ti.min(ti.min(u_here[1] - cNE[1], uW_east - cW), 0.0)
            bplus = ti.max(ti.max(v_here[0] + cNE[0], vS_north + cS), 0.0)
            bminus = ti.min(ti.min(v_here[0] - cNE[0], vS_north - cS), 0.0)

            B_here = self.Bottom[2, i, j]  # noqa: N806
            dB = ti.max(  # noqa: N806, F841
                self.Bottom[2, i, downIdx] - B_here,
                ti.max(
                    self.Bottom[2, i, upIdx] - B_here,
                    ti.max(
                        self.Bottom[2, leftIdx, j] - B_here,
                        self.Bottom[2, rightIdx, j] - B_here,
                    ),
                ),
            )

            # near_dry = self.Bottom[3,pi, pj] # Check the value of this field not used in here

            c_here = ti.Vector([0.0, 0.0], self.precision)
            c_here[0] = self.C[i, j][0]
            c_here[1] = self.C[i, j][1]

            cW_east = self.C[rightIdx, j][3]  # noqa: N806
            cS_north = self.C[i, upIdx][2]  # noqa: N806

            phix = 0.5
            phiy = 0.5

            minH = ti.min(h_vec.w, ti.min(h_vec.z, ti.min(h_vec.y, h_vec.x)))  # noqa: N806
            mass_diff_x = hW_east - h_here.y
            mass_diff_y = hS_north - h_here.x

            P_diff_x = hW_east * uW_east - h_here.y * u_here.y  # noqa: N806
            P_diff_y = hS_north * uS_north - h_here.x * u_here.x  # noqa: N806

            Q_diff_x = hW_east * vW_east - h_here.y * v_here.y  # noqa: N806
            Q_diff_y = hS_north * vS_north - h_here.x * v_here.x  # noqa: N806

            if minH <= self.delta:
                mass_diff_x = 0.0
                mass_diff_y = 0.0
                phix = 1.0
                phiy = 1.0

            xflux = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
            yflux = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)

            xflux[0] = NumericalFlux(
                aplus, aminus, hW_east * uW_east, h_here.y * u_here.y, mass_diff_x
            )
            xflux[1] = NumericalFlux(
                aplus,
                aminus,
                hW_east * uW_east * uW_east,
                h_here.y * u_here.y * u_here.y,
                P_diff_x,
            )
            xflux[2] = NumericalFlux(
                aplus,
                aminus,
                hW_east * uW_east * vW_east,
                h_here.y * u_here.y * v_here.y,
                Q_diff_x,
            )
            xflux[3] = NumericalFlux(
                aplus,
                aminus,
                hW_east * uW_east * cW_east,
                h_here.y * u_here.y * c_here.y,
                phix * (hW_east * cW_east - h_here.y * c_here.y),
            )

            yflux[0] = NumericalFlux(
                bplus, bminus, hS_north * vS_north, h_here.x * v_here.x, mass_diff_y
            )
            yflux[1] = NumericalFlux(
                bplus,
                bminus,
                hS_north * uS_north * vS_north,
                h_here.x * u_here.x * v_here.x,
                P_diff_y,
            )
            yflux[2] = NumericalFlux(
                bplus,
                bminus,
                hS_north * vS_north * vS_north,
                h_here.x * v_here.x * v_here.x,
                Q_diff_y,
            )
            yflux[3] = NumericalFlux(
                bplus,
                bminus,
                hS_north * cS_north * vS_north,
                h_here.x * c_here.x * v_here.x,
                phiy * (hS_north * cS_north - h_here.x * c_here.x),
            )

            # Write Fluxes fluid
            self.XFlux[i, j] = xflux
            self.YFlux[i, j] = yflux

            if self.useSedTransModel == True:  # noqa: E712
                c1_here = ti.Vector([0.0, 0.0], self.precision)
                c1_here[0] = self.Sed_C[i, j][0]
                c1_here[1] = self.Sed_C[i, j][1]
                c1W_east = self.Sed_C[rightIdx, j][3]  # noqa: N806
                c1S_north = self.Sed_C[i, upIdx][2]  # noqa: N806

                xflux_Sed = NumericalFlux(  # noqa: N806
                    aplus,
                    aminus,
                    hW_east * uW_east * c1W_east,
                    h_here[1] * u_here[1] * c1_here[1],
                    phix * (hW_east * c1W_east - h_here[1] * c1_here[1]),
                )
                yflux_Sed = NumericalFlux(  # noqa: N806
                    bplus,
                    bminus,
                    hS_north * c1S_north * vS_north,
                    h_here[0] * c1_here[0] * v_here[0],
                    phiy * (hS_north * c1S_north - h_here[0] * c1_here[0]),
                )

                # Write Fluxes sediment by classes
                self.XFlux_Sed[i, j].x = xflux_Sed
                self.YFlux_Sed[i, j].x = yflux_Sed

    @ti.kernel
    def Pass3(self, pred_or_corrector: ti.i32):  # noqa: C901, N802, D102
        # PASS 3 - Do timestep and calculate new w_bar, hu_bar, hv_bar.
        zro = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
        for i, j in self.NewState:
            if i >= (self.nx - 2) or j >= (self.ny - 2) or i <= 1 or j <= 1:
                self.NewState[i, j] = zro
                self.dU_by_dt[i, j] = zro
                self.predictedF_G_star[i, j] = zro
                self.current_stateUVstar[i, j] = zro
                continue

            # Load values from txBottom using index
            B_here = self.Bottom[2, i, j]  # noqa: N806

            in_state_here = self.State[
                i, j
            ]  # w, hu and hv, c (cell avgs, evaluated here)
            in_state_here_UV = self.stateUVstar[i, j]  # noqa: N806

            B_south = self.Bottom[2, i, j - 1]  # noqa: N806
            B_north = self.Bottom[2, i, j + 1]  # noqa: N806
            B_west = self.Bottom[2, i - 1, j]  # noqa: N806
            B_east = self.Bottom[2, i + 1, j]  # noqa: N806

            eta_here = in_state_here[0]
            eta_west = self.State[i - 1, j][0]
            eta_east = self.State[i + 1, j][0]
            eta_south = self.State[i, j - 1][0]
            eta_north = self.State[i, j + 1][0]

            h_here = in_state_here[0] - B_here
            h_west = eta_west - B_west
            h_east = eta_east - B_east
            h_north = eta_north - B_north
            h_south = eta_south - B_south

            h_cut = self.delta
            if h_here <= h_cut:
                if (
                    h_north <= h_cut
                    and h_east <= h_cut
                    and h_south <= h_cut
                    and h_west <= h_cut
                ):
                    self.NewState[i, j] = zro
                    self.dU_by_dt[i, j] = zro
                    self.predictedF_G_star[i, j] = zro
                    self.current_stateUVstar[i, j] = zro
                    continue

            h_min = zro
            h_min[0] = ti.min(h_here, h_north)
            h_min[1] = ti.min(h_here, h_east)
            h_min[2] = ti.min(h_here, h_south)
            h_min[3] = ti.min(h_here, h_west)

            detadx = 0.5 * (eta_east - eta_west) * self.one_over_dx
            detady = 0.5 * (eta_north - eta_south) * self.one_over_dy

            # Load values from txXFlux and txYFlux using idx
            xflux_here = self.XFlux[i, j]  # at i+1/2
            xflux_west = self.XFlux[i - 1, j]  # at i-1/2
            yflux_here = self.YFlux[i, j]  # at j+1/2
            yflux_south = self.YFlux[i, j - 1]  # at j-1/2

            friction_here = ti.max(self.friction, self.BottomFriction[i, j][0])
            friction_ = FrictionCalc(
                in_state_here[1],
                in_state_here[2],
                h_here,
                self.base_depth,
                self.delta,
                self.isManning,
                self.g,
                friction_here,
            )

            # Pressure stencil calculations
            P_left = self.ShipPressure[i - 1, j].x  # noqa: N806
            P_right = self.ShipPressure[i + 1, j].x  # noqa: N806
            P_down = self.ShipPressure[i, j - 1].x  # noqa: N806
            P_up = self.ShipPressure[i, j + 1].x  # noqa: N806

            press_x = -0.5 * h_here * self.g_over_dx * (P_right - P_left)
            press_y = -0.5 * h_here * self.g_over_dy * (P_up - P_down)

            # Calculate scalar transport additions
            C_state_here = self.State[i, j][3]  # noqa: N806

            C_state_left = self.State[i - 1, j][3]  # noqa: N806
            C_state_right = self.State[i + 1, j][3]  # noqa: N806
            C_state_up = self.State[i, j + 1][3]  # noqa: N806
            C_state_down = self.State[i, j - 1][3]  # noqa: N806
            C_state_up_left = self.State[i - 1, j + 1][3]  # noqa: N806
            C_state_up_right = self.State[i + 1, j + 1][3]  # noqa: N806
            C_state_down_left = self.State[i - 1, j - 1][3]  # noqa: N806
            C_state_down_right = self.State[i + 1, j - 1][3]  # noqa: N806

            Dxx = self.whiteWaterDispersion  # noqa: N806
            Dxy = self.whiteWaterDispersion  # noqa: N806
            Dyy = self.whiteWaterDispersion  # noqa: N806

            hc_by_dx_dx = (
                Dxx
                * self.one_over_d2x
                * (C_state_right - 2.0 * in_state_here[3] + C_state_left)
            )
            hc_by_dy_dy = (
                Dyy
                * self.one_over_d2y
                * (C_state_up - 2.0 * in_state_here[3] + C_state_down)
            )
            hc_by_dx_dy = (
                0.25
                * Dxy
                * self.one_over_dxdy
                * (
                    C_state_up_right
                    - C_state_up_left
                    - C_state_down_right
                    + C_state_down_left
                )
            )

            c_dissipation = -self.whiteWaterDecayRate * C_state_here

            breaking_B = 0.0  # noqa: N806
            if self.useBreakingModel == True:  # noqa: E712
                breaking_B = self.Breaking[  # noqa: N806
                    i, j
                ].z  # breaking front parameter, non-breaking [0 - 1] breaking

            # fix slope near shoreline
            if h_min.x <= h_cut and h_min.z <= h_cut:
                detady = 0.0
            elif h_min.x <= h_cut:
                detady = 1.0 * (eta_here - eta_south) * self.one_over_dy
            elif h_min.z <= h_cut:
                detady = 1.0 * (eta_north - eta_here) * self.one_over_dy

            if h_min.y <= h_cut and h_min.w <= h_cut:
                detadx = 0.0
            elif h_min.y <= h_cut:
                detadx = 1.0 * (eta_here - eta_west) * self.one_over_dx
            elif h_min.w <= h_cut:
                detadx = 1.0 * (eta_east - eta_here) * self.one_over_dx

            overflow_dry = 0.0
            if B_here > 0.0:
                overflow_dry = (
                    -self.infiltrationRate
                )  # hydraulic conductivity of coarse, unsaturated sand

            source_term = ti.Vector(
                [
                    overflow_dry,
                    -self.g * h_here * detadx
                    - in_state_here.y * friction_
                    + press_x,
                    -self.g * h_here * detady
                    - in_state_here.z * friction_
                    + press_y,
                    hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + c_dissipation,
                ],
                self.precision,
            )

            d_by_dt = (
                (xflux_west - xflux_here) * self.one_over_dx
                + (yflux_south - yflux_here) * self.one_over_dy
                + source_term
            )

            # previous derivatives
            oldies = self.oldGradients[i, j]
            oldOldies = self.oldOldGradients[i, j]  # noqa: N806

            newState = zro  # w , hu, hv ,hc  # noqa: N806

            if self.timeScheme == 0:  # if timeScheme is Euler do:
                newState = in_state_here_UV + self.dt * d_by_dt  # noqa: N806

            elif pred_or_corrector == 1:
                # if time scheme is predictor
                newState = in_state_here_UV + self.dt / 12.0 * (  # noqa: N806
                    23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies
                )

            elif pred_or_corrector == 2:  # noqa: PLR2004
                # if time scheme is corrector
                predicted = self.predictedGradients[i, j]
                newState = in_state_here_UV + self.dt / 24.0 * (  # noqa: N806
                    9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + oldOldies
                )

            if self.showBreaking == True:  # noqa: E712
                # add breaking source
                newState.a = ti.max(
                    newState.w, breaking_B
                )  # use the B value from Kennedy et al as a foam intensity
            elif self.showBreaking == 2:  # noqa: PLR2004
                contaminent_source = self.ContSource[i, j].x
                newState.w = ti.min(1.0, newState.w + contaminent_source)

            F_G_vec = ti.Vector([0.0, 0.0, 0.0, 1.0], self.precision)  # noqa: N806

            self.NewState[i, j] = newState
            self.dU_by_dt[i, j] = d_by_dt
            self.predictedF_G_star[i, j] = F_G_vec
            self.current_stateUVstar[i, j] = newState

    @ti.kernel
    def Pass3_SedTrans(self, pred_or_corrector: ti.i32):  # noqa: N802, D102
        for i, j in self.State_Sed:
            if i >= (self.nx - 2) or j >= (self.ny - 2) or i <= 1 or j <= 1:
                self.NewState_Sed[i, j].x = 0.0
                self.dU_by_dt_Sed[i, j].x = 0.0
                continue
            # Load values from txXFlux and txYFlux using idx
            xflux_here = self.XFlux_Sed[i, j].x  # at i+1/2
            xflux_west = self.XFlux_Sed[i - 1, j].x  # at i-1/2
            yflux_here = self.YFlux_Sed[i, j].x  # at j+1/2
            yflux_south = self.YFlux_Sed[i, j - 1].x  # at j-1/2

            # Calculate scalar transport additions
            C_state_here = self.State_Sed[i, j].x  # noqa: N806
            C_state_left = self.State_Sed[i - 1, j].x  # noqa: N806
            C_state_right = self.State_Sed[i + 1, j].x  # noqa: N806
            C_state_up = self.State_Sed[i, j + 1].x  # noqa: N806
            C_state_down = self.State_Sed[i, j - 1].x  # noqa: N806
            C_state_up_left = self.State_Sed[i - 1, j + 1].x  # noqa: N806
            C_state_up_right = self.State_Sed[i + 1, j + 1].x  # noqa: N806
            C_state_down_left = self.State_Sed[i - 1, j - 1].x  # noqa: N806
            C_state_down_right = self.State_Sed[i + 1, j - 1].x  # noqa: N806

            Dxx = 1.0  # noqa: N806
            Dxy = 1.0  # noqa: N806
            Dyy = 1.0  # noqa: N806

            hc_by_dx_dx = (
                Dxx
                * self.one_over_d2x
                * (C_state_right - 2.0 * C_state_here + C_state_left)
            )
            hc_by_dy_dy = (
                Dyy
                * self.one_over_d2y
                * (C_state_up - 2.0 * C_state_here + C_state_down)
            )
            hc_by_dx_dy = (
                Dxy
                * self.one_over_dxdy
                * (
                    C_state_up_right
                    - C_state_up_left
                    - C_state_down_right
                    + C_state_down_left
                )
                / 4.0
            )

            B = self.Bottom[2, i, j]  # noqa: N806
            in_state_here = self.State[i, j]
            eta = in_state_here[0]
            hu = in_state_here[1]
            hv = in_state_here[2]
            h = eta - B

            divide_by_h = (
                2.0 * h / ti.sqrt(h * h + ti.max(h * h, self.epsilon))
            )

            f = self.friction / 2.0
            if self.isManning:
                f = (
                    9.81
                    * ti.pow(self.friction, 2.0)
                    * ti.pow(ti.abs(divide_by_h), 1.0 / 3.0)
                )

            u = hu * divide_by_h
            v = hv * divide_by_h

            local_speed = ti.sqrt(u**2 + v**2)

            shear_velocity = ti.sqrt(f) * local_speed

            shields = shear_velocity * shear_velocity * self.sediment.Shields
            erosion = 0.0

            if shields >= self.sediment.CriticalShields:
                erosion = (
                    self.sediment.C_erosion
                    * (shields - self.sediment.CriticalShields)
                    * local_speed
                    * divide_by_h
                )

            # working only with one class
            Cmin = ti.max(1.0e-6, C_state_here)  # noqa: N806

            deposition = (
                ti.min(2.0, (1.0 - self.sediment.p) / Cmin)
                * C_state_here
                * self.sediment.C_settling
            )

            source_term = (
                hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + erosion - deposition
            )

            d_by_dt = (
                (xflux_west - xflux_here) * self.one_over_dx
                + (yflux_south - yflux_here) * self.one_over_dy
                + source_term
            )

            # previous derivatives
            oldies = self.oldGradients_Sed[i, j].x
            oldOldies = self.oldOldGradients_Sed[i, j].x  # noqa: N806

            Out = 0.0  # noqa: N806

            if self.timeScheme == 0:  # if timeScheme is Euler do:
                Out = C_state_here + self.dt * d_by_dt  # noqa: N806

            elif pred_or_corrector == 1:  # if time scheme is predictor
                Out = C_state_here + self.dt / 12.0 * (  # noqa: N806
                    23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies
                )

            elif (
                pred_or_corrector == 2  # noqa: PLR2004
            ):  # if time scheme is corrector
                predicted = self.predictedGradients_Sed[i, j].x
                # corrector step
                Out = C_state_here + self.dt / 24.0 * (  # noqa: N806
                    9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + 1.0 * oldOldies
                )

            self.NewState_Sed[i, j].x = Out
            self.dU_by_dt_Sed[i, j].x = d_by_dt
            self.erosion_Sed[i, j].x = erosion
            self.deposition_Sed[i, j].x = deposition

    @ti.kernel
    def Pass3Bous(self, pred_or_corrector: ti.i32):  # noqa: C901, N802, D102, PLR0915
        Bottom = ti.static(self.Bottom)  # noqa: N806
        zro = ti.Vector([0.0, 0.0, 0.0, 0.0], self.precision)
        # PASS 3 - Calculus of fluxes
        for i, j in self.State:
            if i >= (self.nx - 2) or j >= (self.ny - 2) or i <= 1 or j <= 1:
                self.NewState[i, j] = zro
                self.dU_by_dt[i, j] = zro
                self.predictedF_G_star[i, j] = zro
                self.current_stateUVstar[i, j] = zro
                continue

            # Load values from txBottom using index
            B_here = Bottom[2, i, j]  # noqa: N806

            # Load values from txState using index
            in_state_here = self.State[
                i, j
            ]  # w, hu and hv (cell avgs, evaluated here)
            in_state_here_UV = self.stateUVstar[i, j]  # noqa: N806

            h_here = in_state_here.x - B_here  # h = w - B

            # Calculate local h,w
            B_south = Bottom[2, i, j - 1]  # noqa: N806
            B_north = Bottom[2, i, j + 1]  # noqa: N806
            B_west = Bottom[2, i - 1, j]  # noqa: N806
            B_east = Bottom[2, i + 1, j]  # noqa: N806

            eta_here = in_state_here.x

            eta_west = self.State[i - 1, j].x
            eta_east = self.State[i + 1, j].x
            eta_south = self.State[i, j - 1].x
            eta_north = self.State[i, j + 1].x

            h_west = eta_west - B_west
            h_east = eta_east - B_east
            h_north = eta_north - B_north
            h_south = eta_south - B_south

            # if dry and surrounded by dry, then stay dry - no need to calc
            h_cut = self.delta
            if h_here <= h_cut:
                if (
                    h_north <= h_cut
                    and h_east <= h_cut
                    and h_south <= h_cut
                    and h_west <= h_cut
                ):
                    self.NewState[i, j] = zro
                    self.dU_by_dt[i, j] = zro
                    self.predictedF_G_star[i, j] = zro
                    self.current_stateUVstar[i, j] = zro
                    continue

            h_min = zro
            h_min.x = ti.min(h_here, h_north)
            h_min.y = ti.min(h_here, h_east)
            h_min.z = ti.min(h_here, h_south)
            h_min.w = ti.min(h_here, h_west)

            # Load values from txXFlux and txYFlux using idx
            xflux_here = self.XFlux[i, j]  # at i+1/2
            xflux_west = self.XFlux[i - 1, j]  # at i-1/2
            yflux_here = self.YFlux[i, j]  # at j+1/2
            yflux_south = self.YFlux[i, j - 1]  # at j-1/2

            detadx = 0.5 * (eta_east - eta_west) * self.one_over_dx
            detady = 0.5 * (eta_north - eta_south) * self.one_over_dy

            F_star = 0.0  # noqa: N806
            G_star = 0.0  # noqa: N806
            Psi1x = 0.0  # noqa: N806
            Psi2x = 0.0  # noqa: N806
            Psi1y = 0.0  # noqa: N806
            Psi2y = 0.0  # noqa: N806

            d_here = -B_here
            near_dry = self.Bottom[3, i, j]

            # OPTIMIZ: only proceed if not near an initially dry cell
            if near_dry > 0.0:
                d2_here = d_here * d_here
                d3_here = d2_here * d_here

                in_state_left = self.State[i - 1, j].xyz
                in_state_right = self.State[i + 1, j].xyz
                in_state_up = self.State[i, j + 1].xyz
                in_state_down = self.State[i, j - 1].xyz
                in_state_up_left = self.State[i - 1, j + 1].xyz
                in_state_up_right = self.State[i + 1, j + 1].xyz
                in_state_down_left = self.State[i - 1, j - 1].xyz
                in_state_down_right = self.State[i + 1, j - 1].xyz

                F_G_star_oldOldies = self.F_G_star_oldOldGradients[i, j].xyz  # noqa: N806

                # Calculate "d" stencil
                d_left = -B_west
                d_right = -B_east
                d_down = -B_south
                d_up = -B_north

                d_left_left = ti.max(0.0, -Bottom[2, i - 2, j])
                d_right_right = ti.max(0.0, -Bottom[2, i + 2, j])
                d_down_down = ti.max(0.0, -Bottom[2, i, j - 2])
                d_up_up = ti.max(0.0, -Bottom[2, i, j + 2])

                # Calculate "eta" stencil
                eta_here = in_state_here.x
                eta_left = in_state_left.x
                eta_right = in_state_right.x
                eta_down = in_state_down.x
                eta_up = in_state_up.x

                eta_left_left = self.State[i - 2, j].x
                eta_right_right = self.State[i + 2, j].x
                eta_down_down = self.State[i, j - 2].x
                eta_up_up = self.State[i, j + 2].x

                eta_up_left = in_state_up_left.x
                eta_up_right = in_state_up_right.x
                eta_down_left = in_state_down_left.x
                eta_down_right = in_state_down_right.x

                # replace with 4th order when dispersion is included
                # detadx = 1.0 / 12.0 * (eta_left_left - 8.0 * eta_left + 8.0 * eta_right + eta_right_right) * self.one_over_dx
                # detady = 1.0 / 12.0 * (eta_down_down - 8.0 * eta_down + 8.0 * eta_up + eta_up_up) * self.one_over_dy
                detadx = (
                    (
                        -eta_right_right
                        + 8.0 * eta_right
                        - 8.0 * eta_left
                        + eta_left_left
                    )
                    * self.one_over_dx
                    / 12.0
                )
                detady = (
                    (-eta_up_up + 8.0 * eta_up - 8.0 * eta_down + eta_down_down)
                    * self.one_over_dy
                    / 12.0
                )

                u_up = in_state_up.y
                u_down = in_state_down.y
                u_right = in_state_right.y
                u_left = in_state_left.y
                u_up_right = in_state_up_right.y
                u_down_right = in_state_down_right.y
                u_up_left = in_state_up_left.y
                u_down_left = in_state_down_left.y

                v_up = in_state_up.z
                v_down = in_state_down.z
                v_right = in_state_right.z
                v_left = in_state_left.z
                v_up_right = in_state_up_right.z
                v_down_right = in_state_down_right.z
                v_up_left = in_state_up_left.z
                v_down_left = in_state_down_left.z

                dd_by_dx = (
                    (-d_right_right + 8.0 * d_right - 8.0 * d_left + d_left_left)
                    * self.one_over_dx
                    / 12.0
                )
                dd_by_dy = (
                    (-d_up_up + 8.0 * d_up - 8.0 * d_down + d_down_down)
                    * self.one_over_dy
                    / 12.0
                )
                eta_by_dx_dy = (
                    0.25
                    * self.one_over_dx
                    * self.one_over_dy
                    * (eta_up_right - eta_down_right - eta_up_left + eta_down_left)
                )
                eta_by_dx_dx = self.one_over_d2x * (
                    eta_right - 2.0 * eta_here + eta_left
                )
                eta_by_dy_dy = self.one_over_d2y * (
                    eta_up - 2.0 * eta_here + eta_down
                )

                F_star = (1.0 / 6.0) * d_here * (  # noqa: N806
                    dd_by_dx * (0.5 * self.one_over_dy) * (v_up - v_down)
                    + dd_by_dy * (0.5 * self.one_over_dx) * (v_right - v_left)
                ) + (self.Bcoef + 1.0 / 3.0) * d2_here * (
                    self.one_over_dxdy * 0.25
                ) * (v_up_right - v_down_right - v_up_left + v_down_left)

                G_star = (1.0 / 6.0) * d_here * (  # noqa: N806
                    dd_by_dx * (0.5 * self.one_over_dy) * (u_up - u_down)
                    + dd_by_dy * (0.5 * self.one_over_dx) * (u_right - u_left)
                ) + (self.Bcoef + 1.0 / 3.0) * d2_here * (
                    self.one_over_dxdy * 0.25
                ) * (u_up_right - u_down_right - u_up_left + u_down_left)

                Psi1x = (  # noqa: N806
                    self.Bcoef_g
                    * d3_here
                    * (
                        (
                            eta_right_right
                            - 2.0 * eta_right
                            + 2.0 * eta_left
                            - eta_left_left
                        )
                        * (0.5 * self.one_over_d3x)
                        + (
                            eta_up_right
                            - eta_up_left
                            - 2.0 * eta_right
                            + 2.0 * eta_left
                            + eta_down_right
                            - eta_down_left
                        )
                        * (0.5 * self.one_over_dx * self.one_over_d2y)
                    )
                )
                Psi2x = (  # noqa: N806
                    self.Bcoef_g
                    * d2_here
                    * (
                        dd_by_dx * (2.0 * eta_by_dx_dx + eta_by_dy_dy)
                        + dd_by_dy * eta_by_dx_dy
                    )
                    + (F_star - F_G_star_oldOldies.y) / self.dt * 0.5
                )
                Psi1y = (  # noqa: N806
                    self.Bcoef_g
                    * d3_here
                    * (
                        (eta_up_up - 2.0 * eta_up + 2.0 * eta_down - eta_down_down)
                        * (0.5 * self.one_over_d3y)
                        + (
                            eta_up_right
                            + eta_up_left
                            - 2.0 * eta_up
                            + 2.0 * eta_down
                            - eta_down_right
                            - eta_down_left
                        )
                        * (0.5 * self.one_over_dx * self.one_over_d2x)
                    )
                )
                Psi2y = (  # noqa: N806
                    self.Bcoef_g
                    * d2_here
                    * (
                        dd_by_dy * (2.0 * eta_by_dy_dy + eta_by_dx_dx)
                        + dd_by_dx * eta_by_dx_dy
                    )
                    + (G_star - F_G_star_oldOldies.z) / self.dt * 0.5
                )

            friction_here = ti.max(self.friction, self.BottomFriction[i, j][0])
            friction_ = FrictionCalc(
                in_state_here[1],
                in_state_here[2],
                h_here,
                self.base_depth,
                self.delta,
                self.isManning,
                self.g,
                friction_here,
            )

            # Pressure stencil calculations
            P_left = self.ShipPressure[i - 1, j].x  # noqa: N806
            P_right = self.ShipPressure[i + 1, j].x  # noqa: N806
            P_down = self.ShipPressure[i, j - 1].x  # noqa: N806
            P_up = self.ShipPressure[i, j + 1].x  # noqa: N806

            press_x = -0.5 * h_here * self.g_over_dx * (P_right - P_left)
            press_y = -0.5 * h_here * self.g_over_dy * (P_up - P_down)

            # Calculate scalar transport additions
            C_state_here = self.State[i, j].w  # noqa: N806

            C_state_left = self.State[i - 1, j].w  # noqa: N806
            C_state_right = self.State[i + 1, j].w  # noqa: N806
            C_state_up = self.State[i, j + 1].w  # noqa: N806
            C_state_down = self.State[i, j - 1].w  # noqa: N806
            C_state_up_left = self.State[i - 1, j + 1].w  # noqa: N806
            C_state_up_right = self.State[i + 1, j + 1].w  # noqa: N806
            C_state_down_left = self.State[i - 1, j - 1].w  # noqa: N806
            C_state_down_right = self.State[i + 1, j - 1].w  # noqa: N806

            Dxx = self.whiteWaterDispersion  # noqa: N806
            Dxy = self.whiteWaterDispersion  # noqa: N806
            Dyy = self.whiteWaterDispersion  # noqa: N806

            hc_by_dx_dx = (
                Dxx
                * self.one_over_d2x
                * (C_state_right - 2.0 * in_state_here.a + C_state_left)
            )
            hc_by_dy_dy = (
                Dyy
                * self.one_over_d2y
                * (C_state_up - 2.0 * in_state_here.a + C_state_down)
            )
            hc_by_dx_dy = (
                0.25
                * Dxy
                * self.one_over_dxdy
                * (
                    C_state_up_right
                    - C_state_up_left
                    - C_state_down_right
                    + C_state_down_left
                )
            )

            c_dissipation = -self.whiteWaterDecayRate * C_state_here

            # calculate breaking dissipation
            breaking_x = 0.0
            breaking_y = 0.0
            breaking_B = 0.0  # noqa: N806

            if self.useBreakingModel:
                breaking_B = self.Breaking[  # noqa: N806
                    i, j
                ].z  # breaking front parameter, non-breaking [0 - 1] breaking
                nu_flux_here = self.DissipationFlux[i, j]  # noqa: F841
                nu_flux_right = self.DissipationFlux[i + 1, j]
                nu_flux_left = self.DissipationFlux[i - 1, j]
                nu_flux_up = self.DissipationFlux[i, j + 1]
                nu_flux_down = self.DissipationFlux[i, j - 1]

                dPdxx = 0.5 * (nu_flux_right.x - nu_flux_left.x) * self.one_over_dx  # noqa: N806
                dPdyx = 0.5 * (nu_flux_right.y - nu_flux_left.y) * self.one_over_dx  # noqa: N806
                dPdyy = 0.5 * (nu_flux_up.y - nu_flux_down.y) * self.one_over_dy  # noqa: N806

                dQdxx = 0.5 * (nu_flux_right.z - nu_flux_left.z) * self.one_over_dx  # noqa: N806
                dQdxy = 0.5 * (nu_flux_up.z - nu_flux_down.z) * self.one_over_dy  # noqa: N806
                dQdyy = 0.5 * (nu_flux_up.w - nu_flux_down.w) * self.one_over_dy  # noqa: N806

                if near_dry > 0.0:
                    breaking_x = dPdxx + 0.5 * dPdyy + 0.5 * dQdxy
                    breaking_y = dQdyy + 0.5 * dPdyx + 0.5 * dQdxx

            # fix slope near shoreline
            if h_min.x <= h_cut and h_min.z <= h_cut:
                detady = 0.0
            elif h_min.x <= h_cut:
                detady = 1.0 * (eta_here - eta_south) * self.one_over_dy
            elif h_min.z <= h_cut:
                detady = 1.0 * (eta_north - eta_here) * self.one_over_dy

            if h_min.y <= h_cut and h_min.w <= h_cut:
                detadx = 0.0
            elif h_min.y <= h_cut:
                detadx = 1.0 * (eta_here - eta_west) * self.one_over_dx
            elif h_min.w <= h_cut:
                detadx = 1.0 * (eta_east - eta_here) * self.one_over_dx

            overflow_dry = 0.0
            if B_here > 0.0:
                overflow_dry = (
                    -self.infiltrationRate
                )  # hydraulic conductivity of coarse, unsaturated sand

            sx = (
                -self.g * h_here * detadx
                - in_state_here[1] * friction_
                + breaking_x
                + (Psi1x + Psi2x)
                + press_x
            )
            sy = (
                -self.g * h_here * detady
                - in_state_here[2] * friction_
                + breaking_y
                + (Psi1y + Psi2y)
                + press_y
            )

            source_term = ti.Vector(
                [
                    overflow_dry,
                    sx,
                    sy,
                    hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + c_dissipation,
                ],
                self.precision,
            )
            d_by_dt = (
                (xflux_west - xflux_here) * self.one_over_dx
                + (yflux_south - yflux_here) * self.one_over_dy
                + source_term
            )

            # previous derivatives
            oldies = self.oldGradients[i, j]
            oldOldies = self.oldOldGradients[i, j]  # noqa: N806
            newState = zro  # w , hu, hv ,hc  # noqa: N806
            F_G_here = ti.Vector([0.0, F_star, G_star, 0.0], self.precision)  # noqa: N806

            if self.timeScheme == 0:  # if timeScheme is Euler do:
                newState = in_state_here_UV + self.dt * d_by_dt  # noqa: N806

            elif pred_or_corrector == 1:
                newState = in_state_here_UV + self.dt / 12.0 * (  # noqa: N806
                    23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies
                )

            elif pred_or_corrector == 2:  # noqa: PLR2004
                predicted = self.predictedGradients[i, j]
                newState = in_state_here_UV + self.dt / 24.0 * (  # noqa: N806
                    9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + oldOldies
                )

            if self.showBreaking == True:  # noqa: E712
                # add breaking source
                newState.w = ti.max(
                    newState.w, breaking_B
                )  # use the B value from Kennedy et al as a foam intensity
            elif self.showBreaking == 2:  # noqa: PLR2004
                contaminent_source = self.ContSource[i, j].x
                newState.w = ti.min(1.0, newState.w + contaminent_source)

            self.NewState[i, j] = newState
            self.dU_by_dt[i, j] = d_by_dt
            self.predictedF_G_star[i, j] = F_G_here
            self.current_stateUVstar[i, j] = newState

    @ti.kernel
    def Pass_Breaking(self, time: ti.f32):  # noqa: C901, N802, D102
        # PASS Breaking -
        for i, j in self.State:
            # Compute the coordinates of the neighbors
            rightIdx = ti.min(i + 1, self.nx - 1)  # noqa: N806
            upIdx = ti.min(j + 1, self.ny - 1)  # noqa: N806
            leftIdx = ti.max(i - 1, 0)  # noqa: N806
            downIdx = ti.max(j - 1, 0)  # noqa: N806

            xflux_here = self.XFlux[i, j].x
            xflux_west = self.XFlux[leftIdx, j].x

            yflux_here = self.YFlux[i, j].x
            yflux_south = self.YFlux[i, downIdx].x

            P_south = self.State[i, downIdx].y  # noqa: N806
            P_here = self.State[i, j].y  # noqa: N806
            P_north = self.State[i, upIdx].y  # noqa: N806

            Q_west = self.State[leftIdx, j].z  # noqa: N806
            Q_here = self.State[i, j].z  # noqa: N806
            Q_east = self.State[rightIdx, j].z  # noqa: N806

            detadt = self.dU_by_dt[i, j].x

            # Look the dominant direction of flow, and look at the three cells on that 3*3 cube
            t_here = self.Breaking[i, j].x
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0

            if ti.abs(P_here) > ti.abs(Q_here):
                if P_here > 0.0:
                    t1 = self.Breaking[leftIdx, j].x
                    t2 = self.Breaking[leftIdx, upIdx].x
                    t3 = self.Breaking[leftIdx, downIdx].x
                else:
                    t1 = self.Breaking[rightIdx, j].x
                    t2 = self.Breaking[rightIdx, upIdx].x
                    t3 = self.Breaking[rightIdx, downIdx].x
            elif Q_here > 0.0:
                t1 = self.Breaking[i, downIdx].x
                t2 = self.Breaking[rightIdx, downIdx].x
                t3 = self.Breaking[leftIdx, downIdx].x
            else:
                t1 = self.Breaking[i, upIdx].x
                t2 = self.Breaking[rightIdx, upIdx].x
                t3 = self.Breaking[leftIdx, upIdx].x

            t_here = ti.max(t_here, ti.max(t1, ti.max(t2, t3)))

            dPdx = (xflux_here - xflux_west) * self.one_over_dx  # noqa: N806
            dPdy = 0.5 * (P_north - P_south) * self.one_over_dy  # noqa: N806

            dQdx = 0.5 * (Q_east - Q_west) * self.one_over_dx  # noqa: N806
            dQdy = (yflux_here - yflux_south) * self.one_over_dy  # noqa: N806

            B_here = self.Bottom[2, i, j]  # noqa: N806
            eta_here = self.State[i, j].x
            h_here = eta_here - B_here
            c_here = ti.sqrt(self.g * h_here)
            h2 = h_here * h_here
            divide_by_h = 2.0 * h_here / (h2 + ti.max(h2, self.epsilon))

            # Kennedy et al breaking model, default parameters
            T_star = self.T_star_coef * ti.sqrt(h_here / self.g)  # noqa: N806
            dzdt_I = self.dzdt_I_coef * c_here  # noqa: N806
            dzdt_F = self.dzdt_F_coef * c_here  # noqa: N806

            dzdt_star = 0.0

            if t_here <= self.dt:
                dzdt_star = dzdt_I
            elif time - t_here <= T_star:
                dzdt_star = dzdt_I + (time - t_here) / T_star * (dzdt_F - dzdt_I)
            else:
                dzdt_star = dzdt_F

            B_Breaking = 0.0  # noqa: N806
            if detadt < dzdt_star:
                t_here = 0.0
            elif detadt > 2.0 * dzdt_star:
                B_Breaking = 1.0  # noqa: N806
                if t_here <= self.dt:
                    t_here = time
            else:
                B_Breaking = detadt / dzdt_star - 1.0  # noqa: N806
                if t_here <= self.dt:
                    t_here = time

            nu_breaking = ti.min(
                1.0 * self.dx * self.dy / self.dt,
                B_Breaking * self.delta_breaking * h_here * detadt,
            )

            # Smagorinsky subgrid eddy viscosity
            Smag_cm = 0.04  # noqa: N806
            nu_Smag = (  # noqa: N806
                Smag_cm
                * self.dx
                * self.dy
                * ti.sqrt(
                    2.0 * dPdx * dPdx
                    + 2.0 * dQdy * dQdy
                    + (dPdy + dQdx) * (dPdy + dQdx)
                )
                * divide_by_h
            )  # temporary, needs to be corrected to strain rate, right now has extra dHdx terms

            # sum eddy viscosities and calc fluxes
            nu_total = nu_breaking + nu_Smag

            nu_dPdx = nu_total * dPdx  # noqa: N806
            nu_dPdy = nu_total * dPdy  # noqa: N806

            nu_dQdx = nu_total * dQdx  # noqa: N806
            nu_dQdy = nu_total * dQdy  # noqa: N806

            nu_flux = ti.Vector([nu_dPdx, nu_dPdy, nu_dQdx, nu_dQdy], self.precision)
            Bvalues = ti.Vector(  # noqa: N806
                [t_here, nu_breaking, B_Breaking, nu_Smag], self.precision
            )

            self.DissipationFlux[i, j] = nu_flux
            self.Breaking[i, j] = Bvalues

    @ti.kernel
    def TriDiag_PCRx(  # noqa: N802, D102
        self,
        p: int,
        s: int,
        current_buffer: ti.template(),
        next_buffer: ti.template(),
    ):
        for i, j in self.NewState:
            CurrentState = self.NewState[i, j]  # noqa: N806
            idx_left = ti.raw_mod(i - s + self.nx, self.nx)
            idx_right = ti.raw_mod(i + s + self.nx, self.nx)

            aIn, bIn, cIn, dIn = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInLeft, bInLeft, cInLeft, dInLeft = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInRight, bInRight, cInRight, dInRight = 0.0, 0.0, 0.0, 0.0  # noqa: N806

            if p == 0:
                bIn = self.coefMatx[i, j].g  # noqa: N806
                bInLeft = self.coefMatx[idx_left, j].g  # noqa: N806
                bInRight = self.coefMatx[idx_right, j].g  # noqa: N806

                aIn = self.coefMatx[i, j].r / bIn  # noqa: N806
                aInLeft = self.coefMatx[idx_left, j].r / bInLeft  # noqa: N806
                aInRight = self.coefMatx[idx_right, j].r / bInRight  # noqa: N806

                cIn = self.coefMatx[i, j].b / bIn  # noqa: N806
                cInLeft = self.coefMatx[idx_left, j].b / bInLeft  # noqa: N806
                cInRight = self.coefMatx[idx_right, j].b / bInRight  # noqa: N806

                dIn = self.current_stateUVstar[i, j].g / bIn  # noqa: N806
                dInLeft = self.current_stateUVstar[idx_left, j].g / bInLeft  # noqa: N806
                dInRight = self.current_stateUVstar[idx_right, j].g / bInRight  # noqa: N806
            else:
                aIn = current_buffer[i, j][0]  # noqa: N806
                aInLeft = current_buffer[idx_left, j][0]  # noqa: N806
                aInRight = current_buffer[idx_right, j][0]  # noqa: N806

                cIn = current_buffer[i, j][2]  # noqa: N806
                cInLeft = current_buffer[idx_left, j][2]  # noqa: N806
                cInRight = current_buffer[idx_right, j][2]  # noqa: N806

                dIn = current_buffer[i, j][3]  # noqa: N806
                dInLeft = current_buffer[idx_left, j][3]  # noqa: N806
                dInRight = current_buffer[idx_right, j][3]  # noqa: N806

            r = 1.0 / (1.0 - aIn * cInLeft - cIn * aInRight)
            aOut = -r * aIn * aInLeft  # noqa: N806
            cOut = -r * cIn * cInRight  # noqa: N806
            dOut = r * (dIn - aIn * dInLeft - cIn * dInRight)  # noqa: N806

            next_buffer[i, j] = ti.Vector([aOut, 1.0, cOut, dOut], self.precision)
            self.temp2_PCRx[i, j] = ti.Vector(
                [CurrentState.r, dOut, CurrentState.b, CurrentState.a],
                self.precision,
            )

    @ti.kernel
    def TriDiag_PCRy(  # noqa: N802, D102
        self,
        p: int,
        s: int,
        current_buffer: ti.template(),
        next_buffer: ti.template(),
    ):
        for i, j in self.NewState:
            CurrentState = self.NewState[i, j]  # noqa: N806
            idx_left = ti.raw_mod(j - s + self.ny, self.ny)
            idx_right = ti.raw_mod(j + s + self.ny, self.ny)

            aIn, bIn, cIn, dIn = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInLeft, bInLeft, cInLeft, dInLeft = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInRight, bInRight, cInRight, dInRight = 0.0, 0.0, 0.0, 0.0  # noqa: N806

            if p == 0:
                bIn = self.coefMaty[i, j].g  # noqa: N806
                bInLeft = self.coefMaty[i, idx_left].g  # noqa: N806
                bInRight = self.coefMaty[i, idx_right].g  # noqa: N806

                aIn = self.coefMaty[i, j].r / bIn  # noqa: N806
                aInLeft = self.coefMaty[i, idx_left].r / bInLeft  # noqa: N806
                aInRight = self.coefMaty[i, idx_right].r / bInRight  # noqa: N806

                cIn = self.coefMaty[i, j].b / bIn  # noqa: N806
                cInLeft = self.coefMaty[i, idx_left].b / bInLeft  # noqa: N806
                cInRight = self.coefMaty[i, idx_right].b / bInRight  # noqa: N806

                dIn = self.current_stateUVstar[i, j].b / bIn  # noqa: N806
                dInLeft = self.current_stateUVstar[i, idx_left].b / bInLeft  # noqa: N806
                dInRight = self.current_stateUVstar[i, idx_right].b / bInRight  # noqa: N806
            else:
                aIn = current_buffer[i, j][0]  # noqa: N806
                aInLeft = current_buffer[i, idx_left][0]  # noqa: N806
                aInRight = current_buffer[i, idx_right][0]  # noqa: N806

                cIn = current_buffer[i, j][2]  # noqa: N806
                cInLeft = current_buffer[i, idx_left][2]  # noqa: N806
                cInRight = current_buffer[i, idx_right][2]  # noqa: N806

                dIn = current_buffer[i, j][3]  # noqa: N806
                dInLeft = current_buffer[i, idx_left][3]  # noqa: N806
                dInRight = current_buffer[i, idx_right][3]  # noqa: N806

            r = 1.0 / (1.0 - aIn * cInLeft - cIn * aInRight)
            aOut = -r * aIn * aInLeft  # noqa: N806
            cOut = -r * cIn * cInRight  # noqa: N806
            dOut = r * (dIn - aIn * dInLeft - cIn * dInRight)  # noqa: N806

            next_buffer[i, j] = ti.Vector([aOut, 1.0, cOut, dOut], self.precision)
            self.temp2_PCRy[i, j] = ti.Vector(
                [CurrentState.r, CurrentState.g, dOut, CurrentState.a],
                self.precision,
            )

    def Run_Tridiag_solver(self):  # noqa: N802, D102
        if self.model == 'SWE':
            self.copy_states(src=self.current_stateUVstar, dst=self.NewState)
        else:
            for p in range(self.Px):
                s = 1 << p
                if p % 2 == 0:
                    self.TriDiag_PCRx(
                        p=p,
                        s=s,
                        current_buffer=self.temp_PCRx1,
                        next_buffer=self.temp_PCRx2,
                    )
                else:
                    self.TriDiag_PCRx(
                        p=p,
                        s=s,
                        current_buffer=self.temp_PCRx2,
                        next_buffer=self.temp_PCRx1,
                    )
            self.copy_states(src=self.temp2_PCRx, dst=self.NewState)

            for p in range(self.Py):
                s = 1 << p
                if p % 2 == 0:
                    self.TriDiag_PCRy(
                        p=p,
                        s=s,
                        current_buffer=self.temp_PCRy1,
                        next_buffer=self.temp_PCRy2,
                    )
                else:
                    self.TriDiag_PCRy(
                        p=p,
                        s=s,
                        current_buffer=self.temp_PCRy2,
                        next_buffer=self.temp_PCRy1,
                    )
            self.copy_states(src=self.temp2_PCRy, dst=self.NewState)

    # To test pressure
    # @ti.kernel
    # def Ship_pressure(self,px_init:int,py_init:int,steps:int):
    #    pos_x = int(px_init + steps*self.dt)
    #    pos_y = int(py_init + 0.01*steps*self.dt)
    #    self.ShipPressure[pos_x,pos_y].x = 2.5

    @ti.kernel
    def Update_Bottom(self):  # noqa: N802, D102
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            B = self.Bottom[2, i, j]  # noqa: N806
            e = self.erosion_Sed[i, j].x
            d = self.deposition_Sed[i, j].x
            delta_B = self.dt * (e - d) / (1 - self.sediment.p)  # noqa: N806
            self.Bottom[2, i, j] = B + delta_B


if __name__ == '__main__':
    print('Module of functions used in Celeris')  # noqa: T201
