import os  # noqa: INP001, D100

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from celeris.utils import (
    CalcUV,
    CalcUV_Sed,
    FrictionCalc,
    NumericalFlux,
    Reconstruct,
    sineWave,
)
from scipy.interpolate import griddata


@ti.data_oriented
class SedClass:  # noqa: D101
    def __init__(
        self,
        d50=0.01,
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


@ti.data_oriented
class Solver:  # noqa: D101
    def __init__(  # noqa: PLR0913
        self,
        domain=None,
        boundary_conditions=None,
        dissipation_threshold=0.3,
        theta=1.3,
        timeScheme=2,  # noqa: N803
        pred_or_corrector=1,
        show_window=True,  # noqa: FBT002
        maxsteps=1000,
        Bcoef=1.0 / 15.0,  # noqa: N803
        outdir=None,
        model='SWE',
        useSedTransModel=False,  # noqa: FBT002, N803
        whiteWaterDecayRate=0.01,  # noqa: N803
        whiteWaterDispersion=0.1,  # noqa: N803
        sedC1_shields=308.8993914681988,  # noqa: N803
        sedC1_erosion=0.0002746401358265295,  # noqa: N803
        sedC1_fallvel=0.14690813456034352,  # noqa: N803
        sedC1_n=0.4,  # noqa: N803
        sedC1_criticalshields=0.045,  # noqa: N803
        delta=0.05529662583099999,
        infiltrationRate=0.001,  # noqa: N803
        clearCon=1,  # noqa: N803
        useBreakingModel=0,  # noqa: N803
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
        # Bottom [BN, BE, B] # B->bottom
        self.Bottom = self.domain.bottom()
        self.temp_bottom = self.domain.bottom()
        # Physical "state" values, eval at center of cell
        self.State = self.domain.states()  # stores the state (eta, P, Q, hc) at n
        self.TempBound = self.domain.states()
        self.State_Sed = self.domain.states()
        self.BottomFriction = self.domain.states()
        # self.txState = self.domain.states_ad()         # stores the state (eta, P, Q,hc) at n
        # Sediment Transport Vectors
        self.Sed_C1 = self.domain.states()
        self.Sed_C2 = self.domain.states()
        self.Sed_C3 = self.domain.states()
        self.Sed_C4 = self.domain.states()
        self.whiteWaterDecayRate = whiteWaterDecayRate
        self.whiteWaterDispersion = whiteWaterDispersion
        self.sedC1_shields = sedC1_shields
        self.sedC1_erosion = sedC1_erosion
        self.sedC1_fallvel = sedC1_fallvel
        self.sedC1_n = sedC1_n
        self.sedC1_criticalshields = sedC1_criticalshields
        self.delta = delta
        self.infiltrationRate = infiltrationRate
        # breaking model parameters
        self.useBreakingModel = useBreakingModel  # include breaking model when == 1
        self.delta_breaking = delta_breaking  # eddy viscosity coefficient
        self.T_star_coef = T_star_coef  # defines length of time until breaking becomes fully developed
        self.dzdt_I_coef = dzdt_I_coef  # start breaking parameter
        self.dzdt_F_coef = dzdt_F_coef  # end breaking parameter

        self.showBreaking = showBreaking

        # self.current_state =  self.domain.states()  # stores the corrected / final state at n+1
        self.State_pred = self.domain.states()  # stores the predicted state at n+1
        self.current_state_Sed = self.domain.states()
        self.NewState_Sed = self.domain.states()

        # State variables, texture gives values along cell edges
        self.Hnear = self.domain.states()
        self.H = self.domain.states()  # water depth
        self.U = self.domain.states()  # E-W velocity
        self.V = self.domain.states()  # N-S velocity
        self.C = self.domain.states()  # Scalar concentration
        # Normal texture, gives normalized slope of surface, useful for lighting and determining some breaking parameters
        self.Normal = self.domain.states()
        # // r = maxFlowDepth, g=0, b = reconstructed slope, a=breaking_white (for visualization)
        self.Auxiliary2 = self.domain.states()
        # Advective flux vectors F(U)-x - G(U)-y
        self.XFlux = self.domain.states()
        self.YFlux = self.domain.states()

        self.XFlux_Sed = self.domain.states()
        self.YFlux_Sed = self.domain.states()

        self.oldGradients = self.domain.states()
        self.oldOldGradients = self.domain.states()
        self.predictedGradients = self.domain.states()
        self.NewState = self.domain.states()
        self.dU_by_dt = self.domain.states()
        self.in_state_here_UV = self.domain.states()

        # check dimension, no necessary 4
        self.ShipPressure = self.domain.states()
        self.DissipationFlux = self.domain.states()
        self.ContSource = self.domain.states()
        self.Breaking = self.domain.states()

        self.oldGradients_Sed = self.domain.states()
        self.oldOldGradients_Sed = self.domain.states()
        self.predictedGradients_Sed = self.domain.states()

        self.bottom_dxdy = self.domain.states()
        self.coefMatx = self.domain.states()
        self.coefMaty = self.domain.states()
        self.newcoef_x = self.domain.states()
        self.newcoef_y = self.domain.states()
        self.txtemp_PCRx = self.domain.states()
        self.txtemp_PCRy = self.domain.states()
        self.txtemp2_PCRx = self.domain.states()
        self.txtemp2_PCRy = self.domain.states()

        self.StateUVstar = self.domain.states()
        self.current_stateUVstar = self.domain.states()
        self.F_G_star = self.domain.states()
        self.predictedF_G_star = self.domain.states()
        self.F_G_star_oldGradients = (
            self.domain.states()
        )  # stores F* and G* at previous time step
        self.F_G_star_oldOldGradients = (
            self.domain.states()
        )  # stores F* and G* at 2*previous time step
        self.F_G_star_predictedGradients = (
            self.domain.states()
        )  # stores F* and G* at predictor step

        self.R_x = self.domain.reflect_x()
        self.R_y = self.domain.reflect_y()
        self.dx = self.domain.dx()
        self.dy = self.domain.dy()
        self.nSL = self.domain.north_sl
        self.sSL = self.domain.south_sl
        self.eSL = self.domain.east_sl
        self.wSL = self.domain.west_sl
        self.seaLevel = self.domain.seaLevel
        self.maxdepth = self.domain.maxdepth()
        self.maxtopo = self.domain.maxtopo()

        self.WaveData = self.bc.get_data()
        self.Nwaves = self.bc.N_data
        self.init_eta = self.bc.init_eta
        self.dissipation_threshold = dissipation_threshold
        self.whiteWaterDecayRate = whiteWaterDecayRate
        self.theta = theta
        self.epsilon = self.dx / 1000.0
        self.g = self.domain.g
        self.dt = self.domain.dt()
        self.isManning = self.domain.isManning
        self.friction = self.domain.friction
        self.timeScheme = timeScheme
        self.pred_or_corrector = pred_or_corrector
        self.pixel = self.domain.pixels
        self.rgb = self.domain.rgb
        self.nx = self.domain.Nx
        self.ny = self.domain.Ny
        self.pi = np.pi
        self.show_window = show_window
        self.maxsteps = maxsteps
        # % dispersion parameter, 1/15 is optimum value for this set of equations
        self.Bcoef = Bcoef
        self.Bcoef_g = self.Bcoef * self.g
        self.outdir = outdir
        self.model = model

        self.one_over_dx = 1 / self.dx
        self.one_over_dy = 1 / self.dy
        self.one_over_d2x = self.one_over_dx * self.one_over_dx
        self.one_over_d2y = self.one_over_dy * self.one_over_dy
        self.one_over_dxdy = self.one_over_dx * self.one_over_dy
        self.one_over_d3x = self.one_over_d2x * self.one_over_dx
        self.one_over_d3y = self.one_over_d2y * self.one_over_dy
        self.two_theta = 2 * theta
        self.half_g = self.g / 2.0
        self.g_over_dx = self.g / self.dx
        self.g_over_dy = self.g / self.dy

    @ti.kernel
    def remove_single_point_islands(self):  # noqa: D102
        # Remove single-point islands in the bathymetry data
        change_point = 1.0
        while change_point > 0.5:  # noqa: PLR2004
            old_count = ti.atomic_add(self.old_count, 0.0)
            new_count = ti.atomic_add(self.new_count, 0.0)
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    if self.Bottom[2, j, i] >= 0.0:
                        if (
                            self.Bottom[2, i + 1, j] <= 0.0
                            and self.Bottom[2, i - 1, j] <= 0.0
                            and self.Bottom[2, i, j + 1] <= 0.0
                            and self.Bottom[2, i, j - 1] <= 0.0
                        ):
                            self.Bottom[2, i, j] = 0.0
                            ti.atomic_add(self.new_count, 1.0)
            change_point = new_count[None] - old_count[None]

    @ti.kernel
    def fill_bottom_field(self):  # noqa: D102
        lengthCheck = 3  # noqa: N806
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            # self.Bottom[0,j,i] =  (self.Bottom[2,j,i] + self.Bottom[2,j+1,i])/2.  # center - y
            # self.Bottom[1,j,i] =  (self.Bottom[2,j,i+1] + self.Bottom[2,j,i])/2.  # center - x
            self.Bottom[0, j, i] = (
                0.5 * self.Bottom[2, j, i]
                + 0.5 * self.Bottom[2, min(self.ny - 1, j + 1), i]
            )
            self.Bottom[1, j, i] = (
                0.5 * self.Bottom[2, j, i]
                + 0.5 * self.Bottom[2, j, min(self.nx - 1, i + 1)]
            )
            # boolean near-dry check
            for yy in range(
                max(j - lengthCheck, 0), min(j + lengthCheck + 1, self.ny)
            ):
                for xx in range(
                    max(i - lengthCheck, 0), min(i + lengthCheck + 1, self.nx)
                ):
                    if self.Bottom[2, yy, xx] >= 0:
                        self.Bottom[3, j, i] = -99

    @ti.kernel
    def check_depths(self):  # noqa: D102
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            if (self.current_state[j, i][0] + (self.Bottom[3, j, i])) < 0.0:
                self.current_state[j, i][0] = -1.0 * self.Bottom[3, j, i]
                self.current_state[j, i][1] = 0.0
                self.current_state[j, i][2] = 0.0
                self.current_state[j, i][3] = 0.0
            else:
                self.current_state[j, i][0] = 0.0
                self.current_state[j, i][1] = 0.0
                self.current_state[j, i][2] = 0.0
                self.current_state[j, i][3] = 0.0

    @ti.func
    def WavesGenerator(self, NumWaves, Waves, x, y, t, d_here):  # noqa: N802, N803, D102
        result = ti.Vector([0.0, 0.0, 0.0])
        for w in range(NumWaves):
            result += sineWave(
                x, y, t, d_here, Waves[w, 0], Waves[w, 1], Waves[w, 2], Waves[w, 3]
            )
        return result

    #### NEED CHECK HOW WORKS PARALLELIZATION IN/AFTER A CONDITIONAL
    @ti.kernel
    def run_boundaries(self, time: ti.f32, TempBound: ti.template()):  # noqa: C901, N803, D102
        # North boundary
        if self.bc.North == 0:  # Solid
            for i, j in ti.ndrange((0, self.nx), (self.ny - 2, self.ny)):
                TempBound[j, i][0] = TempBound[self.R_y - j, i][0]
                TempBound[j, i][1] = TempBound[self.R_y - j, i][1]
                TempBound[j, i][2] = TempBound[self.R_y - j, i][2] * (-1)
                TempBound[j, i][3] = TempBound[self.R_y - j, i][3]

                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

                TempBound[self.ny - 3, i][2] = 0.0
                self.State_Sed[self.ny - 3, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if self.bc.North == 3:  # IrregularWave  # noqa: PLR2004
            for i, j in ti.ndrange((0, self.nx), (self.ny - 3, self.ny)):
                d_here = max(0, self.nSL - self.Bottom[2, j, i])
                x = i * self.dx
                y = j * self.dy
                eta, hu, hv = 0.0, 0.0, 0.0
                if d_here > 0.0001:  # noqa: PLR2004
                    eta, hu, hv = self.WavesGenerator(
                        self.Nwaves, self.WaveData, x, y, time, d_here
                    )

                TempBound[j, i][0] = eta + self.nSL
                TempBound[j, i][1] = hu
                TempBound[j, i][2] = hv
                TempBound[j, i][3] = 0.0

                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

        # South boundary
        if self.bc.South == 0:  # Solid
            for i, j in ti.ndrange((0, self.nx), (0, 2)):
                TempBound[j, i][0] = TempBound[4 - j, i][0]
                TempBound[j, i][1] = TempBound[4 - j, i][1]
                TempBound[j, i][2] = TempBound[4 - j, i][2] * (-1)
                TempBound[j, i][3] = TempBound[4 - j, i][3]

                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

                TempBound[2, i][2] = 0.0
                self.State_Sed[2, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

        if self.bc.South == 3:  # IrregularWave  # noqa: PLR2004
            for i, j in ti.ndrange((0, self.nx), (self.ny - 3, self.ny)):
                d_here = max(0, self.nSL - self.Bottom[2, j, i])
                x = i * self.dx
                y = j * self.dy
                eta, hu, hv = 0.0, 0.0, 0.0
                if d_here > 0.0001:  # noqa: PLR2004
                    eta, hu, hv = self.WavesGenerator(
                        self.Nwaves, self.WaveData, x, y, time, d_here
                    )

                TempBound[j, i][0] = eta + self.nSL
                TempBound[j, i][1] = hu
                TempBound[j, i][2] = hv
                TempBound[j, i][3] = 0.0

                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

        # East boundary
        if self.bc.East == 0:  # Solid
            for i, j in ti.ndrange((self.nx - 2, self.nx), (0, self.ny)):
                TempBound[j, i][0] = TempBound[j, self.R_x - i][0]
                TempBound[j, i][1] = TempBound[j, self.R_x - i][1] * (-1)
                TempBound[j, i][2] = TempBound[j, self.R_x - i][2]
                TempBound[j, i][3] = TempBound[j, self.R_x - i][3]
                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

                TempBound[j, self.nx - 3][1] = 0.0
                self.State_Sed[j, self.nx - 3] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if self.bc.East == 3:  # IrregularWave  # noqa: PLR2004
            for i, j in ti.ndrange((0, self.nx), (self.ny - 3, self.ny)):
                d_here = max(0, self.nSL - self.Bottom[2, j, i])
                x = i * self.dx
                y = j * self.dy
                eta, hu, hv = 0.0, 0.0, 0.0
                if d_here > 0.0001:  # noqa: PLR2004
                    eta, hu, hv = self.WavesGenerator(
                        self.Nwaves, self.WaveData, x, y, time, d_here
                    )

                TempBound[j, i][0] = eta + self.eSL
                TempBound[j, i][1] = hu
                TempBound[j, i][2] = hv
                TempBound[j, i][3] = 0.0

                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

        # West boundary
        if self.bc.West == 0:  # Solid
            for i, j in ti.ndrange((0, 2), (0, self.ny)):
                TempBound[j, i][0] = TempBound[j, 4 - i][0]
                TempBound[j, i][1] = TempBound[j, 4 - i][1] * (-1)
                TempBound[j, i][2] = TempBound[j, 4 - i][2]
                TempBound[j, i][3] = TempBound[j, 4 - i][3]

                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

                TempBound[j, 2][1] = 0.0
                self.State_Sed[j, 2] = ti.Vector([0.0, 0.0, 0.0, 0.0])

        if self.bc.West == 3:  # IrregularWave  # noqa: PLR2004
            # for i,j in ti.ndrange((0,3),(0,self.ny)):
            for i in range(3):
                for j in range(self.ny):
                    B_here = -self.maxdepth  # noqa: N806
                    d_here = max(0, self.wSL - B_here)
                    x = i * self.dx
                    y = j * self.dy
                    eta, hu, hv = 0.0, 0.0, 0.0
                    if d_here > 0.0001:  # noqa: PLR2004
                        eta, hu, hv = self.WavesGenerator(
                            self.Nwaves, self.WaveData, x, y, time, d_here
                        )

                    TempBound[j, i][0] = eta + self.wSL
                    TempBound[j, i][1] = hu
                    TempBound[j, i][2] = hv
                    TempBound[j, i][3] = 0.0

                    self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

        if self.bc.West == 4 and time == 0:  # ~Dam break  # noqa: PLR2004
            width = int(self.init_eta)
            for i, j in ti.ndrange((0, width), (0, self.ny)):
                TempBound[j, i][0] = self.init_eta
                TempBound[j, i][1] = 0
                TempBound[j, i][2] = 0
                TempBound[j, i][3] = 0

                self.State_Sed[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

    @ti.kernel
    def copy_states(self, src: ti.template(), dst: ti.template()):  # noqa: D102
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            dst[j, i] = src[j, i]

    @ti.kernel
    def bottom_derivatives(self):  # noqa: D102
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.bottom_dxdy[j, i][0] = (
                -self.Bottom[2, j, i + 1] + self.Bottom[2, j, i - 1]
            ) / (2 * self.dx)
            self.bottom_dxdy[j, i][1] = (
                -self.Bottom[2, j + 1, i] + self.Bottom[2, j - 1, i]
            ) / (2 * self.dy)

    @ti.kernel
    def tridiag_coeffs(self):  # noqa: D102
        # to calculate the flux terms - Tridiagonal system pag. 12
        for i, j in ti.ndrange((2, self.nx - 2), (2, self.ny - 2)):
            d = -self.Bottom[2, j, i]
            d_dx = self.bottom_dxdy[j, i][0]
            d_dy = self.bottom_dxdy[j, i][1]

            self.coefMatx[j, i][0] = (
                d * d_dx / (6.0 * self.dx)
                - (self.Bcoef + 1.0 / 3) * d**2 / self.dx**2
            )
            self.coefMatx[j, i][1] = (
                1.0 + 2 * (self.Bcoef + 1.0 / 3) * d**2 / self.dx**2
            )
            self.coefMatx[j, i][2] = (
                -d * d_dx / (6.0 * self.dx)
                - (self.Bcoef + 1.0 / 3) * d**2 / self.dx**2
            )
            if (
                i == 2 or i == self.nx - 2  # noqa: PLR2004
            ):  # P_3=0  % for solid wall, coefMatx(i,j,3) will be set to zero, for wave input it will be set to wave velocity
                self.coefMatx[j, i][0] = 0
                self.coefMatx[j, i][1] = 1
                self.coefMatx[j, i][2] = 0

            self.coefMaty[j, i][0] = (
                d * d_dy / (6.0 * self.dy)
                - (self.Bcoef + 1.0 / 3) * d**2 / self.dy**2
            )
            self.coefMaty[j, i][1] = (
                1.0 + 2 * (self.Bcoef + 1.0 / 3) * d**2 / self.dy**2
            )
            self.coefMaty[j, i][2] = (
                -d * d_dy / (6.0 * self.dy)
                - (self.Bcoef + 1.0 / 3) * d**2 / self.dy**2
            )
            if j == 2 or j == self.ny - 2:  # noqa: PLR2004
                self.coefMaty[j, i][0] = 0
                self.coefMaty[j, i][1] = 1
                self.coefMaty[j, i][2] = 0

    @ti.kernel
    def tridiag_coeffs_X(self):  # noqa: N802, D102
        # to calculate the flux terms - Tridiagonal system pag. 12
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            a, b, c = 0.0, 1.0, 0.0
            neardry = self.Bottom[3, j, i]
            depth_here = -self.Bottom[2, j, i]
            depth_plus = -self.Bottom[2, j, min(i + 1, self.nx - 1)]
            depth_minus = -self.Bottom[2, j, max(i - 1, 0)]
            if i <= 2 or i >= (self.nx - 3) or neardry < 0.0:  # noqa: PLR2004
                a = 0.0
                b = 1.0
                c = 0.0
            else:
                # Calculate the first derivative
                d_dx = (depth_plus - depth_minus) / (2.0 * self.dx)
                # Calculate coefficients
                a = depth_here * d_dx / (6.0 * self.dx) - (
                    self.Bcoef + 1.0 / 3.0
                ) * depth_here * depth_here / (self.dx * self.dx)
                b = 1.0 + 2.0 * (
                    self.Bcoef + 1.0 / 3.0
                ) * depth_here * depth_here / (self.dx * self.dx)
                c = -depth_here * d_dx / (6.0 * self.dx) - (
                    self.Bcoef + 1.0 / 3.0
                ) * depth_here * depth_here / (self.dx * self.dx)
            self.coefMatx[j, i] = ti.Vector([a, b, c, 0.0])

    @ti.kernel
    def tridiag_coeffs_Y(self):  # noqa: N802, D102
        # to calculate the flux terms - Tridiagonal system pag. 12
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            a, b, c = 0.0, 1.0, 0.0
            neardry = self.Bottom[3, j, i]
            depth_here = -self.Bottom[2, j, i]
            depth_plus = -self.Bottom[2, min(j + 1, self.ny - 1), i]
            depth_minus = -self.Bottom[2, max(j - 1, 0), i]

            if j <= 2 or j >= self.ny - 3 or neardry < 0.0:  # noqa: PLR2004
                a = 0.0
                b = 1.0
                c = 0.0
            else:
                # Calculate the first derivative
                d_dy = (depth_plus - depth_minus) / (2.0 * self.dy)
                # Calculate coefficients
                a = depth_here * d_dy / (6.0 * self.dy) - (
                    self.Bcoef + 1.0 / 3.0
                ) * depth_here * depth_here / (self.dy * self.dy)
                b = 1.0 + 2.0 * (
                    self.Bcoef + 1.0 / 3.0
                ) * depth_here * depth_here / (self.dy * self.dy)
                c = -depth_here * d_dy / (6.0 * self.dy) - (
                    self.Bcoef + 1.0 / 3.0
                ) * depth_here * depth_here / (self.dy * self.dy)
            # Store the coefficients in the texture field
            self.coefMaty[j, i] = ti.Vector([a, b, c, 0.0])

    @ti.kernel
    def Pass1(self):  # noqa: N802, D102
        # PASS 0 and Pass1 - edge value construction
        # using Generalized minmod limiter
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            pj = j
            pi = i
            # Compute the coordinates of the neighbors
            rightIdx = min(pi + 1, self.nx - 1)  # noqa: N806
            upIdx = min(pj + 1, self.ny - 1)  # noqa: N806
            leftIdx = max(pi - 1, 0)  # noqa: N806
            downIdx = max(pj - 1, 0)  # noqa: N806

            # Read in the state of the water at this pixel and its neighbors [eta,hu,hv,c]
            in_here = self.State[pj, pi]
            in_S = self.State[downIdx, pi]  # noqa: N806
            in_N = self.State[upIdx, pi]  # noqa: N806
            in_W = self.State[pj, leftIdx]  # noqa: N806
            in_E = self.State[pj, rightIdx]  # noqa: N806

            B_here = self.Bottom[2, pj, pi]  # noqa: N806
            B_south = self.Bottom[2, downIdx, pi]  # noqa: N806
            B_north = self.Bottom[2, upIdx, pi]  # noqa: N806
            B_west = self.Bottom[2, pj, leftIdx]  # noqa: N806
            B_east = self.Bottom[2, pj, rightIdx]  # noqa: N806

            # h = eta - B
            h_here = in_here[0] - B_here
            h_south = in_S[0] - B_south
            h_north = in_N[0] - B_north
            h_west = in_W[0] - B_west
            h_east = in_E[0] - B_east

            # Define h_near = eta_near - B_near
            # HNear_vec neighbours [hN,hE,hS,hW]
            # Write Hnear
            self.Hnear[j, i] = [h_north, h_east, h_south, h_west]

            # To avoid unnecessary computations
            h_cut = self.delta
            if h_here <= h_cut:
                if (
                    h_north <= h_cut
                    and h_east <= h_cut
                    and h_south <= h_cut
                    and h_west <= h_cut
                ):
                    self.H[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    self.U[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    self.V[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    self.C[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    continue

            ########################################################
            # Pass 1
            # Load bed elevation data for this cell's edges.
            # B neighbours [BN,BE,BS,BW]
            B = ti.Vector([0.0, 0.0, 0.0, 0.0])  # noqa: N806
            B[0] = self.Bottom[0, pj, pi]
            B[1] = self.Bottom[1, pj, pi]
            B[2] = self.Bottom[0, downIdx, pi]
            B[3] = self.Bottom[1, pj, leftIdx]

            dB_max = ti.Vector([0.0, 0.0, 0.0, 0.0])  # noqa: N806

            dB_west = abs(B_here - B_west)  # noqa: N806
            dB_east = abs(B_here - B_east)  # noqa: N806
            dB_south = abs(B_here - B_south)  # noqa: N806
            dB_north = abs(B_here - B_north)  # noqa: N806

            # Initialize variables for water height, momentum components, and standard deviation
            h = ti.Vector([0.0, 0.0, 0.0, 0.0])
            w = ti.Vector([0.0, 0.0, 0.0, 0.0])
            hu = ti.Vector([0.0, 0.0, 0.0, 0.0])
            hv = ti.Vector([0.0, 0.0, 0.0, 0.0])
            hc = ti.Vector([0.0, 0.0, 0.0, 0.0])

            # modify limiters based on whether near the inundation limit
            wetdry = min(h_here, min(h_south, min(h_north, min(h_west, h_east))))  # noqa: PLW3301
            rampcoef = min(max(0.0, wetdry / (0.02 * self.maxdepth)), 1.0)
            # transition to full upwinding near the shoreline / inundation limit, start transition with a total water depth of base_depth/50
            TWO_THETAc = self.two_theta * rampcoef + 2.0 * (1.0 - rampcoef)  # noqa: N806

            if wetdry <= self.epsilon:
                dB_max[0] = 0.5 * dB_north
                dB_max[1] = 0.5 * dB_east
                dB_max[2] = 0.5 * dB_south
                dB_max[3] = 0.5 * dB_west

            # Reconstruction eta
            w[3], w[1] = Reconstruct(in_W[0], in_here[0], in_E[0], TWO_THETAc)
            w[2], w[0] = Reconstruct(in_S[0], in_here[0], in_N[0], TWO_THETAc)

            # Reconstruct h from (corrected) w
            h = w - B
            h = ti.max(h, ti.Vector([0.0, 0.0, 0.0, 0.0]))

            # Reconstruction hu ~P
            hu[3], hu[1] = Reconstruct(in_W[1], in_here[1], in_E[1], TWO_THETAc)
            hu[2], hu[0] = Reconstruct(in_S[1], in_here[1], in_N[1], TWO_THETAc)

            # Reconstruction hv - Q
            hv[3], hv[1] = Reconstruct(in_W[2], in_here[2], in_E[2], TWO_THETAc)
            hv[2], hv[0] = Reconstruct(in_S[2], in_here[2], in_N[2], TWO_THETAc)

            # Reconstruction hc - scalar - sediment concentration or contaminent
            hc[3], hc[1] = Reconstruct(in_W[3], in_here[3], in_E[3], TWO_THETAc)
            hc[2], hc[0] = Reconstruct(in_S[3], in_here[3], in_N[3], TWO_THETAc)

            output_u, output_v, output_c = CalcUV(
                h, hu, hv, hc, self.epsilon, dB_max
            )

            # Froude number limiter
            epsilon_c = ti.max(self.epsilon, dB_max)
            divide_by_h = 2.0 * h / (h**2 + max(h**2, epsilon_c))

            speed = tm.sqrt(output_u * output_u + output_v * output_v)
            Fr = speed / tm.sqrt(9.81 / divide_by_h)  # noqa: N806
            Frumax = max(Fr.x, max(Fr.y, max(Fr.z, Fr.w)))  # noqa: N806, PLW3301
            dBdx = abs(B_east - B_west) / (2.0 * self.dx)  # noqa: N806
            dBdy = abs(B_north - B_south) / (2.0 * self.dy)  # noqa: N806
            dBds_max = ti.max(dBdx, dBdy)  # noqa: N806
            # max Fr allowed on slopes less than 45 degrees is 3; for very steep slopes, artificially slow velocity - physics are just completely wrong here anyhow
            Fr_maxallowed = 3.0 / ti.max(1.0, dBds_max)  # noqa: N806
            if Frumax > Fr_maxallowed:
                Fr_red = Fr_maxallowed / Frumax  # noqa: N806
                output_u = output_u * Fr_red
                output_v = output_v * Fr_red

            # Calculate normal
            normal = ti.Vector([0.0, 0.0, 0.0])
            normal[0] = (in_W[0] - in_E[0]) * self.one_over_dx
            normal[1] = (in_S[0] - in_N[0]) * self.one_over_dy
            normal[2] = 2
            normal = normal / normal.normalized()

            maxInundatedDepth = max(  # noqa: N806
                (h[0] + h[1] + h[2] + h[3]) / 4, self.Auxiliary2[pj, pi][0]
            )

            # output_n = ti.Vector([normal[0], normal[1], normal[2], 0])
            breaking_white = self.Auxiliary2[pj, pi][3]

            # planing delete this part linked to old reconstruct function
            if (
                0.5 * tm.sign(normal[0] * in_here[1] + normal[1] * in_here[2])
                > self.dissipation_threshold
            ):
                breaking_white = 1.0
            output_aux = ti.Vector([maxInundatedDepth, 0, 0.5, breaking_white])

            # Write H, U, V, C vector fields
            self.H[j, i] = h
            self.U[j, i] = output_u
            self.V[j, i] = output_v
            self.C[j, i] = output_c
            self.Auxiliary2[j, i] = output_aux

    @ti.kernel
    def Pass1_SedTrans(self):  # noqa: N802, D102
        # PASS 0 and Pass1 - edge value construction
        # Reconstruct h ,hu,hv, four edges of each cell
        # using Generalized minmod limiter

        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            pi = min(self.nx - 2, max(1, i))
            pj = min(self.ny - 2, max(1, j))

            in_here = self.State_Sed[pj, pi]
            in_S = self.State_Sed[pj - 1, pi]  # noqa: N806
            in_N = self.State_Sed[pj + 1, pi]  # noqa: N806
            in_W = self.State_Sed[pj, pi - 1]  # noqa: N806
            in_E = self.State_Sed[pj, pi + 1]  # noqa: N806

            B_here = self.Bottom[2, pj, pi]  # noqa: N806
            B_south = self.Bottom[2, pj - 1, pi]  # noqa: N806
            B_north = self.Bottom[2, pj + 1, pi]  # noqa: N806
            B_west = self.Bottom[2, pj, pi - 1]  # noqa: N806
            B_east = self.Bottom[2, pj, pi + 1]  # noqa: N806

            dB_west = abs(B_here - B_west)  # noqa: N806
            dB_east = abs(B_here - B_east)  # noqa: N806
            dB_south = abs(B_here - B_south)  # noqa: N806
            dB_north = abs(B_here - B_north)  # noqa: N806
            dB_max = ti.Vector([0.0, 0.0, 0.0, 0.0])  # noqa: N806
            dB_max[0] = 0.5 * dB_north
            dB_max[1] = 0.5 * dB_east
            dB_max[2] = 0.5 * dB_south
            dB_max[3] = 0.5 * dB_west

            h_here = in_here[0] - B_here
            h_south = in_S[0] - B_south
            h_north = in_N[0] - B_north
            h_west = in_W[0] - B_west
            h_east = in_E[0] - B_east

            # Pass 1
            #  Initialize variables for water height, momentum components, and standard deviation
            hc1 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            hc2 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            hc3 = ti.Vector([0.0, 0.0, 0.0, 0.0])
            hc4 = ti.Vector([0.0, 0.0, 0.0, 0.0])

            # modify limiters based on whether near the inundation limit
            wetdry = min(h_here, min(h_south, min(h_north, min(h_west, h_east))))  # noqa: PLW3301
            rampcoef = min(max(0.0, wetdry / (0.02 * self.maxdepth)), 1.0)
            # transition to full upwinding near the shoreline / inundation limit, start transition with a total water depth of base_depth/50
            TWO_THETAc = self.two_theta * rampcoef + 2.0 * (1.0 - rampcoef)  # noqa: N806

            # Reconstruction class1
            hc1[3], hc1[1] = Reconstruct(in_W[0], in_here[0], in_E[0], TWO_THETAc)
            hc1[2], hc1[0] = Reconstruct(in_S[0], in_here[0], in_N[0], TWO_THETAc)

            # Reconstruction class2
            hc2[3], hc2[1] = Reconstruct(in_W[1], in_here[1], in_E[1], TWO_THETAc)
            hc2[2], hc2[0] = Reconstruct(in_S[1], in_here[1], in_N[1], TWO_THETAc)

            # Reconstruction class3
            hc3[3], hc3[1] = Reconstruct(in_W[2], in_here[2], in_E[2], TWO_THETAc)
            hc3[2], hc3[0] = Reconstruct(in_S[2], in_here[2], in_N[2], TWO_THETAc)

            # Reconstruction class4
            hc4[3], hc4[1] = Reconstruct(in_W[3], in_here[3], in_E[3], TWO_THETAc)
            hc4[2], hc4[0] = Reconstruct(in_S[3], in_here[3], in_N[3], TWO_THETAc)

            h = self.H[pj, pi]

            output_c1, output_c2, output_c3, output_c4 = CalcUV_Sed(
                h, hc1, hc2, hc3, hc4, self.epsilon, dB_max
            )

            # Write Sediment concentration by classes
            self.Sed_C1[j, i] = output_c1
            self.Sed_C2[j, i] = output_c2
            self.Sed_C3[j, i] = output_c3
            self.Sed_C4[j, i] = output_c4

    @ti.kernel
    def Pass2(self):  # noqa: N802, D102
        # PASS 2 - Calculus of fluxes
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            pi = i
            pj = j
            rightIdx = min(pi + 1, self.nx - 1)  # noqa: N806
            upIdx = min(pj + 1, self.ny - 1)  # noqa: N806
            leftIdx = max(pi - 1, 0)  # noqa: N806
            downIdx = max(pj - 1, 0)  # noqa: N806

            h_vec = self.Hnear[pj, pi]

            h_here = ti.Vector([0.0, 0.0])
            h_here[0] = self.H[pj, pi][0]
            h_here[1] = self.H[pj, pi][1]

            hW_east = self.H[pj, rightIdx][3]  # noqa: N806
            hS_north = self.H[upIdx, pi][2]  # noqa: N806

            u_here = ti.Vector([0.0, 0.0])
            u_here[0] = self.U[pj, pi][0]
            u_here[1] = self.U[pj, pi][1]

            uW_east = self.U[pj, rightIdx][3]  # noqa: N806
            uS_north = self.U[upIdx, pi][2]  # noqa: N806

            v_here = ti.Vector([0.0, 0.0])
            v_here[0] = self.V[pj, pi][0]
            v_here[1] = self.V[pj, pi][1]

            vW_east = self.V[pj, rightIdx][3]  # noqa: N806
            vS_north = self.V[upIdx, pi][2]  # noqa: N806

            # Compute wave speeds
            cNE = tm.sqrt(self.g * h_here)  # noqa: N806
            cW = tm.sqrt(self.g * hW_east)  # cW evaluated at (j+1, k)  # noqa: N806
            cS = tm.sqrt(self.g * hS_north)  # cS evaluated at (j, k+1)  # noqa: N806

            # Compute propagation speeds
            aplus = max(max(u_here[1] + cNE[1], uW_east + cW), 0.0)  # noqa: PLW3301
            aminus = min(min(u_here[1] - cNE[1], uW_east - cW), 0.0)  # noqa: PLW3301
            bplus = max(max(v_here[0] + cNE[0], vS_north + cS), 0.0)  # noqa: PLW3301
            bminus = min(min(v_here[0] - cNE[0], vS_north - cS), 0.0)  # noqa: PLW3301

            B_here = self.Bottom[2, pj, pi]  # noqa: N806
            dB = max(  # noqa: N806, F841, PLW3301
                self.Bottom[2, downIdx, pi] - B_here,
                max(  # noqa: PLW3301
                    self.Bottom[2, upIdx, pi] - B_here,
                    max(
                        self.Bottom[2, pj, leftIdx] - B_here,
                        self.Bottom[2, pj, rightIdx] - B_here,
                    ),
                ),
            )

            near_dry = self.Bottom[  # noqa: F841
                3, pj, pi
            ]  # Check the value of this field not used in here

            c_here = ti.Vector([0.0, 0.0])
            c_here[0] = self.C[pj, pi][0]
            c_here[1] = self.C[pj, pi][1]

            cW_east = self.C[pj, rightIdx][3]  # noqa: N806
            cS_north = self.C[upIdx, pi][2]  # noqa: N806

            phix = 0.5
            phiy = 0.5

            minH = min(h_vec.w, min(h_vec.z, min(h_vec.y, h_vec.x)))  # noqa: N806, PLW3301
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

            xflux = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
            yflux = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
            auxiliary = ti.Vector([0.0, 0.0, 0.0, 0.0], float)  # noqa: F841

            xflux[0] = NumericalFlux(
                aplus, aminus, hW_east * uW_east, h_here.y * u_here.y, mass_diff_x
            )
            xflux[1] = NumericalFlux(
                aplus, aminus, hW_east * uW_east**2, h_here.y * u_here.y**2, P_diff_x
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
                hS_north * vS_north**2,
                h_here.x * v_here.x**2,
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
            self.XFlux[j, i] = xflux
            self.YFlux[j, i] = yflux

            breaking_white = self.Auxiliary2[pj, pi][3]
            breaking_white *= abs(self.whiteWaterDecayRate) ** self.dt

            self.Auxiliary2[j, i] = [
                self.Auxiliary2[pj, pi][0],
                0.0,
                self.Auxiliary2[pj, pi][2],
                breaking_white,
            ]

            if self.useSedTransModel == True:  # noqa: E712
                c1_here = ti.Vector([0.0, 0.0])
                c2_here = ti.Vector([0.0, 0.0])
                c3_here = ti.Vector([0.0, 0.0])
                c4_here = ti.Vector([0.0, 0.0])

                c1_here[0] = self.Sed_C1[pj, pi][0]
                c1_here[1] = self.Sed_C1[pj, pi][1]
                c1W_east = self.Sed_C1[pj, rightIdx][3]  # noqa: N806
                c1S_north = self.Sed_C1[upIdx, pi][2]  # noqa: N806

                c2_here[0] = self.Sed_C2[pj, pi][0]
                c2_here[1] = self.Sed_C2[pj, pi][1]
                c2W_east = self.Sed_C2[pj, rightIdx][3]  # noqa: N806
                c2S_north = self.Sed_C2[upIdx, pi][2]  # noqa: N806

                c3_here[0] = self.Sed_C3[pj, pi][0]
                c3_here[1] = self.Sed_C3[pj, pi][1]
                c3W_east = self.Sed_C3[pj, rightIdx][3]  # noqa: N806
                c3S_north = self.Sed_C3[upIdx, pi][2]  # noqa: N806

                c4_here[0] = self.Sed_C4[pj, pi][0]
                c4_here[1] = self.Sed_C4[pj, pi][1]
                c4W_east = self.Sed_C4[pj, rightIdx][3]  # noqa: N806
                c4S_north = self.Sed_C4[upIdx, pi][2]  # noqa: N806

                xflux_Sed = ti.Vector([0.0, 0.0, 0.0, 0.0], float)  # noqa: N806
                yflux_Sed = ti.Vector([0.0, 0.0, 0.0, 0.0], float)  # noqa: N806

                xflux_Sed[0] = NumericalFlux(
                    aplus,
                    aminus,
                    hW_east * uW_east * c1W_east,
                    h_here[1] * u_here[1] * c1_here[1],
                    phix * (hW_east * c1W_east - h_here[1] * c1_here[1]),
                )
                xflux_Sed[1] = NumericalFlux(
                    aplus,
                    aminus,
                    hW_east * uW_east * c2W_east,
                    h_here[1] * u_here[1] * c2_here[1],
                    phix * (hW_east * c2W_east - h_here[1] * c2_here[1]),
                )
                xflux_Sed[2] = NumericalFlux(
                    aplus,
                    aminus,
                    hW_east * uW_east * c3W_east,
                    h_here[1] * u_here[1] * c3_here[1],
                    phix * (hW_east * c3W_east - h_here[1] * c3_here[1]),
                )
                xflux_Sed[3] = NumericalFlux(
                    aplus,
                    aminus,
                    hW_east * uW_east * c4W_east,
                    h_here[1] * u_here[1] * c4_here[1],
                    phix * (hW_east * c4W_east - h_here[1] * c4_here[1]),
                )

                yflux_Sed[0] = NumericalFlux(
                    bplus,
                    bminus,
                    hS_north * c1S_north * vS_north,
                    h_here[0] * c1_here[0] * v_here[0],
                    phiy * (hS_north * c1S_north - h_here[0] * c1_here[0]),
                )
                yflux_Sed[1] = NumericalFlux(
                    bplus,
                    bminus,
                    hS_north * c2S_north * vS_north,
                    h_here[0] * c2_here[0] * v_here[0],
                    phiy * (hS_north * c2S_north - h_here[0] * c2_here[0]),
                )
                yflux_Sed[2] = NumericalFlux(
                    bplus,
                    bminus,
                    hS_north * c3S_north * vS_north,
                    h_here[0] * c3_here[0] * v_here[0],
                    phiy * (hS_north * c3S_north - h_here[0] * c3_here[0]),
                )
                yflux_Sed[3] = NumericalFlux(
                    bplus,
                    bminus,
                    hS_north * c4S_north * vS_north,
                    h_here[0] * c4_here[0] * v_here[0],
                    phiy * (hS_north * c4S_north - h_here[0] * c4_here[0]),
                )

                # Write Fluxes sediment by classes
                self.XFlux_Sed[j, i] = xflux_Sed
                self.YFlux_Sed[j, i] = yflux_Sed

    @ti.kernel
    def Pass3(self):  # noqa: C901, N802, D102
        # PASS 3 - Do timestep and calculate new w_bar, hu_bar, hv_bar.
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            pi = i
            pj = j
            if pi >= (self.nx - 2) or pj >= (self.ny - 2) or pi <= 1 or pj <= 1:
                # self.current_state[j,i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                # self.predictedGradients[j,i]= ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.NewState[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.dU_by_dt[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.F_G_star[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.current_stateUVstar[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                continue

            # Load values from txBottom using index
            B_here = self.Bottom[2, pj, pi]  # noqa: N806

            in_state_here = self.State[
                pj, pi
            ]  # w, hu and hv, c (cell avgs, evaluated here)
            in_state_here_UV = self.StateUVstar[pj, pi]  # noqa: N806

            B_south = self.Bottom[2, pj - 1, pi]  # noqa: N806
            B_north = self.Bottom[2, pj + 1, pi]  # noqa: N806
            B_west = self.Bottom[2, pj, pi - 1]  # noqa: N806
            B_east = self.Bottom[2, pj, pi + 1]  # noqa: N806

            eta_here = in_state_here[0]
            eta_west = self.State[pj, pi - 1][0]
            eta_east = self.State[pj, pi + 1][0]
            eta_south = self.State[pj - 1, pi][0]
            eta_north = self.State[pj + 1, pi][0]

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
                    # self.current_state[j,i] = ti.Vector([0.0, 0.0, 0.0, 0.0],float)
                    # self.predictedGradients[j,i] = ti.Vector([0.0, 0.0, 0.0, 0.0],float)
                    self.NewState[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
                    self.dU_by_dt[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0], float)

                    self.F_G_star[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
                    self.current_stateUVstar[j, i] = ti.Vector(
                        [0.0, 0.0, 0.0, 0.0], float
                    )
                    continue

            h_min = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
            h_min[0] = min(h_here, h_north)
            h_min[1] = min(h_here, h_east)
            h_min[2] = min(h_here, h_south)
            h_min[3] = min(h_here, h_west)

            detadx = 0.5 * (eta_east - eta_west) * self.one_over_dx
            detady = 0.5 * (eta_north - eta_south) * self.one_over_dy

            # Load values from txXFlux and txYFlux using idx
            xflux_here = self.XFlux[pj, pi]  # at i+1/2
            xflux_west = self.XFlux[pj, pi - 1]  # at i-1/2
            yflux_here = self.YFlux[pj, pi]  # at j+1/2
            yflux_south = self.YFlux[pj - 1, pi]  # at j-1/2

            friction_here = max(self.friction, self.BottomFriction[pj, pi][0])
            friction_ = FrictionCalc(
                in_state_here[1],
                in_state_here[2],
                h_here,
                self.maxdepth,
                self.delta,
                self.epsilon,
                self.isManning,
                self.g,
                friction_here,
            )

            # Pressure stencil calculations
            P_left = self.ShipPressure[pj, pi - 1].x  # noqa: N806
            P_right = self.ShipPressure[pj, pi + 1].x  # noqa: N806
            P_down = self.ShipPressure[pj - 1, pi].x  # noqa: N806
            P_up = self.ShipPressure[pj + 1, pi].x  # noqa: N806

            press_x = -0.5 * h_here * self.g_over_dx * (P_right - P_left)
            press_y = -0.5 * h_here * self.g_over_dy * (P_up - P_down)

            # Calculate scalar transport additions
            C_state_here = self.State[pj, pi][3]  # noqa: N806

            C_state_left = self.State[pj, pi - 1][3]  # noqa: N806
            C_state_right = self.State[pj, pi + 1][3]  # noqa: N806
            C_state_up = self.State[pj + 1, pi][3]  # noqa: N806
            C_state_down = self.State[pj - 1, pi][3]  # noqa: N806
            C_state_up_left = self.State[pj + 1, pi - 1][3]  # noqa: N806
            C_state_up_right = self.State[pj + 1, pi + 1][3]  # noqa: N806
            C_state_down_left = self.State[pj - 1, pi - 1][3]  # noqa: N806
            C_state_down_right = self.State[pj - 1, pi + 1][3]  # noqa: N806

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
            if self.useBreakingModel == 1:
                breaking_B = self.Breaking[  # noqa: N806
                    pj, pi
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
                ]
            )

            d_by_dt = (
                (xflux_west - xflux_here) * self.one_over_dx
                + (yflux_south - yflux_here) * self.one_over_dy
                + source_term
            )

            # previous derivatives
            oldies = self.oldGradients[pj, pi]
            oldOldies = self.oldOldGradients[pj, pi]  # noqa: N806

            newState = ti.Vector(  # noqa: N806
                [0.0, 0.0, 0.0, 0.0], float
            )  # w , hu, hv ,hc

            if self.timeScheme == 0:  # if timeScheme is Euler do:
                newState = in_state_here_UV + self.dt * d_by_dt  # noqa: N806
                # Out = in_state_here + self.dt * d_by_dt

            elif self.pred_or_corrector == 1:  # if time scheme is predictor
                newState = in_state_here_UV + self.dt / 12.0 * (  # noqa: N806
                    23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies
                )
                # Out = in_state_here + self.dt/12. * (23.*d_by_dt - 16.*oldies + 5.*oldOldies)

            elif (
                self.pred_or_corrector == 2  # noqa: PLR2004
            ):  # if time scheme is corrector
                # check this line, in the Pat's models this is saved in other vector
                predicted = self.predictedGradients[pj, pi]
                # corrector step
                newState = in_state_here_UV + self.dt / 24.0 * (  # noqa: N806
                    9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + 1.0 * oldOldies
                )
                # Out = in_state_here + self.dt/24. * (9.*d_by_dt + 19.*predicted - 5.*oldies + 1.*oldOldies)

            # add breaking source
            newState.a = max(
                newState.a, breaking_B
            )  # use the B value from Kennedy et al as a foam intensity

            if self.showBreaking == 1:
                # add breaking source
                newState.a = max(
                    newState.a, breaking_B
                )  # use the B value from Kennedy et al as a foam intensity
            elif self.showBreaking == 2:  # noqa: PLR2004
                contaminent_source = self.ContSource[pj, pi]
                newState.a = min(1.0, newState.a + contaminent_source)

            F_G_vec = ti.Vector([0.0, 0.0, 0.0, 1.0], float)  # noqa: N806

            self.NewState[j, i] = newState
            self.dU_by_dt[j, i] = d_by_dt  # in Pat's model this is aved on DU_by_dt
            self.F_G_star[pj, pi] = F_G_vec
            self.current_stateUVstar[pj, pi] = newState

    @ti.kernel
    def Pass3_SedTrans(self):  # noqa: N802, D102
        # PASS 3 - Do timestep and calculate new w_bar, hu_bar, hv_bar.

        for i, j in ti.ndrange((2, self.nx - 2), (2, self.ny - 2)):
            pi = i
            pj = j

            # Load values from txXFlux and txYFlux using idx
            xflux_here = self.XFlux_Sed[pj, pi]  # at i+1/2
            xflux_west = self.XFlux_Sed[pj, pi - 1]  # at i-1/2
            yflux_here = self.YFlux_Sed[pj, pi]  # at j+1/2
            yflux_south = self.YFlux_Sed[pj - 1, pi]  # at j-1/2

            # Calculate scalar transport additions
            C_state_here = self.State_Sed[pj, pi]  # noqa: N806

            C_state_left = self.State_Sed[pj, pi - 1]  # noqa: N806
            C_state_right = self.State_Sed[pj, pi + 1]  # noqa: N806

            C_state_up = self.State_Sed[pj + 1, pi]  # noqa: N806
            C_state_down = self.State_Sed[pj - 1, pi]  # noqa: N806

            C_state_up_left = self.State_Sed[pj + 1, pi - 1]  # noqa: N806
            C_state_up_right = self.State_Sed[pj + 1, pi + 1]  # noqa: N806

            C_state_down_left = self.State_Sed[pj - 1, pi - 1]  # noqa: N806
            C_state_down_right = self.State_Sed[pj - 1, pi + 1]  # noqa: N806

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

            B = self.Bottom[2, pj, pi]  # noqa: N806
            in_state_here = self.State[pj, pi]
            eta = in_state_here[0]
            hu = in_state_here[1]
            hv = in_state_here[2]
            h = eta - B

            h2 = h * h
            divide_by_h = 2.0 * h / tm.sqrt(h2 + max(h2, self.epsilon))

            f = self.friction / 2.0
            if self.isManning:
                f = 9.81 * pow(self.friction, 2.0) * pow(abs(divide_by_h), 1.0 / 3.0)

            u = hu * divide_by_h
            v = hv * divide_by_h

            local_speed = tm.sqrt(u**2 + v**2)

            shear_velocity = tm.sqrt(f) * local_speed
            shields = shear_velocity * shear_velocity * self.sedC1_shields

            erosion = 0.0

            if shields >= self.sedC1_criticalshields:
                erosion = (
                    self.sedC1_erosion
                    * (shields - self.sedC1_criticalshields)
                    * local_speed
                    * divide_by_h
                )

            Cmin = max(1.0e-6, C_state_here.x)  # only for C1 right now  # noqa: N806

            deposition = (
                min(2.0, (1.0 - self.sedC1_n) / Cmin)
                * C_state_here.x
                * self.sedC1_fallvel
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
            oldies = self.oldGradients_Sed[pj, pi]
            oldOldies = self.oldOldGradients_Sed[pj, pi]  # noqa: N806

            Out = ti.Vector(  # noqa: N806
                [0.0, 0.0, 0.0, 0.0], float
            )  # w , hu, hv ,hc

            if self.timeScheme == 0:  # if timeScheme is Euler do:
                Out = C_state_here + self.dt * d_by_dt  # noqa: N806

            elif self.pred_or_corrector == 1:  # if time scheme is predictor
                Out = C_state_here + self.dt / 12.0 * (  # noqa: N806
                    23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies
                )

            elif (
                self.pred_or_corrector == 2  # noqa: PLR2004
            ):  # if time scheme is corrector
                predicted = self.predictedGradients_Sed[pj, pi]
                # corrector step
                Out = C_state_here + self.dt / 24.0 * (  # noqa: N806
                    9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + 1.0 * oldOldies
                )

            self.current_state_Sed[j, i] = Out
            self.predictedGradients_Sed[j, i] = d_by_dt

    @ti.kernel
    def Pass3Bous(self):  # noqa: C901, N802, D102, PLR0915
        # PASS 3 - Calculus of fluxes
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            pi = i
            pj = j
            if pi >= self.nx - 2 or pj >= self.ny - 2 or pi <= 1 or pj <= 1:
                self.NewState[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.dU_by_dt[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.F_G_star[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.current_stateUVstar[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                continue

            # Load values from txBottom using index
            B_here = self.Bottom[2, pj, pi]  # noqa: N806

            # Load values from txState using index
            in_state_here = self.State[
                pj, pi
            ]  # w, hu and hv (cell avgs, evaluated here)
            in_state_here_UV = self.StateUVstar[pj, pi]  # noqa: N806

            # Calculate local h,w
            B_south = self.Bottom[2, pj - 1, pi]  # noqa: N806
            B_north = self.Bottom[2, pj + 1, pi]  # noqa: N806
            B_west = self.Bottom[2, pj, pi - 1]  # noqa: N806
            B_east = self.Bottom[2, pj, pi + 1]  # noqa: N806

            eta_here = in_state_here.x

            eta_west = self.State[pj, pi - 1].x
            eta_east = self.State[pj, pi + 1].x
            eta_south = self.State[pj - 1, pi].x
            eta_north = self.State[pj + 1, pi].x

            h_here = in_state_here.x - B_here  # h = w - B

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
                    self.NewState[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
                    self.dU_by_dt[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0], float)

                    self.F_G_star[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
                    self.current_stateUVstar[j, i] = ti.Vector(
                        [0.0, 0.0, 0.0, 0.0], float
                    )
                    continue

            h_min = ti.Vector([0.0, 0.0, 0.0, 0.0], float)
            h_min.x = min(h_here, h_north)
            h_min.y = min(h_here, h_east)
            h_min.z = min(h_here, h_south)
            h_min.w = min(h_here, h_west)

            near_dry = self.Bottom[3, pj, pi]

            # Load values from txXFlux and txYFlux using idx
            xflux_here = self.XFlux[pj, pi]  # at i+1/2
            xflux_west = self.XFlux[pj, pi - 1]  # at i-1/2
            yflux_here = self.YFlux[pj, pi]  # at j+1/2
            yflux_south = self.YFlux[pj - 1, pi]  # at j-1/2

            detadx = 0.5 * (eta_east - eta_west) * self.one_over_dx
            detady = 0.5 * (eta_north - eta_south) * self.one_over_dy

            # previous derivatives
            oldies = self.oldGradients[pj, pi]
            oldOldies = self.oldOldGradients[pj, pi]  # noqa: N806

            F_star = 0.0  # noqa: N806
            G_star = 0.0  # noqa: N806
            Psi1x = 0.0  # noqa: N806
            Psi2x = 0.0  # noqa: N806
            Psi1y = 0.0  # noqa: N806
            Psi2y = 0.0  # noqa: N806

            d_here = -B_here

            # OPTIMIZ: only proceed if not near an initially dry cell
            if near_dry > 0.0:
                d2_here = d_here * d_here
                d3_here = d2_here * d_here

                in_state_left = self.State[pj, pi - 1].xyz
                in_state_right = self.State[pj, pi + 1].xyz
                in_state_up = self.State[pj + 1, pi].xyz
                in_state_down = self.State[pj - 1, pi].xyz
                in_state_up_left = self.State[pj + 1, pi - 1].xyz
                in_state_up_right = self.State[pj + 1, pi + 1].xyz
                in_state_down_left = self.State[pj - 1, pi - 1].xyz
                in_state_down_right = self.State[pj - 1, pi + 1].xyz

                F_G_star_oldOldies = self.F_G_star_oldOldGradients[pj, pi].xyz  # noqa: N806

                # Calculate "d" stencil
                d_left = -B_west
                d_right = -B_east
                d_down = -B_south
                d_up = -B_north

                d_left_left = max(0.0, -self.Bottom[2, pj, pi - 2])
                d_right_right = max(0.0, -self.Bottom[2, pj, pi + 2])
                d_down_down = max(0.0, -self.Bottom[2, pj - 2, pi])
                d_up_up = max(0.0, -self.Bottom[2, pj + 2, pi])

                # Calculate "eta" stencil
                eta_here = in_state_here.x
                eta_left = in_state_left.x
                eta_right = in_state_right.x
                eta_down = in_state_down.x
                eta_up = in_state_up.x

                eta_left_left = self.State[pj, pi - 2].x
                eta_right_right = self.State[pj, pi + 2].x
                eta_down_down = self.State[pj - 2, pi].x
                eta_up_up = self.State[pj + 2, pi].x

                eta_up_left = in_state_up_left.x
                eta_up_right = in_state_up_right.x
                eta_down_left = in_state_down_left.x
                eta_down_right = in_state_down_right.x

                # replace with 4th order when dispersion is included
                detadx = (
                    1.0
                    / 12.0
                    * (
                        eta_left_left
                        - 8.0 * eta_left
                        + 8.0 * eta_right
                        + eta_right_right
                    )
                    * self.one_over_dx
                )
                detady = (
                    1.0
                    / 12.0
                    * (eta_down_down - 8.0 * eta_down + 8.0 * eta_up + eta_up_up)
                    * self.one_over_dy
                )

                v_up = in_state_up.z
                v_down = in_state_down.z
                v_right = in_state_right.z
                v_left = in_state_left.z
                v_up_right = in_state_up_right.z
                v_down_right = in_state_down_right.z
                v_up_left = in_state_up_left.z
                v_down_left = in_state_down_left.z

                u_up = in_state_up.y
                u_down = in_state_down.y
                u_right = in_state_right.y
                u_left = in_state_left.y
                u_up_right = in_state_up_right.y
                u_down_right = in_state_down_right.y
                u_up_left = in_state_up_left.y
                u_down_left = in_state_down_left.y

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

            friction_here = max(self.friction, self.BottomFriction[pj, pi].x)
            friction_ = FrictionCalc(
                in_state_here.y,
                in_state_here.z,
                h_here,
                self.maxdepth,
                self.delta,
                self.epsilon,
                self.isManning,
                self.g,
                friction_here,
            )

            # Pressure stencil calculations
            P_left = self.ShipPressure[pj, pi - 1].x  # noqa: N806
            P_right = self.ShipPressure[pj, pi + 1].x  # noqa: N806
            P_down = self.ShipPressure[pj - 1, pi].x  # noqa: N806
            P_up = self.ShipPressure[pj + 1, pi].x  # noqa: N806

            press_x = -0.5 * h_here * self.g_over_dx * (P_right - P_left)
            press_y = -0.5 * h_here * self.g_over_dy * (P_up - P_down)

            # Calculate scalar transport additions
            C_state_here = self.State[pj, pi].w  # noqa: N806

            C_state_left = self.State[pj, pi - 1].w  # noqa: N806
            C_state_right = self.State[pj, pi + 1].w  # noqa: N806
            C_state_up = self.State[pj + 1, pi].w  # noqa: N806
            C_state_down = self.State[pj - 1, pi].w  # noqa: N806
            C_state_up_left = self.State[pj + 1, pi - 1].w  # noqa: N806
            C_state_up_right = self.State[pj + 1, pi + 1].w  # noqa: N806
            C_state_down_left = self.State[pj - 1, pi - 1].w  # noqa: N806
            C_state_down_right = self.State[pj - 1, pi + 1].w  # noqa: N806

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

            if self.useBreakingModel == 1:
                breaking_B = self.Breaking[  # noqa: N806
                    pj, pi
                ].z  # breaking front parameter, non-breaking [0 - 1] breaking

                nu_flux_here = self.DissipationFlux[pj, pi]  # noqa: F841
                nu_flux_right = self.DissipationFlux[pj, pi + 1]
                nu_flux_left = self.DissipationFlux[pj, pi - 1]
                nu_flux_up = self.DissipationFlux[pj + 1, pi]
                nu_flux_down = self.DissipationFlux[pj - 1, pi]

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

            source_term = ti.Vector(
                [
                    overflow_dry,
                    -self.g * h_here * detadx
                    - in_state_here.y * friction_
                    + breaking_x
                    + (Psi1x + Psi2x)
                    + press_x,
                    -self.g * h_here * detady
                    - in_state_here.z * friction_
                    + breaking_y
                    + (Psi1y + Psi2y)
                    + press_y,
                    hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + c_dissipation,
                ]
            )

            d_by_dt = (
                (xflux_west - xflux_here) * self.one_over_dx
                + (yflux_south - yflux_here) * self.one_over_dy
                + source_term
            )

            newState = ti.Vector([0.0, 0.0, 0.0, 0.0], float)  # noqa: N806
            F_G_here = ti.Vector([0.0, F_star, G_star, 0.0], float)  # noqa: N806

            if self.timeScheme == 0:
                newState = in_state_here_UV + self.dt * d_by_dt  # noqa: N806

            elif self.pred_or_corrector == 1:
                newState = in_state_here_UV + self.dt / 12.0 * (  # noqa: N806
                    23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies
                )

            elif self.pred_or_corrector == 2:  # noqa: PLR2004
                predicted = self.predictedGradients[pj, pi]
                newState = in_state_here_UV + self.dt / 24.0 * (  # noqa: N806
                    9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + 1.0 * oldOldies
                )

            if self.showBreaking == 1:
                # add breaking source
                newState.a = max(
                    newState.a, breaking_B
                )  # use the B value from Kennedy et al as a foam intensity
            if self.showBreaking == 2:  # noqa: PLR2004
                contaminent_source = self.ContSource[pj, pi].r
                newState.a = min(1.0, newState.a + contaminent_source)

            # clear concentration if set
            if self.clearConc == 1:
                newState.a = 0.0

            self.NewState[pj, pi] = newState
            self.dU_by_dt[pj, pi] = d_by_dt
            self.F_G_star[pj, pi] = F_G_here
            self.current_stateUVstar[pj, pi] = newState

    @ti.kernel
    def Pass_Breaking(self, time: ti.f32):  # noqa: C901, N802, D102
        # PASS Breaking -
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            pi = i
            pj = j
            xflux_here = self.XFlux[pj, pi].x
            xflux_west = self.XFlux[pj, pi - 1].x

            yflux_here = self.YFlux[pj, pi].x
            yflux_south = self.YFlux[pj - 1, pi].x

            P_south = self.State[pj - 1, pi].y  # noqa: N806
            P_here = self.State[pj, pi].y  # noqa: N806
            P_north = self.State[pj + 1, pi].y  # noqa: N806

            Q_west = self.State[pj, pi - 1].z  # noqa: N806
            Q_here = self.State[pj, pi].z  # noqa: N806
            Q_east = self.State[pj, pi + 1].z  # noqa: N806

            detadt = self.predictedGradients[pj, pi].x

            # Look the dominant direction of flow, and look at the three cells on that 3*3 cube
            t_here = self.Breaking[pj, pi].x
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0

            if ti.abs(P_here) > ti.abs(Q_here):
                if P_here > 0.0:
                    t1 = self.Breaking[pj, pi - 1].x
                    t2 = self.Breaking[pj + 1, pi - 1].x
                    t3 = self.Breaking[pj - 1, pi - 1].x
                else:
                    t1 = self.Breaking[pj, pi + 1].x
                    t2 = self.Breaking[pj + 1, pi + 1].x
                    t3 = self.Breaking[pj - 1, pi + 1].x
            elif Q_here > 0.0:
                t1 = self.Breaking[pj - 1, pi].x
                t2 = self.Breaking[pj - 1, pi + 1].x
                t3 = self.Breaking[pj - 1, pi - 1].x
            else:
                t1 = self.Breaking[pj + 1, pi].x
                t2 = self.Breaking[pj + 1, pi + 1].x
                t3 = self.Breaking[pj + 1, pi - 1].x
            t_here = ti.max(t_here, ti.max(t1, ti.max(t2, t3)))

            dPdx = (xflux_here - xflux_west) * self.one_over_dx  # noqa: N806
            dPdy = 0.5 * (P_north - P_south) * self.one_over_dy  # noqa: N806

            dQdx = 0.5 * (Q_east - Q_west) * self.one_over_dx  # noqa: N806
            dQdy = (yflux_here - yflux_south) * self.one_over_dy  # noqa: N806

            B_here = self.Bottom[pj, pi].z  # noqa: N806
            eta_here = self.State[pj, pi].x
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
                    t_here = self.time
            else:
                B_Breaking = detadt / dzdt_star - 1.0  # noqa: N806
                if t_here <= self.dt:
                    t_here = self.total_time

            nu_breaking = min(
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

            nu_flux = ti.Vector([nu_dPdx, nu_dPdy, nu_dQdx, nu_dQdy])
            Bvalues = ti.Vector([t_here, nu_breaking, B_Breaking, nu_Smag])  # noqa: N806

            self.DissipationFlux[j, i] = nu_flux
            self.Breaking[j, i] = Bvalues

    @ti.kernel
    def Pass_Dissipation(self):  # noqa: C901, N802, D102, PLR0915
        for i, j in ti.ndrange(self.nx, self.ny):
            if i >= self.nx - 2 or j >= self.ny - 2 or i <= 1 or j <= 1:
                self.NewState[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])  # txNewState
                self.dU_by_dt[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])  # dU_by_dt
                self.F_G_star[j, i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.current_stateUVstar = ti.Vector(
                    [0.0, 0.0, 0.0, 0.0]
                )  # txNewState StateUVstar  #check

                continue
            B_here = self.Bottom[j, i].z  # noqa: N806
            near_dry = self.Bottom[j, i].w

            in_state_here = self.State[j, i]
            in_state_here_UV = self.StateUVstar[j, i]  # noqa: N806, F841

            # Calculate local h,w
            h_vec = self.H[j, i]  # noqa: F841
            h_here = in_state_here.x - B_here

            # Load values from txXFlux and txYFlux using idx
            xflux_here = self.XFlux[j, i]
            xflux_west = self.XFlux[j, i - 1]

            yflux_here = self.YFlux[j, i]
            yflux_south = self.YFlux[j - 1, i]

            B_south = self.Bottom[j - 1, i].z  # check  # noqa: N806
            B_north = self.Bottom[j + 1, i].z  # noqa: N806
            B_west = self.Bottom[j, i - 1].z  # noqa: N806
            B_east = self.Bottom[j, i + 1].z  # noqa: N806

            eta_here = in_state_here.x
            eta_west = self.State[j, i - 1].x
            eta_east = self.State[j, i + 1].x
            eta_south = self.State[j - 1, i].x
            eta_north = self.State[j + 1, i].x

            h_west = eta_west - B_west
            h_east = eta_east - B_east
            h_north = eta_north - B_north
            h_south = eta_south - B_south

            h_min = ti.Vector([0.0, 0.0, 0.0, 0.0])
            h_min.x = min(h_here, h_north)
            h_min.y = min(h_here, h_east)
            h_min.z = min(h_here, h_south)
            h_min.w = min(h_here, h_west)

            detadx = 0.5 * (eta_east - eta_west) * self.one_over_dx
            detady = 0.5 * (eta_north - eta_south) * self.one_over_dy

            # previous derivatives
            oldies = self.oldGradients[j, i]
            oldOldies = self.oldOldGradients[j, i]  # noqa: N806

            F_star = 0.0  # noqa: N806
            G_star = 0.0  # noqa: N806
            Psi1x = 0.0  # noqa: N806
            Psi2x = 0.0  # noqa: N806
            Psi1y = 0.0  # noqa: N806
            Psi2y = 0.0  # noqa: N806

            d_here = self.seaLevel - B_here

            # in order to proceed only if not near and initially dry cell
            if near_dry > 0.0:
                d2_here = d_here * d_here
                d3_here = d2_here * d_here

                in_state_right = self.txState[j, i + 1].xyz
                in_state_left = self.State[j, i - 1].xyz
                in_state_up = self.State[j + 1, i].xyz
                in_state_down = self.State[j - 1, i].xyz

                in_state_up_left = self.State[j + 1, i - 1].xyz
                in_state_up_right = self.State[j + 1, i + 1].xyz
                in_state_down_left = self.State[j - 1, i - 1].xyz
                in_state_down_right = self.State[j - 1, i + 1].xyz

                F_G_star_oldOldies = self.F_G_star_oldOldGradients[j, i].xyz  # noqa: N806

                # Calculate d stencil
                d_left = self.seaLevel - B_west
                d_right = self.seaLevel - B_east
                d_down = self.seaLevel - B_south
                d_up = self.seaLevel - B_north

                d_left_left = max(
                    0.0, self.seaLevel - self.Bottom[j, i - 2].z
                )  # check
                d_right_right = max(0.0, self.seaLevel - self.Bottom[j, i + 2].z)
                d_down_down = max(0.0, self.seaLevel - self.Bottom[j - 2, i].z)
                d_up_up = max(0.0, self.seaLevel - self.Bottom[j + 2, i].z)

                # Calculate eta stencil
                eta_here = in_state_here.x - self.seaLevel
                eta_left = in_state_left.x - self.seaLevel
                eta_right = in_state_right.x - self.seaLevel
                eta_down = in_state_down.x - self.seaLevel
                eta_up = in_state_up.x - self.seaLevel

                eta_left_left = self.State[j, i - 2].x - self.seaLevel
                eta_right_right = self.State[j, i + 2].x - self.seaLevel
                eta_down_down = self.State[j - 2, i].x - self.seaLevel
                eta_up_up = self.State[j + 2, i].x - self.seaLevel

                eta_up_left = in_state_up_left.x - self.seaLevel
                eta_up_right = in_state_up_right.x - self.seaLevel
                eta_down_left = in_state_down_left.x - self.seaLevel
                eta_down_right = in_state_down_right.x - self.seaLevel

                # replace with 4th order when dispersion is included
                detadx = (
                    1.0
                    / 12.0
                    * (
                        eta_left_left
                        - 8.0 * eta_left
                        + 8.0 * eta_right
                        + eta_right_right
                    )
                    * self.one_over_dx
                )
                detady = (
                    1.0
                    / 12.0
                    * (eta_down_down - 8.0 * eta_down + 8.0 * eta_up + eta_up_up)
                    * self.one_over_dy
                )

                v_up = in_state_up.z
                v_down = in_state_down.z
                v_right = in_state_right.z
                v_left = in_state_left.z
                v_up_right = in_state_up_right.z
                v_down_right = in_state_down_right.z
                v_up_left = in_state_up_left.z
                v_down_left = in_state_down_left.z

                u_up = in_state_up.y
                u_down = in_state_down.y
                u_right = in_state_right.y
                u_left = in_state_left.y
                u_up_right = in_state_up_right.y
                u_down_right = in_state_down_right.y
                u_up_left = in_state_up_left.y
                u_down_left = in_state_down_left.y

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

            # calc friction
            h = h_here
            hu = in_state_here.y
            hv = in_state_here.z

            h2 = h * h
            divide_by_h = 2.0 * h / ti.sqrt(h2 + max(h2, self.epsilon))

            f = 0.0
            if self.isManning == True:  # noqa: E712
                f = (
                    self.g
                    * pow(self.friction, 2.0)
                    * ti.pow(ti.abs(divide_by_h), 1.0 / 3.0)
                )
            else:
                f = self.friction / 2.0
            f = f * ti.sqrt(hu * hu + hv * hv) * divide_by_h * divide_by_h

            # calc subgrid dissipation
            cm = 0.2  # Smag coef  # noqa: F841
            # 2.*strain rate tensor = 4*u_x^2 + 4*vy^2 +4*ux*vy + (uy + vx)^2
            u = self.U[j, i]
            v = self.V[j, i]
            u_x = (u.y - u.w) * self.one_over_dx  # noqa: F841
            v_x = (v.y - v.w) * self.one_over_dx  # noqa: F841
            u_y = (u.x - u.z) * self.one_over_dy  # noqa: F841
            v_y = (v.x - v.z) * self.one_over_dy  # noqa: F841

            # nuH_Smag = cm*self.dx*self.dy*ti.sqrt(4*u_x*u_x + 4*vy*vy +4*ux*vy + (uy + vx)*(uy + vx))
            ## assume gradients in nu are not important right now
            # temp1 = nuH_Smag * u_x;
            # temp2 = nuH_Smag * u_y;
            # temp3 = nuH_Smag * (u_x + v_x);

            # temp2 = (vtj(i,j)*ujy(i,j)-vtj(i,j-1)*ujy(i,j-1))/dy
            # temp3 = vtv(i,j)*(uix(i,j)+viy(i,j) - uix(i-1,j)-viy(i-1,j))/dx
            # temp4 = (vtj(i,j)*tmpj1(i,j)-vtj(i,j-1)*tmpj1(i,j-1))/dy
            # F(i,j) = F(i,j) + (2*temp1+temp2-1*temp3+temp4)*H(i,j)
            # too many gradients.  Probably want to make this a velocity derivative and eddy viscosity calc pass.

            # Pressure stencil calculations
            P_left = self.ShipPressure[j, i - 1].x  # noqa: N806
            P_right = self.ShipPressure[j, i + 1].x  # noqa: N806
            P_down = self.ShipPressure[j - 1, i].x  # noqa: N806
            P_up = self.ShipPressure[j + 1, i].x  # noqa: N806

            press_x = -0.5 * h_here * self.g_over_dx * (P_right - P_left)
            press_y = -0.5 * h_here * self.g_over_dy * (P_up - P_down)

            # Calculate scalar transport additions
            C_state_here = self.State[j, i].w  # noqa: N806
            C_state_right = self.State[j, i + 1].w  # noqa: N806
            C_state_left = self.State[j, i - 1].w  # noqa: N806
            C_state_up = self.State[j + 1, i].w  # noqa: N806
            C_state_down = self.State[j - 1, i].w  # noqa: N806
            C_state_up_left = self.State[j + 1, i - 1].w  # noqa: N806
            C_state_up_right = self.State[j + 1, i + 1].w  # noqa: N806
            C_state_down_left = self.State[j - 1, i - 1].w  # noqa: N806
            C_state_down_right = self.State[j - 1, i + 1].w  # noqa: N806

            Dxx = 1.0  # noqa: N806
            Dxy = 1.0  # noqa: N806
            Dyy = 1.0  # noqa: N806

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

            # fix slope near shoreline
            h_cut = self.delta
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
                    -0.001
                )  # hydraulic conductivity of coarse, unsaturated sand
            # check friction_
            friction_here = max(self.friction, self.BottomFriction[j, i].x)
            friction_ = FrictionCalc(
                in_state_here.y,
                in_state_here.z,
                h_here,
                self.maxdepth,
                self.delta,
                self.epsilon,
                self.isManning,
                self.g,
                friction_here,
            )
            source_term = ti.Vector(
                [
                    overflow_dry,
                    -self.g * h_here * detadx
                    - in_state_here.y * friction_
                    + (Psi1x + Psi2x)
                    + press_x,
                    -self.g * h_here * detady
                    - in_state_here.z * friction_
                    + (Psi1y + Psi2y)
                    + press_y,
                    hc_by_dx_dx + hc_by_dy_dy + 2.0 * hc_by_dx_dy + c_dissipation,
                ]
            )

            d_by_dt = (
                (xflux_west - xflux_here) * self.one_over_dx
                + (yflux_south - yflux_here) * self.one_over_dy
                + source_term
            )
            newState = ti.Vector([0.0, 0.0, 0.0, 0.0])  # noqa: N806
            F_G_here = ti.Vector([0.0, F_star, G_star, 0.0])  # noqa: N806

            if self.timeScheme == 0:
                newState = (  # noqa: N806
                    self.in_state_here_UV + self.dt * d_by_dt
                )  # check
            elif self.pred_or_corrector == 1:
                newState = self.in_state_here_UV + self.dt / 12.0 * (  # noqa: N806
                    23.0 * d_by_dt - 16.0 * oldies + 5.0 * oldOldies
                )
            elif self.pred_or_corrector == 2:  # noqa: PLR2004
                predicted = self.predictedGradients[j, i]
                newState = self.in_state_here_UV + self.dt / 24.0 * (  # noqa: N806
                    9.0 * d_by_dt + 19.0 * predicted - 5.0 * oldies + oldOldies
                )

            # add passive tracer sources
            if (
                ti.max(ti.abs(detadx), ti.abs(detady))
                * ti.sign(detadx * newState.y + detady * newState.z)
                > ti.dissipation_threshold
            ):
                newState.a = 1.0
            contaminent_source = self.ContSource[j, i].r
            newState.a = ti.min(1.0, newState.a + contaminent_source)

            # clear concentration if set
            if self.clearConc == 1:
                newState.a = 0.0

            self.NewState[j, i] = newState
            self.dU_by_dt[j, i] = d_by_dt  # check
            self.F_G_star = F_G_here
            self.current_stateUVstar = newState  # check

    @ti.kernel
    def TriDiag_PCRx(self, p: int, s: int):  # noqa: N802, D102
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            CurrentState = self.NewState[j, i]  # noqa: N806
            # CurrentState = self.current_stateUVstar[j,i]

            # Convert the index wrapping logic
            idx_left = (i - s + self.nx) % self.nx
            idx_right = (i + s + self.nx) % self.nx

            aIn, bIn, cIn, dIn = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInLeft, bInLeft, cInLeft, dInLeft = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInRight, bInRight, cInRight, dInRight = 0.0, 0.0, 0.0, 0.0  # noqa: N806

            if p == 0:
                bIn = self.newcoef_x[j, i][1]  # noqa: N806
                bInLeft = self.newcoef_x[j, idx_left][1]  # noqa: N806
                bInRight = self.newcoef_x[j, idx_right][1]  # noqa: N806

                aIn = self.newcoef_x[j, i][0] / bIn  # noqa: N806
                aInLeft = self.newcoef_x[j, idx_left][0] / bInLeft  # noqa: N806
                aInRight = self.newcoef_x[j, idx_right][0] / bInRight  # noqa: N806

                cIn = self.newcoef_x[j, i][2] / bIn  # noqa: N806
                cInLeft = self.newcoef_x[j, idx_left][2] / bInLeft  # noqa: N806
                cInRight = self.newcoef_x[j, idx_right][2] / bInRight  # noqa: N806

                dIn = self.current_stateUVstar[j, i][1] / bIn  # check  # noqa: N806
                dInLeft = self.current_stateUVstar[j, idx_left][1] / bInLeft  # noqa: N806
                dInRight = self.current_stateUVstar[j, idx_right][1] / bInRight  # noqa: N806

            else:
                aIn = self.newcoef_x[j, i][0]  # noqa: N806
                aInLeft = self.newcoef_x[j, idx_left][0]  # noqa: N806
                aInRight = self.newcoef_x[j, idx_right][0]  # noqa: N806

                cIn = self.newcoef_x[j, i][2]  # noqa: N806
                cInLeft = self.newcoef_x[j, idx_left][2]  # noqa: N806
                cInRight = self.newcoef_x[j, idx_right][2]  # noqa: N806

                dIn = self.newcoef_x[j, i][3]  # noqa: N806
                dInLeft = self.newcoef_x[j, idx_left][3]  # noqa: N806
                dInRight = self.newcoef_x[j, idx_right][3]  # noqa: N806

            r = 1.0 / (1.0 - aIn * cInLeft - cIn * aInRight)
            aOut = -r * aIn * aInLeft  # noqa: N806
            cOut = -r * cIn * cInRight  # noqa: N806
            dOutput = r * (dIn - aIn * dInLeft - cIn * dInRight)  # noqa: N806

            self.txtemp_PCRx[j, i] = ti.Vector([aOut, 1.0, cOut, dOutput])
            self.txtemp2_PCRx[j, i] = ti.Vector(
                [CurrentState[0], dOutput, CurrentState[2], CurrentState[3]]
            )

    @ti.kernel
    def TriDiag_PCRy(self, p: ti.i32, s: ti.i32):  # noqa: N802, D102
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            CurrentState = self.NewState[j, i]  # noqa: N806
            # CurrentState = self.current_stateUVstar[j,i]

            idx_left = (j - s + self.ny) % self.ny
            idx_right = (j + s + self.ny) % self.ny

            aIn, bIn, cIn, dIn = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInLeft, bInLeft, cInLeft, dInLeft = 0.0, 0.0, 0.0, 0.0  # noqa: N806
            aInRight, bInRight, cInRight, dInRight = 0.0, 0.0, 0.0, 0.0  # noqa: N806

            if p == 0:
                bIn = self.newcoef_y[j, i][1]  # noqa: N806
                bInLeft = self.newcoef_y[idx_left, i][1]  # noqa: N806
                bInRight = self.newcoef_y[idx_right, i][1]  # noqa: N806

                aIn = self.newcoef_y[j, i][0] / bIn  # noqa: N806
                aInLeft = self.newcoef_y[idx_left, i][0] / bInLeft  # noqa: N806
                aInRight = self.newcoef_y[idx_right, i][0] / bInRight  # noqa: N806

                cIn = self.newcoef_y[j, i][2] / bIn  # noqa: N806
                cInLeft = self.newcoef_y[idx_left, i][2] / bInLeft  # noqa: N806
                cInRight = self.newcoef_y[idx_right, i][2] / bInRight  # noqa: N806

                dIn = self.current_stateUVstar[j, i][1] / bIn  # check  # noqa: N806
                dInLeft = self.current_stateUVstar[idx_left, i][1] / bInLeft  # noqa: N806
                dInRight = self.current_stateUVstar[idx_right, i][1] / bInRight  # noqa: N806

            else:
                aIn = self.newcoef_y[j, i][0]  # noqa: N806
                aInLeft = self.newcoef_y[idx_left, i][0]  # noqa: N806
                aInRight = self.newcoef_y[idx_right, i][0]  # noqa: N806

                cIn = self.newcoef_y[j, i][2]  # noqa: N806
                cInLeft = self.newcoef_y[idx_left, i][2]  # noqa: N806
                cInRight = self.newcoef_y[idx_right, i][2]  # noqa: N806

                dIn = self.newcoef_y[j, i][3]  # noqa: N806
                dInLeft = self.newcoef_y[idx_left, i][3]  # noqa: N806
                dInRight = self.newcoef_y[idx_right, i][3]  # noqa: N806

            r = 1.0 / (1.0 - aIn * cInLeft - cIn * aInRight)
            aOut = -r * aIn * aInLeft  # noqa: N806
            cOut = -r * cIn * cInRight  # noqa: N806
            dOutput = r * (dIn - aIn * dInLeft - cIn * dInRight)  # noqa: N806

            self.txtemp_PCRy[j, i] = ti.Vector([aOut, 1.0, cOut, dOutput])
            self.txtemp2_PCRy[j, i] = ti.Vector(
                [CurrentState[0], CurrentState[1], dOutput, CurrentState[3]]
            )

    def Run_Tridiag_solver(self):  # noqa: N802, D102
        if self.model == 'SWE':
            self.copy_states(src=self.current_stateUVstar, dst=self.NewState)
        else:
            Px = int(ti.ceil(ti.log(self.nx) / ti.log(2)))  # noqa: N806
            Py = int(ti.ceil(ti.log(self.ny) / ti.log(2)))  # noqa: N806

            self.copy_states(src=self.coefMatx, dst=self.newcoef_x)
            for p in range(Px):
                s = 1 << p
                self.TriDiag_PCRx(p, s)
                self.copy_states(src=self.txtemp_PCRx, dst=self.newcoef_x)
            self.copy_states(src=self.txtemp2_PCRx, dst=self.NewState)  # check

            self.copy_states(src=self.coefMaty, dst=self.newcoef_y)
            for p in range(Py):
                s = 1 << p
                self.TriDiag_PCRy(p, s)
                self.copy_states(src=self.txtemp_PCRy, dst=self.newcoef_y)
            self.copy_states(src=self.txtemp2_PCRy, dst=self.NewState)

    @ti.kernel
    def Update_Tridiag_coef(self):  # noqa: N802, D102
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            idx = ti.Vector([j, i])  # noqa: F841
            d_here = -self.txBottom[j, i][2]
            near_dry = d_here  # textureLoad(txBottom, idx, 0).w

            a, b, c = 0.0, 1.0, 0.0
            # X-coefs
            coefx = ti.Vector([0.0, 0.0, 0.0, 0.0])
            if i <= 2 or i >= self.nx - 3 or near_dry < 0.0:  # noqa: PLR2004
                a = 0.0
                b = 1.0
                c = 0.0
            else:
                d_west = -self.Bottom[j, i - 1][2]
                d_east = -self.Bottom[j, i + 1][2]

                # Calculate the first derivative of the depth
                d_dx = (d_east - d_west) / (2.0 * self.dx)

                # Calculate coefficients based on the depth and its derivative
                a = d_here * d_dx / (6.0 * self.dx) - (
                    self.Bcoef + 1.0 / 3.0
                ) * d_here * d_here / (self.dx * self.dx)
                b = 1.0 + 2.0 * (self.Bcoef + 1.0 / 3.0) * d_here * d_here / (
                    self.dx * self.dx
                )
                c = -d_here * d_dx / (6.0 * self.dx) - (
                    self.Bcoef + 1.0 / 3.0
                ) * d_here * d_here / (self.dx * self.dx)

            coefx = ti.Vector([a, b, c, 0.0])

            # Y-coefs
            coefy = ti.Vector([0.0, 0.0, 0.0, 0.0])
            if j <= 2 or j >= self.ny - 3 or near_dry < 0.0:  # noqa: PLR2004
                a = 0.0
                b = 1.0
                c = 0.0
            else:
                d_south = -self.Bottom[j - 1, i][2]
                d_north = -self.Bottom[j + 1, i][2]

                # Calculate the first derivative of the depth
                d_dy = (d_north - d_south) / (2.0 * self.dy)

                # Calculate coefficients based on the depth and its derivative
                a = d_here * d_dy / (6.0 * self.dy) - (
                    self.Bcoef + 1.0 / 3.0
                ) * d_here * d_here / (self.dy * self.dy)
                b = 1.0 + 2.0 * (self.Bcoef + 1.0 / 3.0) * d_here * d_here / (
                    self.dy * self.dy
                )
                c = -d_here * d_dy / (6.0 * self.dy) - (
                    self.Bcoef + 1.0 / 3.0
                ) * d_here * d_here / (self.dy * self.dy)

            coefy = ti.Vector([a, b, c, 0.0])

            self.coefMatx[j, i] = coefx
            self.coefMaty[j, i] = coefy

    @ti.kernel
    def Update_neardry(self):  # noqa: N802, D102
        lengthCheck = 3  # check within three points  # noqa: N806
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            B_here = self.txBottom[j, i]  # noqa: N806
            # Update neardry
            B_here[3] = 99.0
            for yy in range(j - lengthCheck, j + lengthCheck + 1):
                for xx in range(i - lengthCheck, i + lengthCheck + 1):
                    xC = min(self.nx - 1, max(0, xx))  # noqa: N806
                    yC = min(self.ny - 1, max(0, yy))  # noqa: N806

                    bathy_C = self.Bottom[yC, xC][2]  # noqa: N806
                    if bathy_C >= 0.0:
                        B_here[3] = -99.0

            B_south = self.Bottom[j - 1, i][2]  # noqa: N806
            B_north = self.Bottom[j + 1, i][2]  # noqa: N806
            B_west = self.Bottom[j, i - 1][2]  # noqa: N806
            B_east = self.Bottom[j, i + 1][2]  # noqa: N806

            if B_here[2] > 0.0:
                if B_south < 0.0 and B_north < 0.0 and B_west < 0.0 and B_east < 0.0:
                    B_here[2] = 0.0
                    B_here[0] = B_north / 2.0
                    B_here[1] = B_east / 2.0

            self.temp_bottom[j, i] = B_here


if __name__ == '__main__':
    baty = Topodata('dummy.xyz', 'xyz')  # noqa: F821
    d = Domain(1.0, 100.0, 1.0, 50.0, 200, 100)  # noqa: F821
