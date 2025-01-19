import os  # noqa: INP001, D100
import time

import imageio  # For creating GIFs
from celeris.solver import *  # noqa: F403
from celeris.utils import *  # noqa: F403
from taichi import tools  # For saving images

# Done at top of translation unit for now, clean-up later - JB
base_frame_dir = './plots/'  # ! Assumes we are in CelerisAi/ directory
os.makedirs(  # noqa: PTH103
    base_frame_dir, exist_ok=True
)  # Ensure base_frame_dir exists
frame_paths = []  # List of frame paths for later use, i.e. making gifs/mp4s


@ti.data_oriented  # noqa: F405
class Evolve:  # noqa: D101
    def __init__(
        self,
        domain=None,  # noqa: ARG002
        boundary_conditions=None,  # noqa: ARG002
        solver=None,
        maxsteps=1000,
        outdir=None,  # noqa: ARG002
        saveimg=False,  # noqa: FBT002
        vmin=-1.5,
        vmax=1.5,
    ):
        self.solver = solver
        self.maxsteps = maxsteps
        self.dt = self.solver.dt
        self.timeScheme = self.solver.timeScheme
        self.saveimg = saveimg
        self.vmin = vmin
        self.vmax = vmax
        # To visualization
        self.image = ti.Vector.field(  # noqa: F405
            3,
            dtype=ti.f32,  # noqa: F405
            shape=(self.solver.nx, self.solver.ny),
        )
        self.ocean = ti.Vector.field(3, dtype=ti.f16, shape=16)  # noqa: F405
        self.colormap_ocean = 'Blues_r'

    def Evolve_0(self):  # noqa: N802, D102
        self.solver.fill_bottom_field()
        self.solver.InitStates()
        self.solver.tridiag_coeffs_X()
        self.solver.tridiag_coeffs_Y()
        print('Model: ', self.solver.model)  # noqa: T201
        print(  # noqa: T201
            'Numerical Scheme: ',
            self.solver.timeScheme,
            ' dx:',
            self.solver.dx,
            ' dy:',
            self.solver.dy,
        )
        print(  # noqa: T201
            'Breaking Model: ',
            self.solver.useBreakingModel,
            ' Sediment Transport: ',
            self.solver.useSedTransModel,
        )
        print('Time delta: ', self.dt)  # noqa: T201

    def Evolve_Steps(self, step=0):  # noqa: C901, N802, D102
        i = step

        self.solver.Pass1()

        if self.solver.useSedTransModel:
            self.solver.Pass1_SedTrans()

        self.solver.Pass2()

        if self.solver.useBreakingModel:
            self.solver.Pass_Breaking(time=self.dt * i - self.dt)

        if self.solver.model == 'SWE':
            self.solver.Pass3(pred_or_corrector=1)  # Predictor Step in 'SWE'
        else:
            self.solver.Pass3Bous(pred_or_corrector=1)  # Predicto Step in 'BOUSS'

        self.solver.copy_states(
            src=self.solver.dU_by_dt, dst=self.solver.predictedGradients
        )

        if self.solver.useSedTransModel:
            self.solver.Pass3_SedTrans(pred_or_corrector=1)
            self.solver.copy_states(
                src=self.solver.dU_by_dt_Sed, dst=self.solver.predictedGradients_Sed
            )

        self.solver.BoundaryPass(
            time=self.dt * i, txState=self.solver.current_stateUVstar
        )

        self.solver.Run_Tridiag_solver()  # Run TridiagSolver for Bouss and copy current_stateUVstar to NewState

        if self.solver.model != 'SWE':
            self.solver.BoundaryPass(time=self.dt * i, txState=self.solver.NewState)

        if self.solver.model == 'Bouss':
            self.solver.copy_states(
                src=self.solver.F_G_star_oldGradients,
                dst=self.solver.F_G_star_oldOldGradients,
            )
            self.solver.copy_states(
                src=self.solver.predictedF_G_star,
                dst=self.solver.F_G_star_oldGradients,
            )

        if self.solver.timeScheme == 2:  # noqa: PLR2004
            self.solver.copy_states(src=self.solver.NewState, dst=self.solver.State)

            if self.solver.useSedTransModel:
                self.solver.copy_states(
                    src=self.solver.NewState_Sed, dst=self.solver.State_Sed
                )

            self.solver.Pass1()

            if self.solver.useSedTransModel:
                self.solver.Pass1_SedTrans()

            self.solver.Pass2()

            if self.solver.useBreakingModel:
                self.solver.Pass_Breaking(time=self.dt * i)

            if self.solver.model == 'SWE':
                self.solver.Pass3(pred_or_corrector=2)
            else:
                self.solver.Pass3Bous(pred_or_corrector=2)

            if self.solver.useSedTransModel:
                self.solver.Pass3_SedTrans(pred_or_corrector=2)

            self.solver.BoundaryPass(
                time=self.dt * i, txState=self.solver.current_stateUVstar
            )

            self.solver.Run_Tridiag_solver()

            if self.solver.model != 'SWE':
                self.solver.BoundaryPass(
                    time=self.dt * i, txState=self.solver.NewState
                )

            if self.solver.useSedTransModel:
                self.solver.Update_Bottom()
                if self.solver.model == 'Bouss':
                    self.solver.fill_bottom_field()
                    self.solver.tridiag_coeffs_X()
                    self.solver.tridiag_coeffs_Y()

        # shift gradients
        self.solver.copy_states(
            src=self.solver.oldGradients, dst=self.solver.oldOldGradients
        )
        self.solver.copy_states(
            src=self.solver.predictedGradients, dst=self.solver.oldGradients
        )

        # Copy future states
        self.solver.copy_states(src=self.solver.NewState, dst=self.solver.State)
        self.solver.copy_states(
            src=self.solver.current_stateUVstar, dst=self.solver.stateUVstar
        )

        if self.solver.useSedTransModel:
            self.solver.copy_states(
                src=self.solver.oldGradients_Sed, dst=self.solver.oldOldGradients_Sed
            )
            self.solver.copy_states(
                src=self.solver.predictedGradients_Sed,
                dst=self.solver.oldGradients_Sed,
            )
            self.solver.copy_states(
                src=self.solver.NewState_Sed, dst=self.solver.State_Sed
            )

        # To test pressure
        # self.solver.Ship_pressure(px_init=10,py_init=50,steps=int(i))

    def Evolve_Headless(self):  # noqa: N802, D102
        self.Evolve_0()
        start_time = time.time()

        for i in range(self.maxsteps):
            self.Evolve_Steps(i)
            if i == 1:
                start_time = (
                    time.time() - 0.00001
                )  # reset the "start" time as there is overhead before loop starts, and add small shift to prevent float divide by zero

            if i == 1 or (i % 100) == 0:
                compTime = time.time() - start_time  # noqa: N806
                print(  # noqa: T201
                    f'Current Simulation time: {self.dt*i:2.2f}s at step: {i}-- Ratio:{(self.dt*i)/compTime:2.2f}--CompTime:{compTime:2.2f}'
                )
                if self.solver.outdir:
                    state = self.solver.State.to_numpy()
                    np.save(f'{self.outdir}/state_{int(i)}.npy', state)  # noqa: F405

    @ti.func  # noqa: F405
    def brk_color(self, x, y0, y1, x0, x1):  # noqa: D102
        # Interp. to get changes in color
        return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

    @ti.kernel  # noqa: F405
    def paint_new(self):  # noqa: D102
        for i, j in ti.ndrange((0, self.solver.nx), (0, self.solver.ny)):  # noqa: F405
            self.solver.pixel[i, j] = self.brk_color(
                self.solver.Bottom[2, i, j],
                0.75,
                1,
                self.solver.maxtopo,
                -1 * self.solver.maxtopo,
            )
            flow = self.solver.State[i, j][0] - self.solver.Bottom[2, i, j]
            if flow > 0.0001:  # noqa: PLR2004
                # self.solver.pixel[i,j] = self.brk_color(self.solver.State[i,j][0], 0, 0.75,self.vmin,self.vmax)
                self.solver.pixel[i, j] = self.brk_color(
                    flow, 0, 0.75, 0.0001, self.solver.base_depth + 3
                )

    @ti.kernel  # noqa: F405
    def InitColors(self, arr: ti.types.ndarray(dtype=ti.f16, ndim=2)):  # noqa: N802, D102, F405
        for i in self.ocean:
            self.ocean[i].x = arr[i, 0]
            self.ocean[i].y = arr[i, 1]
            self.ocean[i].z = arr[i, 2]

    @ti.kernel  # noqa: F405
    def painting_h(self):  # noqa: D102
        num_colors = self.ocean.shape[0]
        step = 1.0 / num_colors
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):  # noqa: F405
            flow = (
                self.solver.State[i, j][0] - self.solver.Bottom[2, i, j]
            )  # only water
            land = max(0.0, self.solver.Bottom[2, i, j])  # only positive topo
            water_col = flow / self.solver.base_depth  # Water column normalized
            land_elevation = land / self.solver.maxtopo  # Topo normalized
            index = int(water_col / step)  # which color interval we're in
            index = ti.min(  # noqa: F405
                index, num_colors - 2
            )  # clamp index to avoid out-of-bounds
            t = (
                water_col - index * step
            ) / step  # fractional position between the colors
            if flow > 0.0:  # Water area
                self.image[i, j] = (
                    self.ocean[index] * (1 - t) + self.ocean[index + 1] * t
                )
            elif -0.25 < flow < 0:  # noqa: PLR2004
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.8039, 0.7921, 0.7372]
                )  # Wet Sand
            else:
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25]
                )

    @ti.kernel  # noqa: F405
    def painting_eta(self):  # noqa: D102
        num_colors = self.ocean.shape[0]
        step = 1.0 / num_colors
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):  # noqa: F405
            flow = (
                self.solver.State[i, j][0] - self.solver.Bottom[2, i, j]
            )  # only water
            wave = self.solver.State[i, j][0]
            wave = (wave - self.vmin) / (self.vmax - self.vmin)
            land = max(0.0, self.solver.Bottom[2, i, j])  # only positive topo
            land_elevation = land / self.solver.maxtopo  # Topo normalized
            index = int(wave / step)  # which color interval we're in
            index = ti.min(  # noqa: F405
                index, num_colors - 2
            )  # clamp index to avoid out-of-bounds
            t = (
                wave - index * step
            ) / step  # fractional position between the colors
            if flow > 0.0:  # Water area
                self.image[i, j] = (
                    self.ocean[index] * (1 - t) + self.ocean[index + 1] * t
                )
            elif -0.25 < flow < 0:  # noqa: PLR2004
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.8039, 0.7921, 0.7372]
                )  # Wet Sand
            else:
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25]
                )

    @ti.kernel  # noqa: F405
    def painting_vor(self):  # noqa: D102
        num_colors = self.ocean.shape[0]
        step = 1.0 / num_colors
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):  # noqa: F405
            rightIdx = ti.min(i + 1, self.solver.nx - 1)  # noqa: N806, F405
            upIdx = ti.min(j + 1, self.solver.ny - 1)  # noqa: N806, F405

            B = self.solver.Bottom[2, i, j]  # noqa: N806
            B_right = self.solver.Bottom[2, rightIdx, j]  # noqa: N806
            B_up = self.solver.Bottom[2, i, upIdx]  # noqa: N806

            q = self.solver.State[i, j]
            q_up = self.solver.State[i, upIdx]
            q_right = self.solver.State[rightIdx, j]

            h = q.x - B
            h_right = q_right.x - B_right
            h_up = q_up.x - B_up

            v_right = 0.0
            u_up = 0.0
            u = 0.0
            v = 0.0

            if h_right > 0.05:  # noqa: PLR2004
                v_right = q_right.z / h_right
            if h_up > 0.05:  # noqa: PLR2004
                u_up = q_right.y / h_up
            if h > 0.05:  # noqa: PLR2004
                v = q.z / h
                u = q.y / h

            vor = (v_right - v) / self.solver.dx - (u_up - u) / self.solver.dy

            vor = (vor - self.vmin) / (self.vmax - self.vmin)
            land = max(0.0, B)  # only positive topo
            land_elevation = land / self.solver.maxtopo  # Topo normalized
            index = int(vor / step)  # which color interval we're in
            index = ti.min(  # noqa: F405
                index, num_colors - 2
            )  # clamp index to avoid out-of-bounds
            t = (vor - index * step) / step  # fractional position between the colors
            if h > 0.0:  # Water area
                self.image[i, j] = (
                    self.ocean[index] * (1 - t) + self.ocean[index + 1] * t
                )
            elif -0.25 < h < 0:  # noqa: PLR2004
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.8039, 0.7921, 0.7372]
                )  # Wet Sand
            else:
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25]
                )

    @ti.kernel  # noqa: F405
    def paint(self):  # noqa: D102
        for i, j in ti.ndrange((0, self.solver.nx), (0, self.solver.ny)):  # noqa: F405
            self.solver.pixel[i, j] = self.brk_color(
                self.solver.Bottom[2, i, j],
                0.75,
                1,
                self.solver.maxtopo,
                -1 * self.solver.maxtopo,
            )
            flow = self.solver.State[i, j][0] - self.solver.Bottom[2, i, j]
            # Merge water and topo values
            if flow > 0.0001:  # noqa: PLR2004
                self.solver.pixel[i, j] = self.brk_color(
                    flow, 0, 0.75, 0.0001, self.solver.base_depth + 3
                )
                if self.solver.useSedTransModel:
                    sed = self.solver.State_Sed[i, j][0] / flow
                    if sed > 0.0001:  # noqa: PLR2004
                        self.solver.pixel[i, j] = self.brk_color(
                            sed, 0.65, 0.75, 0.1, 5.0
                        )  # Sed

    def Evolve_Display(  # noqa: C901, N802, D102
        self,
        vmin=None,
        vmax=None,
        variable='h',
        cmapWater='Blues_r',  # noqa: N803
        showSediment=False,  # noqa: FBT002, N803
    ):
        if vmin != None:  # noqa: E711
            self.vmin = vmin
        if vmax != None:  # noqa: E711
            self.vmax = vmax

        plotpath = './plots'
        if not os.path.exists(plotpath):  # noqa: PTH110
            os.makedirs(plotpath)  # noqa: PTH103
        i = 0.0
        use_ggui = None
        window = None
        canvas = None
        try:
            window = ti.ui.Window('CelerisAi', (self.solver.nx, self.solver.ny))  # noqa: F405
            canvas = window.get_canvas()
            use_ggui = True
        except:  # noqa: E722
            # TODO : Formal error handling and logging  # noqa: TD002
            print('GGUI not available, reverting to legacy Taichi GUI.')  # noqa: T201
            use_ggui = False
            use_fast_gui = False  # Need ti.Vector.field equiv to self.solver.pixel to use fast_gui
            window = ti.GUI(  # noqa: F405
                'CelerisAi', (self.solver.nx, self.solver.ny), fast_gui=use_fast_gui
            )  # fast_gui - display directly on frame buffer if not drawing shapes or text
            canvas = None
            print('Legacy GUI initialized.')  # noqa: T201
        else:
            print('GGUI initialized without issues.')  # noqa: T201

        # Customized colormap
        if showSediment:
            cmap = celeris_matplotlib(  # noqa: F405
                water=cmapWater,
                sediment='afmhot_r',
                SedTrans=self.solver.useSedTransModel,
            )
        else:
            cmap = celeris_matplotlib(water=cmapWater)  # noqa: F405, F841
        # cmap = celeris_waves()
        # cmap = celeris_matplotlib(water='Blues_r',sediment='afmhot_r', SedTrans=self.useSedTransModel )

        self.Evolve_0()
        # Set colors - using the matplotlib colormapsand convert these into Taichi tensors
        numpy_ocean = ColorsfromMPL(cmapWater)  # noqa: F405
        self.InitColors(numpy_ocean)

        start_time = time.time()

        while window.running:
            # self.paint()
            # self.paint_new()
            # self.paint()
            if variable == 'h':
                self.painting_h()
            if variable == 'eta':
                self.painting_eta()
            if variable == 'vor':
                self.painting_vor()

            if use_ggui:
                # canvas.contour(self.solver.pixel, cmap_name=cmap ) # Same functionality as set cmap-pixel-to np
                # canvas.contour(self.solver.pixel,cmap_name='plasma') # Same functionality as set cmap-pixel-to np
                canvas.set_image(
                    self.image
                )  # using the Taichi tensors to render the image
            else:
                # window.set_image(self.solver.pixel)
                window.set_image(
                    self.image
                )  # using the Taichi tensors to render the image
            self.Evolve_Steps(i)

            if i == 1:
                start_time = (
                    time.time() - 0.00001
                )  # reset the "start" time as there is overhead before loop starts, and add small shift to prevent float divide by zero

            if i == 1 or (i % 100) == 0:
                compTime = time.time() - start_time  # noqa: N806
                print(  # noqa: T201
                    f'Current Simulation time: {self.dt*i:2.2f}s at step: {i}-- Ratio:{(self.dt*i)/compTime:2.2f}--CompTime:{compTime:2.2f}'
                )
                frame = int(i)
                frame_filename = f'frame_{frame}.png'
                frame_path = os.path.join(base_frame_dir, frame_filename)  # noqa: PTH118
                frame_paths.append(frame_path)
                if self.saveimg:
                    if use_ggui:
                        window.save_image(frame_path)
                    else:
                        tools.imwrite(self.solver.pixel.to_numpy(), frame_path)

                # if self.saveimg and not use_ggui:
                #     try:
                #         window.show(frame_path)
                #     except Exception as e:
                #         print(f"Error showing frame: {e},  fallback to tools.imwrite...")
                #         try:
                #             tools.imwrite(self.solver.pixel.to_numpy(), frame_path)
                #             frame_paths.append(frame_path)
                #         except Exception as e:
                #             print(f"Error writing frame with tools.imwrite: {e}")
                #     else:
                #         frame_paths.append(frame_path)
                # elif self.saveimg and use_ggui:
                #     try:
                #         tools.imwrite(self.solver.pixel.to_numpy(), frame_path)
                #         frame_paths.append(frame_path)
                #     except Exception as e:
                #         print(f"Error writing frame with tools.imwrite: {e}")
                # elif not use_ggui and not self.saveimg:
                #     window.show()
                # else:
                #     print("WARNING - No output method selected, frame not saved or displayed...")
                # if not use_ggui:
                #     continue

                # window.show()
                if self.solver.outdir:
                    state = self.solver.State.to_numpy()
                    np.save(f'{self.outdir}/state_{int(i)}.npy', state)  # noqa: F405
            # Show window in the right position (after save image) for GGUI systems
            if i % 5 == 0:
                # Improve the performance.The visualization is done only every 5 timesteps
                window.show()

            if i > self.maxsteps:
                if frame_paths:  # Check if there are frames to create a GIF
                    gif_filename = 'video.gif'
                    gif_path = os.path.join(base_frame_dir, gif_filename)  # noqa: PTH118
                    try:
                        with imageio.get_writer(
                            gif_path, mode='I', duration=0.1
                        ) as writer:
                            for frame_path in frame_paths:
                                image = imageio.imread(frame_path)
                                writer.append_data(image)
                        print(f'GIF created at {gif_path}')  # noqa: T201
                    except Exception as e:  # noqa: BLE001
                        print(f'Error creating GIF: {e}')  # noqa: T201
                break
            i = i + 1
