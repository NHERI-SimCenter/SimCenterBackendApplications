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
        outdir=None,
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
        self.buffer_step = 1
        self.sensor_step = 1
        self.render_step = 5
        self.image_step = 100
        self.state_step = 1000
        self.output_forces = True
        self.output_wave_gauges = True
        self.output_velocimeters = True
        self.output_gif = True
        # To visualization
        self.image = ti.Vector.field(  # noqa: F405
            3,
            dtype=ti.f32,  # noqa: F405
            shape=(self.solver.nx, self.solver.ny),
        )
        self.ocean = ti.Vector.field(3, dtype=ti.f16, shape=16)  # noqa: F405
        self.land = ti.Vector.field(3, dtype=ti.f16, shape=16)  # noqa: F405
        self.colormap_ocean = 'Blues_r'
        self.colormap_land = 'terrain'
        self.outdir = outdir
        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)  # noqa: PTH103
        # To visualize 1D
        self.bottom1D = ti.Vector.field(2, dtype=ti.f32, shape = self.solver.nx)
        self.indexbottom1D = ti.field(dtype=ti.i32, shape = 2*self.solver.nx)
        self.eta1D = ti.Vector.field(2, dtype=ti.f32, shape = self.solver.nx)
        self.x_scale = self.solver.nx * self.solver.dx
        self.y_scale = 2 * self.solver.base_depth

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
        self.solver.update_step()
        self.solver.Pass1(step=int(i))

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

            self.solver.Pass1(step=int(i))

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
                    self.outdir = self.solver.outdir
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
    def InitColors(  # noqa: N802, D102
        self,
        ocean_arr: ti.types.ndarray(dtype=ti.f16, ndim=2),  # noqa: F405
        land_arr: ti.types.ndarray(dtype=ti.f16, ndim=2),  # noqa: F405
    ):
        for i in self.ocean:
            self.ocean[i].x = ocean_arr[i, 0]
            self.ocean[i].y = ocean_arr[i, 1]
            self.ocean[i].z = ocean_arr[i, 2]
        for i in self.land:
            self.land[i].x = land_arr[i, 0]
            self.land[i].y = land_arr[i, 1]
            self.land[i].z = land_arr[i, 2]

    @ti.kernel  # noqa: F405
    def painting_h(self):  # noqa: D102
        num_colors = self.ocean.shape[0]
        step = 1.0 / num_colors

        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):  # noqa: F405
            flow = (
                self.solver.State[i, j][0] - self.solver.Bottom[2, i, j]
            )  # only water
            # land = max(0.0, self.solver.Bottom[2, i, j])  # only positive topo
            land = max(0.0, self.solver.Bottom[2, i, j])  # only positive topo
            # land = self.solver.Bottom[2,i,j]
            # if (self.solver.Bottom[2, :, :].min() < 0) and (land == 0.0):

            # if (self.solver.maxtopo < self.solver.base_depth):
            #     land_datum = self.solver.base_depth - self.solver.maxtopo
            # elif (self.solver.maxtopo >= self.solver.base_depth):
            #     land_datum = self.solver.maxtopo - self.solver.base_depth

            # land -= min(self.solver.maxtopo, self.solver.base_depth)
            approximate_wave_to_depth_breaking_ratio = 0.85  # 0.77
            water_col = ti.pow(flow, 1 / 2) / (  # noqa: F405
                self.solver.base_depth / approximate_wave_to_depth_breaking_ratio
            )  # Water column normalized
            # land_elevation = land / (land_datum)  # Topo normalized
            land_elevation = land
            # land_elevation = (land + self.solver.base_depth) / (self.solver.maxtopo + self.solver.base_depth)  # Topo normalized
            index = int(water_col / step)  # which color interval we're in
            index = ti.min(  # noqa: F405
                index, num_colors - 2
            )  # clamp index to avoid out-of-bounds
            t = float(
                float(water_col) - float(index) * float(step)
            ) / float(step)  # fractional position between the colors
            sand_capillary_height = 0.01
            if flow > 0.0:  # Water area
                self.image[i, j] = (
                    float(self.ocean[index]) * float(1.0 - t) + float(self.ocean[index + 1]) * t
                )
                continue
            if sand_capillary_height < flow < 0.0:
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.8039, 0.7921, 0.7372]
                )  # Wet Sand
                continue

            # print(self.solver.maxtopo)
            # print(self.solver.base_depth)
            land_shift = abs(self.solver.maxtopo) / abs(  # noqa: F841
                self.solver.base_depth - self.solver.maxtopo
            )
            land_step = abs(self.solver.maxtopo) / self.land.shape[0]
            land_index = abs(
                int((land_elevation) / land_step)
            )  # which color interval we're in
            land_index = int(max(
                0.0, min(float(land_index) + 0.0 * float(self.land.shape[0]), float(self.land.shape[0]) - 2.0)
            ))  # clamp index to avoid out-of-bounds
            land_t = (
                land_elevation - land_index * land_step
            ) / land_step  # fractional position between the colors
            self.image[i, j] = (
                self.land[land_index] * (1 - land_t)
                + self.land[land_index + 1] * land_t
            )
            # ti.Vector(
            # [0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25]
            # )
            continue

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
            land_elevation = land
            # land_elevation = land / self.solver.maxtopo  # Topo normalized
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
                continue
            if -0.25 < flow < 0:  # noqa: PLR2004
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.8039, 0.7921, 0.7372]
                )  # Wet Sand
                continue
            # print(self.solver.maxtopo)
            # print(self.solver.base_depth)
            land_shift = abs(self.solver.maxtopo) / abs(  # noqa: F841
                self.solver.base_depth - self.solver.maxtopo
            )
            land_step = (abs(self.solver.maxtopo) / 1) / self.land.shape[0]
            land_index = abs(
                int((land_elevation) / land_step)
            )  # which color interval we're in
            land_index = max(
                0, min(land_index + 0.0 * self.land.shape[0], self.land.shape[0] - 2)
            )  # clamp index to avoid out-of-bounds
            land_t = (
                land_elevation - land_index * land_step
            ) / land_step  # fractional position between the colors
            self.image[i, j] = (
                self.land[land_index] * (1 - land_t)
                + self.land[land_index + 1] * land_t
            )
            # ti.Vector(
            # [0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25]
            # )

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
            # land = max(0.0, B)  # only positive topo
            land = B
            # land_elevation = land / self.solver.maxtopo  # Topo normalized
            # land_elevation = land
            land_elevation = abs(land - self.solver.base_depth)
            index = int(vor / step)  # which color interval we're in
            index = ti.min(  # noqa: F405
                index, num_colors - 2
            )  # clamp index to avoid out-of-bounds
            t = (vor - index * step) / step  # fractional position between the colors
            if h > 0.0:  # Water area
                self.image[i, j] = (
                    self.ocean[index] * (1 - t) + self.ocean[index + 1] * t
                )
                continue
            if -0.25 < h < 0:  # noqa: PLR2004
                self.image[i, j] = ti.Vector(  # noqa: F405
                    [0.8039, 0.7921, 0.7372]
                )  # Wet Sand
                continue
            land_shift = abs(self.solver.maxtopo) / abs(  # noqa: F841
                self.solver.base_depth - self.solver.maxtopo
            )
            land_step = (abs(self.solver.maxtopo) / 1) / self.land.shape[0]
            land_index = int(abs(
                int((land_elevation) / land_step)
            ))  # which color interval we're in
            land_index = int(max(
                0.0, min(float(land_index) + 0.0 * self.land.shape[0], float(self.land.shape[0]) - 2.0))
            )  # clamp index to avoid out-of-bounds
            land_t = float(float(
                float(land_elevation) - float(land_index * land_step)
            ) / land_step)  # fractional position between the colors
            self.image[i, j] = (
                float(self.land[land_index]) * float(1.0 - land_t)
                + float(self.land[land_index + 1]) * float(land_t)
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


    @ti.kernel
    def bottom_paint(self):
        for i in self.bottom1D:
            self.bottom1D[i].x = i*self.solver.dx/self.x_scale
            self.bottom1D[i].y = 0.5+self.solver.Bottom[2,i,0]/self.y_scale

        for i in range(2*self.solver.nx-2):
            self.indexbottom1D[i] = (i + 1) // 2
            

    @ti.kernel
    def eta_paint(self):
        for i in self.eta1D:
            self.eta1D[i].x = i*self.solver.dx/self.x_scale
            self.eta1D[i].y = 0.5+self.solver.State[i,0][0]/20
    
    def Evolve_1D_Display(self):
        plotpath = './plots'
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
        i = 0.0
        use_ggui = None
        window = None
        canvas = None
        try:
            window = ti.ui.Window("CelerisAi(1D)", (1000,200))
            canvas = window.get_canvas()
            canvas.set_background_color(color=(1,1,1))
            use_ggui = True
        except:
            # TODO : Formal error handling and logging
            print("GGUI not available, reverting to legacy Taichi GUI.")
            use_ggui = False
            use_fast_gui = False # Need ti.Vector.field equiv to self.solver.pixel to use fast_gui
            window = ti.GUI(  # noqa: F405
                'CelerisAi(1D)', (1000, 200), fast_gui=use_fast_gui
                ) # fast_gui - display directly on frame buffer if not drawing shapes or text
            canvas = None
            print("Legacy GUI initialized.")
        else:
            print("GGUI initialized without issues.")

        self.Evolve_0()
        self.bottom_paint() # To plot the bottom line
        start_time = time.time() - 0.00001

        while window.running:
            self.eta_paint()
            if use_ggui:
                try:
                    canvas.circles(self.eta1D,radius=0.005,color = (0., 150/255., 255./255))
                    canvas.lines(self.bottom1D,width=0.01,indices=self.indexbottom1D,color = (128/255., 0.0, 0.))
                except Exception as e:
                    print(f"Error in GGUI circles / lines rendering: {e}")
            else:
                try:
                    for ii in range(self.solver.nx):
                        window.circle(pos=[self.eta1D[ii][0], self.eta1D[ii][1]], radius=1, color=0x00FFFF)
                        if ii >= self.solver.nx - 1:
                            continue 
                        window.line(self.bottom1D[ii], self.bottom1D[ii+1], radius=1, color = 0x39FF14)
                except Exception as e:
                    print(f"Error in legacy GUI circle / line rendering: {e}")
            self.Evolve_Steps(i)

            if i==1 or (i%100)==0:
                compTime = time.time() - start_time
                print('Current Simulation time: {:2.2f}s at step: {}-- Ratio:{:2.2f}--CompTime:{:2.2f}'.format(self.dt*i,i,(self.dt*i)/compTime,compTime))
                
                if self.saveimg:
                    frame = int(i)
                    frame_filename = 'frame_{}.png'.format(frame)
                    frame_path = os.path.join(base_frame_dir, frame_filename)
                    frame_paths.append(frame_path)
                    if use_ggui:
                        window.save_image(frame_path)
                    else:
                        tools.imwrite(self.solver.pixel.to_numpy(), frame_path)

                #window.show()
                if self.solver.outdir:
                    state=self.solver.State.to_numpy()
                    np.save('{}/state_{}.npy'.format(self.outdir,int(i)),state)
           
            window.show()
            
            if i > self.maxsteps:
                if frame_paths: # Check if there are frames to create a GIF
                    gif_filename = f"video.gif"
                    gif_path = os.path.join(base_frame_dir, gif_filename)
                    try:
                        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
                            for frame_path in frame_paths:
                                image = imageio.imread(frame_path)
                                writer.append_data(image)
                        print(f"GIF created at {gif_path}")
                    except Exception as e:
                        print(f"Error creating GIF: {e}")
                break
            i = i+1


    def Evolve_Display(  # noqa: C901, N802, D102
        self,
        vmin=None,
        vmax=None,
        variable='h',
        cmapWater='Blues_r',  # noqa: N803
        cmapLand='terrain',  # noqa: N803
        showSediment=False,  # noqa: FBT002, N803
    ):
        if vmin != None:  # noqa: E711
            self.vmin = vmin
        if vmax != None:  # noqa: E711
            self.vmax = vmax

        plotpath = './plots'
        if not os.path.exists(plotpath):  # noqa: PTH110
            os.makedirs(plotpath)  # noqa: PTH103
        i = 0
        show_gui = None
        use_ggui = None
        window = None
        canvas = None
        import platform
        os_name = platform.system()
        ON_HPC = os.environ.get("TACC_JOB_ID") or os.environ.get("SLURM_JOB_ID")
        use_ggui = not bool(ON_HPC)
        show_gui = not bool(ON_HPC)
        if show_gui:
            try:
                if bool(ON_HPC):
                    raise Exception('Headless mode detected, reverting to legacy GUI.')
                if 'Windows' in os_name:
                    # Throw exception to force legacy GUI
                    raise Exception('Windows detected, reverting GGUI to legacy GUI for reliability.')
                window = ti.ui.Window('CelerisAi', (self.solver.nx, self.solver.ny))  # noqa: F405
                canvas = window.get_canvas()
                use_ggui = True
                show_gui = True
            except:  # noqa: E722
                # TODO : Formal error handling and logging  # noqa: TD002
                print('GGUI not available, reverting to legacy Taichi GUI.')  # noqa: T201
                use_ggui = False
                use_fast_gui = False  # Need ti.Vector.field equiv to self.solver.pixel to use fast_gui
                show_gui = True
                try:
                    if bool(ON_HPC):
                        raise Exception('Headless mode detected, reverting to legacy GUI.')
                    show_gui = True
                    print('Default: Show GUI window (e.g., for desktop use.')
                    window = ti.GUI(  # noqa: F405
                        'CelerisAi', (self.solver.nx, self.solver.ny), fast_gui=use_fast_gui, show_gui=show_gui
                    )  # fast_gui - display directly on frame buffer if not drawing shapes or text
                except:
                    show_gui = False
                    print('Headless: Do not show GUI window (e.g., for command-line use on remote HPC systems.')
                    window = ti.GUI(  # noqa: F405
                        'CelerisAi', (self.solver.nx, self.solver.ny), fast_gui=use_fast_gui, show_gui=show_gui
                    )  # fast_gui - display directly on frame buffer if not drawing shapes or text
                canvas = None
                print('Legacy GUI initialized.')  # noqa: T201
            else:
                print('GGUI initialized without issues.')  # noqa: T201

        # Customized colormap
        if showSediment:
            cmap = celeris_matplotlib(  # noqa: F405
                water=cmapWater,
                land=cmapLand,
                sediment='afmhot_r',
                SedTrans=self.solver.useSedTransModel,
            )
        else:
            cmap = celeris_matplotlib(water=cmapWater, land=cmapLand)  # noqa: F405, F841
        # cmap = celeris_waves()
        # cmap = celeris_matplotlib(water='Blues_r',sediment='afmhot_r', SedTrans=self.useSedTransModel )

        self.Evolve_0()
        # Set colors - using the matplotlib colormapsand convert these into Taichi tensors
        numpy_ocean = ColorsfromMPL(cmapWater)  # noqa: F405
        numpy_land = ColorsfromMPL(cmapLand)  # noqa: F405

        self.InitColors(ocean_arr=numpy_ocean, land_arr=numpy_land)

        start_time = time.time()

        self.solver.overwrite_force(max_steps=int(self.maxsteps))
        self.solver.overwrite_wave_gauge(max_steps=int(self.maxsteps))
        self.solver.overwrite_velocity(max_steps=int(self.maxsteps))

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

            try:
                window.line(self.solver.force_sensor_begin_scaled, self.solver.force_sensor_end_scaled, radius=1, color=0x39FF14)
                for k in range(self.solver.num_wave_gauges):
                    window.circle(pos=[self.solver.wave_gauge_scaled[int(k),int(0)], self.solver.wave_gauge_scaled[int(k),int(1)]], radius=2, color=0xFF5733)
            except Exception as e:  # noqa: BLE001
                print(f'Error in rendering sensors or wave gauges using line / circle: {e}')  # noqa:

            if i == self.buffer_step:
                start_time = (
                    time.time() - 0.00001
                )  # reset the "start" time as there is overhead before loop starts, and add small shift to prevent float divide by zero
            if i == self.buffer_step or (i % self.image_step) == 0:
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
                        tools.imwrite(self.image.to_numpy(), frame_path)

            if i == self.buffer_step or (i % self.state_step) == 0:
                if self.saveimg:
                    if self.solver.outdir and not self.outdir:
                        self.outdir = self.solver.outdir
                        os.makedirs(self.outdir, exist_ok=True)  # noqa: PTH103

                    if self.outdir:
                        state = self.solver.State.to_numpy()
                        np.save(f'{self.outdir}/state_{int(i)}.npy', state)  # noqa: F405

            # Show window in the right position (after save image) for GGUI systems
            if i % self.render_step == 0 or i == 1:
                # Improve the performance.The visualization is done only every render_step timesteps
                window.show()

            if (self.output_forces) and (i % self.sensor_step == 0):
                self.solver.write_force(step=int(i))
            
            if (self.output_wave_gauges) and (i % self.sensor_step == 0):
                self.solver.write_wave_gauge(step=int(i))    
            
            if (self.output_velocimeters) and (i % self.sensor_step == 0):
                self.solver.write_velocity(step=int(i))
                
            if (self.output_gif) and (i > self.maxsteps):                 
                if self.saveimg:
                    print('Creating GIF...')  # noqa: T201
                    if frame_paths:  # Check if there are frames to create a GIF
                        print(  # noqa: T201
                            'Stitching frames ',
                            str(len(frame_paths)),
                            ' together...',
                        )
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
