from celeris.solver import *
from celeris.utils import *

USE_NEW_GUI = False # New GGUI uses Vulkan, not always available on windows/macos/debian/wsl. Provide option to use old GUI - jb

@ti.data_oriented
class Evolve:
    def __init__(self,
                 domain=None,
                 boundary_conditions = None,
                 solver=None,
                 maxsteps= 1000,
                 outdir=None
                 ):
        self.solver = solver
        self.maxsteps=maxsteps
        self.dt = self.solver.dt
        self.timeScheme = self.solver.timeScheme
    
    def Evolve_Headless(self):
        self.solver.fill_bottom_field()
        self.solver.check_depths()
        self.solver.bottom_derivatives()
        self.solver.tridiag_coeffs()
        
        for i in range(self.maxsteps):
            self.solver.switch_states()
            self.solver.run_boundaries(time=self.dt*i)
            print('Current Simulation time: {:2.2f}s at step: {}'.format(self.dt*i,i))
            # Loops kernels / passes
            self.solver.Pass1()
            self.solver.Pass1_SedTrans()
            self.solver.Pass2()
            self.solver.Pass3()
            self.solver.switch_dUdt()
            self.solver.Pass3_SedTrans()
            
            if self.timeScheme ==2:
                ttime = (i+1)*self.dt
                self.solver.switch_states_pred()
                for num_corr in range(1):
                    self.solver.switch_states()
                    self.solver.run_boundaries(time=ttime)
                    self.solver.Pass1()
                    self.solver.Pass1_SedTrans()
                    self.solver.Pass2()
                    self.solver.Pass3()
                    self.solver.Pass3_SedTrans()
        
            self.solver.switch_fluxes()
            if i==0 or (i%100)==0:
                if self.solver.outdir:
                    state=self.solver.State.to_numpy()
                    np.save('{}/state_{}.npy'.format(self.outdir,int(i)),state)
    
    @ti.func
    def brk_color (self, x,y0,y1,x0,x1):
        # Interp. to get changes in color
        return  (y0 * (x1 - x) + y1 * (x - x0) ) / (x1 - x0)
    
    @ti.kernel
    def paint(self):
        # Recall b = txBottom[2,j,i]
        # eta = h + b ; flow = eta - b
        for i,j in ti.ndrange((0,self.solver.nx),(0,self.solver.ny)):
            self.solver.pixel[i,j] = self.brk_color(self.solver.Bottom[2,j,i], 0.75, 1,self.solver.maxtopo, -1*self.solver.maxtopo)
            # To determine where the water is 
            flow = self.solver.State[j,i][0] -self.solver.Bottom[2,j,i]
            # Merge water and topo values
            if (self.solver.Bottom[2,j,i] < 0.0) or self.solver.Auxiliary2[j,i][0] > 0.0001 :
                if self.solver.useSedTransModel :
                    self.solver.pixel[i,j] = self.brk_color(self.solver.State[j,i][0], 0, 0.5,-5, 5)    # Water
                    h = max(self.solver.delta, self.solver.State[j,i][0]-self.solver.Bottom[2,j,i])
                    sed = self.solver.State_Sed[j,i][0]/h
                    
                    if sed>1.5:
                        self.solver.pixel[i,j] = self.brk_color(sed, 0.5, 0.75,0,5) # Sed
                else:
                    # eta
                    self.solver.pixel[i,j] = self.brk_color(self.solver.State[j,i][0], 0, 0.75,-2, 2)
                    
            # cmap = celeris_matplotlib(water='Blues_r',sediment='default', SedTrans=self.solver.useSedTransModel )  
            # self.solver.rgb[i,j] = cmap(self.solver.pixel[i,j])  
            self.solver.rgb[i,j] = ti.Vector([self.solver.pixel[i,j]*1.0, self.solver.pixel[i,j]*1.0, self.solver.pixel[i,j]*1.0])      
            
    def Evolve_Display(self):
        plotpath = './plots'
        if not os.path.exists(plotpath):
            os.makedirs(plotpath) 
        n = 0.0

        if ti.static(USE_NEW_GUI):
            window = ti.ui.Window("Celeris", (self.solver.nx,self.solver.ny), vsync=True)
            canvas = window.get_canvas()
        else:
            window = ti.GUI("Celeris", (self.solver.nx,self.solver.ny))#, fast_gui=True) # fast_gui - display directly on frame buffer if not drawing shapes or text
        
        
        # Customized colormap
        cmap = celeris_matplotlib(water='Blues_r',sediment='default', SedTrans=self.solver.useSedTransModel )  
        #cmap = celeris_matplotlib(water='Blues_r',sediment='afmhot_r', SedTrans=self.useSedTransModel )  
        
        self.solver.fill_bottom_field()
        #self.solver.bottom_derivatives()
        #self.solver.tridiag_coeffs()
        self.solver.tridiag_coeffs_X()
        self.solver.tridiag_coeffs_Y()

        
        while window.running:      
            # Time evolution set later
            #self.solver.switch_states()      #In any given loop, state values at "n" are in "txState" and predicted/corrected values at "n+1" are in "current_state"
            #self.solver.run_boundaries(time=self.dt*i)
            #self.solver.correctBottom()
            if (n%100)==0:
                print('Current Simulation time: {:2.2f}s at step: {}'.format(self.dt*n,n))
                self.paint()
                if ti.static(USE_NEW_GUI):
                    canvas.contour(self.solver.pixel,cmap_name=cmap) # Same functionality as set cmap-pixel-to np
                else:
                    # numpy_pixel = cmap(self.solver.pixel.to_numpy())
                    # for i,j in ti.ndrange((0,self.solver.nx),(0,self.solver.ny)):
                        # print(cmap(self.solver.pixel[i,j]))
                        # self.solver.rgb[i,j] = ti.Vector(numpy_pixel[i,j])
                    window.set_image(self.solver.rgb)
                    
                window.show()

            # Loops kernels / passes
            self.solver.Pass1()
            #if self.solver.useSedTransModel == True:
            #    self.solver.Pass1_SedTrans()
            
            self.solver.Pass2()

            #if self.useBreakingModel==1:
            #    self.solver.PassBreaking()
            #    self.cp_tempBreak_2_Breaking

            self.solver.pred_or_corrector= 1
            if self.solver.model=='SWE':
                self.solver.Pass3()
            
            if self.solver.model=='BOUS':
                self.solver.Pass3Bous()

            self.solver.copy_states(src=self.solver.dU_by_dt,dst=self.solver.predictedGradients)

            #if self.solver.useSedTransModel == True:
            #    self.solver.Pass3_SedTrans()
            #    self.solver.cp_dUdt_2_predGrad_sed()
                    
            self.solver.run_boundaries(time=self.dt*n,TempBound=self.solver.current_stateUVstar) # current_stateUVstar

            #1221
            #if self.solver.useSedTransModel == True:
            #    self.cp_tempBound_Sed_2_NewState_Sed()
            
            # 1225
            self.solver.Run_Tridiag_solver()

            if self.solver.model=='BOUS':
                self.solver.run_boundaries(time=self.dt*n,TempBound=self.solver.NewState) # NewState

            self.solver.run_boundaries(time=self.dt*n,TempBound=self.solver.NewState)

            #if self.solver.model=='BOUS':
            self.solver.copy_states(src=self.solver.F_G_star_oldGradients,dst=self.solver.F_G_star_oldOldGradients)
            self.solver.copy_states(src=self.solver.predictedF_G_star,dst=self.solver.F_G_star_oldGradients)

            if self.timeScheme ==2:
                self.solver.copy_states(src=self.solver.NewState,dst=self.solver.State)

                if self.solver.useSedTransModel == True:
                    self.solver.copy_states(src=self.solver.NewState_Sed,dst=self.solver.State_Sed)
                
                self.solver.Pass1()

                if self.solver.useSedTransModel == True:
                    self.solver.Pass1_SedTrans()
                
                self.solver.Pass2()
                # Add breaking
                #if self.useBreakingModel==1:
                #    self.solver.PassBreaking()
                #    self.cp_tempBreak_2_Breaking
                self.solver.pred_or_corrector= 2
                if self.solver.model=='SWE':
                        self.solver.Pass3()
                        
                if self.solver.model=='BOUS':
                        self.solver.Pass3Bous()
                
                #if self.solver.useSedTransModel==1:
                #        self.solver.Pass3_SedTrans()
                
                self.solver.run_boundaries(time=self.dt*n,TempBound=self.solver.current_stateUVstar) #current_stateUVStar        

                #if self.solver.useSedTransModel == True:
                    #    self.cp_tempBound_Sed_2_NewState_Sed()
                
                #1296
                self.solver.Run_Tridiag_solver()

                if self.solver.model=='BOUS':
                    self.solver.run_boundaries(time=self.dt*n,TempBound=self.solver.NewState) # NewState

                # Update the Newstate (both NLSW and BOUSS)
                self.solver.run_boundaries(time=self.dt*n,TempBound=self.solver.NewState)
                

            #if self.solver.useSedTransModel == True:
            #    self.solver.cp_temp_SedTransBottom_2_Bottom()
            #    self.solver.cp_temp_SedTransChange_2_BottomChange()
            #    if self.solver.model=='BOUS':
            #        self.solver.Update_neardry()
            #        self.solver.cp_temp_Bottom_2_Bottom()
            #        self.Update_Tridiag_coef()
            
            # 1334
            # COPY GRADIENTS
            self.solver.copy_states(src=self.solver.oldGradients,dst=self.solver.oldOldGradients)
            self.solver.copy_states(src=self.solver.predictedGradients,dst=self.solver.oldGradients)
            # COPY future states
            self.solver.copy_states(src=self.solver.NewState,dst=self.solver.State)
            self.solver.copy_states(src=self.solver.current_stateUVstar,dst=self.solver.StateUVstar)

                
            if self.solver.useSedTransModel == True:
                self.solver.copy_states(src=self.solver.oldGradients_Sed,dst=self.solver.oldOldGradients_Sed)
                self.solver.copy_states(src=self.solver.predictedGradients_Sed,dst=self.solver.oldGradients_Sed)
                self.solver.copy_states(src=self.solver.NewState_Sed,dst=self.solver.State_Sed)


            
            #if i==0 or (i%100)==0:
            #    window.save_image('./plots/frame_{}.png'.format(i))
            #    if self.solver.outdir:
            #        state=self.solver.State.to_numpy()
            #        np.save('{}/state_{}.npy'.format(self.outdir,int(i)),state)
            if n> self.maxsteps:
                break
            n = n+1
