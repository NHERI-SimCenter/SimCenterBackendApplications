# --------------------------------------------------------------------------------------------------
# Example 8. 3D Steel W-section frame
#			Silvia Mazzoni & Frank McKenna, 2006
# nonlinearBeamColumn element, inelastic fiber section
#



# SET UP ----------------------------------------------------------------------------
model BasicBuilder -ndm 3 -ndf 6;	# Define the model builder, ndm=#dimension, ndf=#dofs


# ==================================================================================
# units 
# ==================================================================================
set meter   1.; 			            # define basic units -- input units
set Newton  1.; 			            # define basic units -- input units
set kN      1000.; 			            # define basic units -- input units
set MN      1.e6; 			            # define basic units -- input units
set mm      0.001; 			            # define basic units -- input units
set cm      0.01; 			            # define basic units -- input units
set Pa      1.; 			            # define basic units -- input units
set kPa     1.e3; 			            # define basic units -- input units
set MPa     1.e6; 			            # define basic units -- input units
set GPa     1.e9; 			            # define basic units -- input units
set in      0.0254; 				    # define basic units -- output units
set kip     4448.2216153; 			    # define basic units -- output units
set sec 1.; 			                # define basic units -- output units
set LunitTXT "meter";			        # define basic-unit text for output
set FunitTXT "Newton";			        # define basic-unit text for output
set TunitTXT "sec";			            # define basic-unit text for output
set ft  [expr 12.*$in]; 		            # define engineering units
set ksi [expr $kip/pow($in,2)];
set psi [expr $ksi/1000.];
set lbf [expr $psi*$in*$in];		    # pounds force
set pcf [expr $lbf/pow($ft,3)];		    # pounds per cubic foot
set psf [expr $lbf/pow($ft,3)];		    # pounds per square foot
set in2 [expr $in*$in]; 		        # inch^2
set in4 [expr $in*$in*$in*$in]; 		# inch^4
set cm  [expr $in/2.54];		            # centimeter, needed for displacement input in MultipleSupport excitation
set PI  [expr 2*asin(1.0)]; 		        # define constants
set g   [expr 32.2*$ft/pow($sec,2)]; 	    # gravitational acceleration
set Ubig 1.e10; 			            # a really large number
set Usmall [expr 1/$Ubig]; 		        # a really small number

# ==================================================================================
# fiber section W-section
# ==================================================================================
proc Wsection { secID matID d bf tf tw nfdw nftw nfbf nftf} {
	# ###################################################################
	# Wsection  $secID $matID $d $bf $tf $tw $nfdw $nftw $nfbf $nftf
	# ###################################################################
	# create a standard W section given the nominal section properties
	# written: Remo M. de Souza
	# date: 06/99
	# modified: 08/99  (according to the new general modelbuilder)
	# input parameters
	# secID - section ID number
	# matID - material ID number 
	# d  = nominal depth
	# tw = web thickness
	# bf = flange width
	# tf = flange thickness
	# nfdw = number of fibers along web depth 
	# nftw = number of fibers along web thickness
	# nfbf = number of fibers along flange width
	# nftf = number of fibers along flange thickness
  	
	set dw [expr $d - 2 * $tf]
	set y1 [expr -$d/2]
	set y2 [expr -$dw/2]
	set y3 [expr  $dw/2]
	set y4 [expr  $d/2]
  
	set z1 [expr -$bf/2]
	set z2 [expr -$tw/2]
	set z3 [expr  $tw/2]
	set z4 [expr  $bf/2]
  
	section fiberSec  $secID  {
   		#                     nfIJ  nfJK    yI  zI    yJ  zJ    yK  zK    yL  zL
   		patch quadr  $matID  $nfbf $nftf   $y1 $z4   $y1 $z1   $y2 $z1   $y2 $z4
   		patch quadr  $matID  $nftw $nfdw   $y2 $z3   $y2 $z2   $y3 $z2   $y3 $z3
   		patch quadr  $matID  $nfbf $nftf   $y3 $z4   $y3 $z1   $y4 $z1   $y4 $z4
	}
}



# ==================================================================================
# define Geometry
# ==================================================================================
set BaseColumnsIDs {};		# list of nodes at base of columns
set StructureNodes {};		# list of all nodes in the structure
# define NODAL COORDINATES
set Dlevel 10000;	# numbering increment for new-level nodes
set Dframe 100;	# numbering increment for new-frame nodes
set NFrame [expr $NBayZ + 1];	# number of frames

for {set frame 1} {$frame <=[expr $NFrame]} {incr frame 1} {
	set Y [expr ($frame-1)*$LGird];
	set Y [expr $Y + $StartNodeY]
	for {set level 1} {$level <=[expr $NStory+1]} {incr level 1} {
		set Z [expr ($level-1)*$LCol];
		set Z [expr $Z + $StartNodeZ]
		for {set pier 1} {$pier <= [expr $NBay+1]} {incr pier 1} {
			set X [expr ($pier-1)*$LBeam];
			set X [expr $X + $StartNodeX]
			 
			set nodeID [expr $frame*$Dframe+$level*$Dlevel+$pier]
			node $nodeID $X $Y $Z;		# actually define node
			lappend StructureNodes $nodeID
			if {$level == 1} {lappend BaseColumnsIDs $nodeID}
		}
	}
}


# ==================================================================================
# rigid diaphragm nodes
# ==================================================================================
set RigidDiaphragm OFF ;		# options: ON, OFF. specify this before the analysis parameters are set the constraints are handled differently.
if {$RigidDiaphragm == "ON"} {
	set Xa [expr ($NBay*$LBeam)/2];		# mid-span coordinate for rigid diaphragm
	set Ya [expr ($NFrame-1)*$LGird/2];
	# move the structure to the start node
	set Xa [expr $Xa + $StartNodeX]
	set Ya [expr $Ya + $StartNodeY]

	set iMasterNode ""
	for {set level 2} {$level <=[expr $NStory+1]} {incr level 1} {
		set Z [expr ($level-1)*$LCol];
		set Z  [expr $Z + $StartNodeZ]


		# rigid-diaphragm nodes in center of each diaphram
		set MasterNodeID [expr 9900+$level]
		node $MasterNodeID $Xa $Ya $Z;		    # master nodes for rigid diaphragm
		fix $MasterNodeID 0  1  0  1  0  1;		# constrain other dofs that don't belong to rigid diaphragm control
		lappend iMasterNode $MasterNodeID
		set perpDirn 3;				# perpendicular to plane of rigid diaphragm
		for {set frame 1} {$frame <=[expr $NFrame]} {incr frame 1} {
			for {set pier 1} {$pier <= [expr $NBay+1]} {incr pier 1} {
				set nodeID [expr $level*$Dlevel+$frame*$Dframe+$pier]
				rigidDiaphragm $perpDirn $MasterNodeID $nodeID; 	# define Rigid Diaphram,
				# puts [nodeCoord $nodeID]
				# puts [nodeCoord $MasterNodeID]
			}
		}
		lappend StructureNodes $MasterNodeID
	}
}


# ==================================================================================
# calculated MODEL PARAMETERS, particular to this model
# ==================================================================================
# Set up parameters that are particular to the model for displacement control
set IDctrlNode [expr int((($NStory+1)*$Dlevel+$NFrame*$Dframe)+1)];		# node where displacement is read for displacement control
set IDctrlDOF 1;					                                    # degree of freedom of displacement read for displacement control
set LBuilding [expr $NStory*$LCol];			                            # total building height



# ==================================================================================
# Define  SECTIONS -------------------------------------------------------------
# ==================================================================================

# define section tags:
set ColSecTag       1
set BeamSecTag      2
set GirdSecTag      3
set ColSecTagFiber  4
set BeamSecTagFiber 5
set GirdSecTagFiber 6
set SecTagTorsion   70


if {$SectionType == "Elastic"} {
	# material properties:
	set Es [expr 29000*$ksi];		# Steel Young's Modulus
	set nu 0.3;			# Poisson's ratio
	set Gs [expr $Es/2./[expr 1+$nu]];  	# Torsional stiffness Modulus
	set J $Ubig;			# set large torsional stiffness
	# column sections: W27x114
	set AgCol [expr 33.5*pow($in,2)];		# cross-sectional area
	set IzCol [expr 4090.*pow($in,4)];		# moment of Inertia
	set IyCol [expr 159.*pow($in,4)];		# moment of Inertia
	# beam sections: W24x94
	set AgBeam [expr 27.7*pow($in,2)];		# cross-sectional area
	set IzBeam [expr 2700.*pow($in,4)];		# moment of Inertia
	set IyBeam [expr 109.*pow($in,4)];		# moment of Inertia
	# girder sections: W24x94
	set AgGird [expr 27.7*pow($in,2)];		# cross-sectional area
	set IzGird [expr 2700.*pow($in,4)];		# moment of Inertia
	set IyGird [expr 109.*pow($in,4)];		# moment of Inertia
	
	section Elastic $ColSecTag  $Es $AgCol $IzCol $IyCol $Gs $J
	section Elastic $BeamSecTag $Es $AgBeam $IzBeam $IyBeam $Gs $J
	section Elastic $GirdSecTag $Es $AgGird $IzGird $IyGird $Gs $J

	set matIDhard 1;		# material numbers for recorder (this stressstrain recorder will be blank, as this is an elastic section)

} elseif {$SectionType == "FiberSection"} {
	# define MATERIAL properties 
	set Fy [expr 60.0*$ksi]
	set Es [expr 29000*$ksi];		# Steel Young's Modulus
	set nu 0.3;
	set Gs [expr $Es/2./[expr 1+$nu]];  # Torsional stiffness Modulus
	set Hiso 0
	set Hkin 1000
	set matIDhard 1
	uniaxialMaterial Hardening  $matIDhard $Es $Fy   $Hiso  $Hkin

	# ELEMENT properties
	# Structural-Steel W-section properties
	# column sections: W27x114
	set d  [expr 27.29*$in];	# depth
	set bf [expr 10.07*$in];	# flange width
	set tf [expr 0.93*$in];	    # flange thickness
	set tw [expr 0.57*$in];	    # web thickness
	set nfdw 16;		        # number of fibers along dw
	set nftw 2;		            # number of fibers along tw
	set nfbf 16;		        # number of fibers along bf
	set nftf 4;			        # number of fibers along tf
	Wsection  $ColSecTagFiber $matIDhard $d $bf $tf $tw $nfdw $nftw $nfbf $nftf
	# beam sections: W24x94
	set d [expr 24.31*$in];	# depth
	set bf [expr 9.065*$in];	# flange width
	set tf [expr 0.875*$in];	# flange thickness
	set tw [expr 0.515*$in];	# web thickness
	set nfdw 16;		# number of fibers along dw
	set nftw 2;		# number of fibers along tw
	set nfbf 16;		# number of fibers along bf
	set nftf 4;			# number of fibers along tf
	Wsection  $BeamSecTagFiber $matIDhard $d $bf $tf $tw $nfdw $nftw $nfbf $nftf
	# girder sections: W24x94
	set d [expr 24.31*$in];	# depth
	set bf [expr 9.065*$in];	# flange width
	set tf [expr 0.875*$in];	# flange thickness
	set tw [expr 0.515*$in];	# web thickness
	set nfdw 16;		# number of fibers along dw
	set nftw 2;		# number of fibers along tw
	set nfbf 16;		# number of fibers along bf
	set nftf 4;			# number of fibers along tf
	Wsection  $GirdSecTagFiber $matIDhard $d $bf $tf $tw $nfdw $nftw $nfbf $nftf
	
	# assign torsional Stiffness for 3D Model
	uniaxialMaterial Elastic $SecTagTorsion $Ubig
	section Aggregator $ColSecTag $SecTagTorsion T -section $ColSecTagFiber
	section Aggregator $BeamSecTag $SecTagTorsion T -section $BeamSecTagFiber
	section Aggregator $GirdSecTag $SecTagTorsion T -section $GirdSecTagFiber
} else {
	puts "No section has been defined"
	return -1
}

set QdlCol [expr 114*$lbf/$ft]; 	# W-section weight per length
set QBeam  [expr 94*$lbf/$ft];		# W-section weight per length
set QGird  [expr 94*$lbf/$ft];		# W-section weight per length

# ==================================================================================
# define ELEMENTS
# ==================================================================================
# set up geometric transformations of element
#   separate columns and beams, in case of P-Delta analysis for columns
set IDColTransf  1; # all columns
set IDBeamTransf 2; # all beams
set IDGirdTransf 3; # all girds
set ColTransfType Linear ;		# options for columns: Linear PDelta Corotational 
geomTransf $ColTransfType  $IDColTransf  0 1 0;			# orientation of column stiffness affects bidirectional response.
geomTransf Linear $IDBeamTransf 0 -1 0;
geomTransf Linear $IDGirdTransf 1  0 0;


# Define Beam-Column Elements
set StructureElements {}
set numIntgrPts 5;	# number of Gauss integration points for nonlinear curvature distribution
# columns
set N0col [expr 10000-1];	# column element numbers
set level 0
for {set frame 1} {$frame <=[expr $NFrame]} {incr frame 1} {
	for {set level 1} {$level <=$NStory} {incr level 1} {
		for {set pier 1} {$pier <= [expr $NBay+1]} {incr pier 1} {
			set elemID [expr $N0col  +$level*$Dlevel + $frame*$Dframe+$pier]
			set nodeI [expr  $level*$Dlevel + $frame*$Dframe+$pier]
			set nodeJ  [expr  ($level+1)*$Dlevel + $frame*$Dframe+$pier]
			element nonlinearBeamColumn $elemID $nodeI $nodeJ $numIntgrPts $ColSecTag $IDColTransf;		# columns
			lappend StructureElements $elemID
		}
	}
}

# beams -- parallel to X-axis
set N0beam 1000000;	# beam element numbers
for {set frame 1} {$frame <=[expr $NFrame]} {incr frame 1} {
	for {set level 2} {$level <=[expr $NStory+1]} {incr level 1} {
		for {set bay 1} {$bay <= $NBay} {incr bay 1} {
			set elemID [expr $N0beam +$level*$Dlevel + $frame*$Dframe+ $bay]
			set nodeI [expr $level*$Dlevel + $frame*$Dframe+ $bay]
			set nodeJ  [expr $level*$Dlevel + $frame*$Dframe+ $bay+1]
			element nonlinearBeamColumn $elemID $nodeI $nodeJ $numIntgrPts $BeamSecTag $IDBeamTransf;	# beams
			lappend StructureElements $elemID
		}
	}
}

# girders -- parallel to Z-axis
set N0gird 2000000;	# gird element numbers
for {set frame 1} {$frame <=[expr $NFrame-1]} {incr frame 1} {
	for {set level 2} {$level <=[expr $NStory+1]} {incr level 1} {
		for {set bay 1} {$bay <= $NBay+1} {incr bay 1} {
			set elemID [expr $N0gird + $level*$Dlevel +$frame*$Dframe+ $bay]
			set nodeI [expr   $level*$Dlevel + $frame*$Dframe+ $bay]
			set nodeJ  [expr  $level*$Dlevel + ($frame+1)*$Dframe+ $bay]
			element nonlinearBeamColumn $elemID $nodeI $nodeJ $numIntgrPts $GirdSecTag $IDGirdTransf;		# Girds
			lappend StructureElements $elemID
		}
	}
}

# --------------------------------------------------------------------------------------------------------------------------------
# Define GRAVITY LOADS, weight and masses
# calculate dead load of frame, assume this to be an internal frame (do LL in a similar manner)
# calculate distributed weight along the beam length
set GammaConcrete    [expr 150*$pcf];   		# Reinforced-Concrete floor slabs
set Tslab            [expr 6*$in];			    # 6-inch slab
set Lslab            [expr 2*$LGird/2]; 	    # slab extends a distance of $LGird/2 in/out of plane
set DLfactor         2.0;				# scale dead load up a little
set Qslab            [expr $GammaConcrete*$Tslab*$Lslab*$DLfactor]; 
set QdlBeam          [expr $Qslab + $QBeam]; 	# dead load distributed along beam (one-way slab)
set QdlGird          $QGird; 			# dead load distributed along girder
set WeightCol        [expr $QdlCol*$LCol];  		# total Column weight
set WeightBeam       [expr $QdlBeam*$LBeam]; 	# total Beam weight
set WeightGird       [expr $QdlGird*$LGird]; 	# total Beam weight

# assign masses to the nodes that the columns are connected to 
# each connection takes the mass of 1/2 of each element framing into it (mass=weight/$g)
set iFloorWeight ""
set WeightTotal 0.0
set sumWiHi 0.0;		# sum of storey weight times height, for lateral-load distribution

for {set frame 1} {$frame <=[expr $NFrame]} {incr frame 1} {
	if {$frame == 1 || $frame == $NFrame}  {
		set GirdWeightFact 1;		# 1x1/2girder on exterior frames
	} else {
		set GirdWeightFact 2;		# 2x1/2girder on interior frames
	}
	for {set level 2} {$level <=[expr $NStory+1]} {incr level 1} { ;		
		set FloorWeight 0.0
		if {$level == [expr $NStory+1]}  {
			set ColWeightFact 1;		# one column in top story
		} else {
			set ColWeightFact 2;		# two columns elsewhere
		}
		for {set pier 1} {$pier <= [expr $NBay+1]} {incr pier 1} {;
			if {$pier == 1 || $pier == [expr $NBay+1]} {
				set BeamWeightFact 1;	# one beam at exterior nodes
			} else {;
				set BeamWeightFact 2;	# two beams elewhere
			}
			set WeightNode [expr $ColWeightFact*$WeightCol/2 + $BeamWeightFact*$WeightBeam/2 + $GirdWeightFact*$WeightGird/2]
			set MassNode [expr $WeightNode/$g];
			set nodeID [expr $level*$Dlevel+$frame*$Dframe+$pier]
			mass $nodeID $MassNode $MassNode 0. 0. 0. 0.;			# define mass
			set FloorWeight [expr $FloorWeight+$WeightNode];
		}
		lappend iFloorWeight $FloorWeight
		set WeightTotal [expr $WeightTotal+ $FloorWeight]
		set sumWiHi [expr $sumWiHi+$FloorWeight*($level-1)*$LCol];		# sum of storey weight times height, for lateral-load distribution
	}
}
set MassTotal [expr $WeightTotal/$g];						# total mass


# # Define RECORDERS -------------------------------------------------------------
# set FreeNodeID [expr $NFrame*$Dframe+($NStory+1)*$Dlevel+($NBay+1)];					# ID: free node
# set SupportNodeFirst [lindex $iSupportNode 0];						# ID: first support node
# set SupportNodeLast [lindex $iSupportNode [expr [llength $iSupportNode]-1]];			# ID: last support node
# set FirstColumn [expr $N0col  + 1*$Dframe+1*$Dlevel +1];							# ID: first column
# recorder Node -file $dataDir/DFree.out -time -node $FreeNodeID  -dof 1 2 3 disp;				# displacements of free node
# recorder Node -file $dataDir/DBase.out -time -nodeRange $SupportNodeFirst $SupportNodeLast -dof 1 2 3 disp;	# displacements of support nodes
# recorder Node -file $dataDir/RBase.out -time -nodeRange $SupportNodeFirst $SupportNodeLast -dof 1 2 3 reaction;	# support reaction
# recorder Drift -file $dataDir/DrNode.out -time -iNode $SupportNodeFirst  -jNode $FreeNodeID   -dof 1 -perpDirn 2;	# lateral drift
# recorder Element -file $dataDir/Fel1.out -time -ele $FirstColumn localForce;					# element forces in local coordinates
# recorder Element -file $dataDir/ForceEle1sec1.out -time -ele $FirstColumn section 1 force;			# section forces, axial and moment, node i
# recorder Element -file $dataDir/DefoEle1sec1.out -time -ele $FirstColumn section 1 deformation;			# section deformations, axial and curvature, node i
# recorder Element -file $dataDir/ForceEle1sec$numIntgrPts.out -time -ele $FirstColumn section $numIntgrPts force;			# section forces, axial and moment, node j
# recorder Element -file $dataDir/DefoEle1sec$numIntgrPts.out -time -ele $FirstColumn section $numIntgrPts deformation;		# section deformations, axial and curvature, node j
# set yFiber [expr 0.];								# fiber location for stress-strain recorder, local coords
# set zFiber [expr 0.];								# fiber location for stress-strain recorder, local coords
# recorder Element -file $dataDir/SSEle1sec1.out -time -ele $FirstColumn section $numIntgrPts fiber $yFiber $zFiber stressStrain;	# steel fiber stress-strain, node i



# GRAVITY -------------------------------------------------------------
# define GRAVITY load applied to beams and columns -- eleLoad applies loads in local coordinate axis
pattern Plain 101 Linear {
	for {set frame 1} {$frame <=[expr $NFrame]} {incr frame 1} {
		for {set level 1} {$level <=$NStory} {incr level 1} {
			for {set pier 1} {$pier <= [expr $NBay+1]} {incr pier 1} {
				set elemID [expr $N0col  + $level*$Dlevel +$frame*$Dframe+$pier]
					eleLoad -ele $elemID -type -beamUniform 0. 0. -$QdlCol; 	# COLUMNS		}
		}
	}
	for {set frame 1} {$frame <=[expr $NFrame]} {incr frame 1} {
		for {set level 2} {$level <=[expr $NStory+1]} {incr level 1} {
			for {set bay 1} {$bay <= $NBay} {incr bay 1} {
				set elemID [expr $N0beam + $level*$Dlevel +$frame*$Dframe+ $bay]
					eleLoad -ele $elemID  -type -beamUniform -$QdlBeam 0.; 	# BEAMS
			}
		}
	}
	for {set frame 1} {$frame <=[expr $NFrame-1]} {incr frame 1} {
		for {set level 2} {$level <=[expr $NStory+1]} {incr level 1} {
			for {set bay 1} {$bay <= $NBay+1} {incr bay 1} {
				set elemID [expr $N0gird + $level*$Dlevel +$frame*$Dframe+ $bay]
				eleLoad -ele $elemID  -type -beamUniform -$QdlGird 0.;	# GIRDS
			}
		}
	}

}



# write beam nodes to a file
set StructureNodesFile "$outputdir/StructureNodes.dat"
if {[file exists $StructureNodesFile] == 1} { file delete $StructureNodesFile }
set StructureNodesID [open $StructureNodesFile "w"]
foreach node $StructureNodes {
	puts $StructureNodesID "$node [nodeCoord $node]"
}
close $StructureNodesID

# write beam elements to a file
set StructureElemsFile "$outputdir/StructureElements.dat"
if {[file exists $StructureElemsFile] == 1} { file delete $StructureElemsFile }
set StructureElemsID [open $StructureElemsFile "w"]
foreach ele $StructureElements {
	puts $StructureElemsID "$ele [eleNodes $ele]"
}
close $StructureElemsID




# write baseColumns information to a file
set BaseColumnsFile "$outputdir/BaseColumnsNodes.dat"
if {[file exists $BaseColumnsFile] == 1} { file delete $BaseColumnsFile }
set BaseColumnsID [open $BaseColumnsFile "w"]
foreach node $BaseColumnsIDs {
	puts $BaseColumnsID "$node [nodeCoord $node]"
}
close $BaseColumnsID




# # Gravity-analysis parameters -- load-controlled static analysis
# set Tol 1.0e-3;			         # convergence tolerance for test
# variable constraintsTypeGravity Plain;		# default;
# if {  [info exists RigidDiaphragm] == 1} {
# 	if {$RigidDiaphragm=="ON"} {
# 		variable constraintsTypeGravity Lagrange;	#  large model: try Transformation
# 	};	# if rigid diaphragm is on
# };	# if rigid diaphragm exists

# constraints $constraintsTypeGravity ;     		# how it handles boundary conditions
# numberer    RCM;			                    # renumber dof's to minimize band-width (optimization), if you want to
# system      BandGeneral ;		                # how to store and solve the system of equations in the analysis (large model: try UmfPack)
# test        EnergyIncr $Tol 6 ; 	         	# determine if convergence has been achieved at the end of an iteration step
# algorithm   Newton;			                    # use Newton's solution algorithm: updates tangent stiffness at every iteration
# set         NstepGravity 10;  		            # apply gravity in 10 steps
# set         DGravity [expr 1./$NstepGravity]; 	# first load increment;
# integrator  LoadControl $DGravity;	            # determine the next time step for an analysis
# analysis    Static;			                    # define type of analysis static or transient
# analyze     $NstepGravity;		                # apply gravity
# # ------------------------------------------------- maintain constant gravity loads and reset time to zero
# loadConst -time 0.0

# # -------------------------------------------------------------
# puts "Model Built"


