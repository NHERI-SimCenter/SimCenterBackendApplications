# ===================================================================== #
# 3D test model for the pml element modeling the plane strain field     #
# University of Washington, Department of Civil and Environmental Eng   #
# Geotechnical Eng Group, A. Pakzad, P. Arduino - Jun 2023              #
# Basic units are m, kg, s (SI units)								    #
# ===================================================================== #

# ============================================================================
# define helper functions
# ============================================================================
proc writeNodesinFile {filename nodes} {
    # check if the file exists
    if {[file exists $filename] == 1} { file delete $filename }
    set f [open $filename "w"]
    foreach node $nodes {
        puts $f "$node [nodeCoord $node]"
    }
    close $f
}

proc writeElesinFile {filename eles} {
    # check if the file exists
    if {[file exists $filename] == 1} { file delete $filename }
    set f [open $filename "w"]
    foreach ele $eles {
        puts $f "$ele [eleNodes $ele]"
    }
    close $f
}


proc getMaxNodeTag {np pid} {
    set maxNodeTag -1
    foreach node [getNodeTags] {if {$node > $maxNodeTag} {set maxNodeTag $node}}
    barrier
    # puts "maxNodeTag: $maxNodeTag pid: $pid np: $np"
    if {$pid == 0} {set maxNodeTaglist {}}
    if {$pid == 0} {
        for {set i 0} {$i < $np} {incr i} {lappend maxNodeTaglist -1}
        # set the first element of the list to the maxNodeTag
        set maxNodeTaglist [lreplace $maxNodeTaglist 0 0 $maxNodeTag]
    }
    barrier
    if {$pid > 0} {
        send -pid 0 $maxNodeTag
    } else {
        for {set i 1} {$i < $np} {incr i} {
            recv -pid $i maxNodeTag
            set maxNodeTaglist [lreplace $maxNodeTaglist $i $i $maxNodeTag]
        }
    }
    barrier
    if {$pid == 0} {
        set maxNodeTag -1
        foreach node $maxNodeTaglist {if {$node > $maxNodeTag} {set maxNodeTag $node}}
        # puts "maximum: $maxNodeTag"
    }

    barrier
    # return the maxNodeTag
    if {$pid == 0} {
        for {set i 1} {$i < $np} {incr i} {
            send -pid $i $maxNodeTag
        }
    } else {
        recv -pid 0 maxNodeTag
    }
    barrier
    if {$maxNodeTag == -1} {set maxNodeTag 0}
    return $maxNodeTag
}


proc getMaxEleTag {np pid} {
    set maxEleTag -1
    foreach ele [getEleTags] {if {$ele > $maxEleTag} {set maxEleTag $ele}}
    barrier
    # puts "maxEleTag: $maxEleTag pid: $pid np: $np"
    if {$pid == 0} {set maxEleTaglist {}}
    if {$pid == 0} {
        for {set i 0} {$i < $np} {incr i} {lappend maxEleTaglist -1}
        # set the first element of the list to the maxEleTag
        set maxEleTaglist [lreplace $maxEleTaglist 0 0 $maxEleTag]
    }
    barrier
    if {$pid > 0} {
        send -pid 0 $maxEleTag
    } else {
        for {set i 1} {$i < $np} {incr i} {
            recv -pid $i maxEleTag
            set maxEleTaglist [lreplace $maxEleTaglist $i $i $maxEleTag]
        }
    }
    barrier
    if {$pid == 0} {
        set maxEleTag -1
        foreach ele $maxEleTaglist {if {$ele > $maxEleTag} {set maxEleTag $ele}}
        # puts "maximum: $maxEleTag"
    }

    barrier
    # return the maxEleTag
    if {$pid == 0} {
        for {set i 1} {$i < $np} {incr i} {
            send -pid $i $maxEleTag
        }
    } else {
        recv -pid 0 maxEleTag
    }
    barrier
    if {$maxEleTag == -1} {set maxEleTag 0}
    return $maxEleTag
}

proc addVartoModelInfoFile {fileName varName varValue pid writerpid} {
    if {$pid != $writerpid} {return}
    set f [open "$fileName" "r"]
    set lines [split [read $f] "\n"]
    close $f
    set f [open "$fileName" "w+"]
    set j 1
    foreach line $lines {
        set nextline [lindex $lines $j]
        if {$line == "\}"} {
            puts $f "\t\"$varName\": \"$varValue\""
            puts $f "\}"
            break
        } 
        if {$nextline == "\}" && $j > 1} {
            puts $f "$line,"
        } else {
            puts $f "$line"
        }
        incr j
    }
    close $f
}
     

proc initializModelinfoFile {fileName pid writerpid } {
    if {$pid != $writerpid} {return}
    if {[file exists $fileName] == 1} { file delete $fileName }
    set f [open "$fileName" "w+"]
    puts $f "\{"
    puts $f "\}"
    close $f
}



# ============================================================================
#  get the number of processors
# ============================================================================

set pid [getPID]
set np  [getNP]
set ModelInfoFile "$outputdir/ModelInfo.json"

if {$DRM_Location == "designsafe"} {
    set ModelInfoFile "ModelInfo.json"
    set meshdir "."
    set outputdir "../Results"
    set DRMFile "../$DRMFile"

}


initializModelinfoFile $ModelInfoFile $pid 0
addVartoModelInfoFile $ModelInfoFile "numberOfProcessors" $np $pid 0



# ============================================================================
# create the structure model
# ============================================================================
#  structure model will be built on the pid=0 processor all 
#  other processors will be used for the for fundation, soil, DRM, and PML elements
if {$pid==0} { 
    
    if {$HaveStructure == "YES"} {
        puts "Creating the structure model"
    } else {
        puts "Structure model is skipped"
    }
}

if {$HaveStructure == "YES" } {
    if {$pid < $structurecores} {
        if {$StructureType == "STEEL"} {
            source StructresFiles/SteelStructure.tcl
        }
        if {$StructureType == "CONCRETE"} {
            source ConcreteStructure.tcl
        }
        if {$StructureType == "Custom"} {
            source CustomStructure.tcl
        }
        puts "Structure model is created"
    }
}
barrier

# ============================================================================
# update the maxNodeTag and maxEleTag
# ============================================================================
if {$HaveStructure== "YES" } {
    set maxNodeTag [getMaxNodeTag $np $pid]
    set maxEleTag  [getMaxEleTag $np $pid]
} else {
    set maxNodeTag 0
    set maxEleTag  0
}
barrier

# ============================================================================
#  Setting the maxNodeTag and maxEleTag for the structure model
# ============================================================================
# this is really important to set the maxNodeTag and maxEleTag for the structure model
set StructureMaxNodeTag $maxNodeTag
set StructureMaxEleTag  $maxEleTag
barrier
addVartoModelInfoFile $ModelInfoFile "StructureMaxNodeTag" $StructureMaxNodeTag $pid 0
addVartoModelInfoFile $ModelInfoFile  "StructureMaxEleTag"  $StructureMaxEleTag $pid 0


# ============================================================================
# bulding regular elements
# ============================================================================
model BasicBuilder -ndm 3 -ndf 3
# set Vs          200.0
# set nu          0.25         ;# --- Poisson's Ratio
# set rho         2300.0                 ;# --- Density KG/M^3
set G           [expr $Vs*$Vs*$rho]
set E           [expr 2*$G*(1+$nu)]
nDMaterial ElasticIsotropic 1 $E $nu $rho;
nDMaterial ElasticIsotropic 2 [expr $E *100] $nu [expr $rho*4.0];
set SoilmatTag1 "1 0.0 0.0 0.0";
set FoundationmatTag1 "2 0.0 0.0 -9.81";


# ============================================================================
# create regular nodes and elements
# ============================================================================
if {$pid>= $structurecores && $pid < [expr $regcores+$structurecores]} {
    puts "StructureMaxNodeTag : $StructureMaxNodeTag\n"
    model BasicBuilder -ndm 3 -ndf 3
    eval "source $meshdir/Nodes$pid.tcl"

    set matTag1 "1 0.0 0.0 0.0";
    set elementType "stdBrick"
    eval "source $meshdir/Elements$pid.tcl"

    puts "Regular elements are created"
    set recordList [getNodeTags]
    set elerecordList [getEleTags]
    set rayleighalpha 0.0
    set rayleighbeta 0.0
}

barrier

# ============================================================================
# creating DRM elements
# ============================================================================
if {$pid >= [expr $regcores +$structurecores] && $pid < [expr $regcores + $drmcores + $structurecores]} {
    model BasicBuilder -ndm 3 -ndf 3
    eval "source $meshdir/Nodes$pid.tcl"
    set matTag1 "1 0.0 0.0 0.0";
    set elementType "stdBrick"
    eval "source $meshdir/Elements$pid.tcl"

    set elelist [getEleTags]
    # region 1 -ele $elelist -rayleigh $Damp_alpha $Damp_beta 0.0 0.0
    set rayleighalpha 0.0
    set rayleighbeta 0.0
    puts "DRM elements are created"
}
barrier

# ============================================================================
#  Adding pile elements
# ============================================================================
if {$pid == 0} {
    if {$HavePiles == "YES"} {
        puts "Creating pile elements"
    } else {
        puts "Pile elements are skipped"
    }
}

if {$HavePiles == "YES"} {
    if {$pid == $structurecores} {
        model BasicBuilder -ndm 3 -ndf 6
        set pileElements {}
        set pileNodes    {}
        # creating pile elements

        set secTag        1
        set transfTag     1
        set diameter      1. ; # pile diameter (m)
        set radius        [expr $diameter/2.]
        set pi            3.141593
        set Epile         1e10
        set nu            0.3
        set Gpile         [expr $Epile/(2*(1+$nu))]
        set Area          [expr ($diameter**2)*$pi/2.]
        set Iy            [expr ($diameter**4)*$pi/64.]
        set Iz            [expr ($diameter**4)*$pi/64.]
        set J             [expr ($diameter**4)*$pi/32.]
        set transfType    "PDelta"; # PDelta, Linear, Corotational
        
        # section Elastic $secTag $E $A $Iz $Iy $G $J 
        section Elastic $secTag $Epile $Area $Iz $Iy $Gpile $J

        set numpiles [llength $pilelist]
        # puts "Number of piles: $numpiles"
        set j 0
        foreach pile $pilelist {
            set xbottom [dict get $pile xbottom]
            set ybottom [dict get $pile ybottom]
            set zbottom [dict get $pile zbottom]
            set xtop    [dict get $pile xtop]
            set ytop    [dict get $pile ytop]
            set ztop    [dict get $pile ztop]
            set pilenumelements [dict get $pile numberofElements]
            set nPeri         8
            set nLong         8
            set numIntgrPts   5


            # creating pile nodes
            set pilenumnodes [expr $pilenumelements + 1]
            for {set i 1} {$i <= $pilenumnodes} {incr i} {
                set xcoord [expr $xbottom + ($xtop-$xbottom)*($i-1)/$pilenumelements]
                set ycoord [expr $ybottom + ($ytop-$ybottom)*($i-1)/$pilenumelements]
                set zcoord [expr $zbottom + ($ztop-$zbottom)*($i-1)/$pilenumelements]
                set Nodeid [expr $soilfoundation_num_nodes + $maxNodeTag + $j*$pilenumnodes + $i]
                node $Nodeid $xcoord $ycoord $zcoord
                lappend pileNodes $Nodeid
            }

            set P1x $xtop
            set P1y $ytop
            set P1z $ztop



            # normal vector to the pile
            set normalX [expr $xtop - $xbottom]
            set normalY [expr $ytop - $ybottom]
            set normalZ [expr $ztop - $zbottom]
            set norm    [expr sqrt($normalX**2 + $normalY**2 + $normalZ**2)]
            set normalX [expr $normalX/$norm]
            set normalY [expr $normalY/$norm]
            set normalZ [expr $normalZ/$norm]

            # ax + by + cz = d
            set d         [expr $normalX*$xtop + $normalY*$ytop + $normalZ*$ztop]

            # find another point on the plane
            set P2x [expr $xtop + 1]
            set P2y [expr $ytop + 1]
            set P2z [expr ($d - $normalX*$P2x - $normalY*$P2y)/$normalZ]

            set VecX_x [expr $P2x - $P1x]
            set VecX_y [expr $P2y - $P1y]
            set VecX_z [expr $P2z - $P1z]
            set norm   [expr sqrt($VecX_x**2 + $VecX_y**2 + $VecX_z**2)]
            set VecX_x [expr $VecX_x/$norm]
            set VecX_y [expr $VecX_y/$norm]
            set VecX_z [expr $VecX_z/$norm]





            set transfTag [expr $soilfoundation_num_cells + $maxEleTag + $j*$pilenumelements + 1]
            eval "geomTransf $transfType $transfTag $VecX_x $VecX_y $VecX_z"
            # creating pile elements
            for {set i 1} {$i < $pilenumnodes} {incr i} {
                set node1  [expr $soilfoundation_num_nodes + $maxNodeTag + $j*$pilenumnodes + $i]
                set node2  [expr $soilfoundation_num_nodes + $maxNodeTag + $j*$pilenumnodes + $i + 1]
                set eleTag [expr $soilfoundation_num_cells + $maxEleTag + $j*$pilenumelements + $i]
                element dispBeamColumn $eleTag $node1 $node2 $numIntgrPts $secTag $transfTag 
                lappend pileElements   $eleTag
            }

            # creating soil-pile interface
            set interfaceElemsFile "$outputdir/PileInterfaceElements$j.dat"
            set connecFile        "$outputdir/PileInterfaceConnections$j.dat"
            if {[file exists $interfaceElemsFile] == 1} { file delete $interfaceElemsFile }
            if {[file exists $connecFile] == 1} { file delete $connecFile }

            set interfaceElems {}
            set first_beam_ele [expr $soilfoundation_num_cells + $maxEleTag + $j*$pilenumelements + 1]
            set last_beam_ele  [expr $soilfoundation_num_cells + $maxEleTag + $j*$pilenumelements + $pilenumelements]
            set interfaceElems [generateInterfacePoints -beamEleRange $first_beam_ele $last_beam_ele   -gPenalty -shape circle -nP $nPeri -nL $nLong -crdTransf $transfTag -radius $radius -penaltyParam 1.0e12 -file $interfaceElemsFile -connectivity $connecFile  ]
            puts "interfaceElems: $interfaceElems"



            incr j
        }







        # puts "writing pile nodes and elements to a file"
        # write pile nodes to a file
        set pileNodesFile "$outputdir/PileNodes.dat"
        writeNodesinFile $pileNodesFile $pileNodes


        # write beam elements to a file
        set beamElemsFile "$outputdir/PileElements.dat"
        writeElesinFile $beamElemsFile $pileElements

        puts "Pile elements are created"
    }
}
barrier
set maxNodeTag [getMaxNodeTag $np $pid]
set maxEleTag  [getMaxEleTag $np $pid]

# ============================================================================
# Creating column and foundation elements
# ============================================================================
if {$pid == 0} {
    if {$HaveStructure == "YES" } {
        if {$StructureType == "STEEL" || $StructureType == "CONCRETE"} {
            puts "Creating base columns and foundation elements"
        } else {
            puts "Base columns and foundation elements are skipped"
        }
    }
}
if {$HaveStructure == "YES" } {
    if {$pid == $structurecores && $structurecores > 0} {
        
        model BasicBuilder -ndm 3 -ndf 6
        set StrucFoundConnecElems {}
        set StrucFoundConnecNodes {}
        
        if {$StructureType == "STEEL" || $StructureType == "CONCRETE"} {set BaseColumnsFile "$outputdir/BaseColumnsNodes.dat"}
        
        # open BaseColumnsNodes.dat file
        set f [open $BaseColumnsFile "r"]
        set lines [split [read $f] "\n"]
        close $f

        set EmbeddingDepth 0.75
        set j 0
        foreach line $lines {
            # check if the line is empty
            if {[string length $line] == 0} {continue}
            incr j
            set nodeTag1 [lindex $line 0]
            set x        [lindex $line 1]
            set y        [lindex $line 2]
            set z        [lindex $line 3]
            node $nodeTag1 $x $y $z
            set nodeTag2 [expr $maxNodeTag + $j]
            set x        [expr $x]
            set y        [expr $y]
            set z        [expr $z - $EmbeddingDepth]
            node $nodeTag2 $x $y $z
            set numIntgrPts 3
            lappend StrucFoundConnecNodes $nodeTag1
            lappend StrucFoundConnecNodes $nodeTag2
            
            set nPeri         5
            set nLong         3
            set secTag        [expr $maxNodeTag + $j]
            set transfTag     [expr $maxNodeTag + $j]
            set diameter      0.25 ; 
            set radius        [expr $diameter/2.]
            set pi            3.141593
            set Epile         1e10
            set nu            0.3
            set Gpile         [expr $Epile/(2*(1+$nu))]
            set Area          [expr ($diameter**2)*$pi/2.]
            set Iy            [expr ($diameter**4)*$pi/64.]
            set Iz            [expr ($diameter**4)*$pi/64.]
            set J             [expr ($diameter**4)*$pi/32.]
            set transfType    "Linear"; # PDelta, Linear, Corotational


            section Elastic $secTag $Epile $Area $Iz $Iy $Gpile $J
            geomTransf $transfType $transfTag 1 0 0


            set eleTag [expr $maxEleTag + $j]
            element dispBeamColumn $eleTag $nodeTag2 $nodeTag1 $numIntgrPts $secTag $transfTag
            lappend StrucFoundConnecElems $eleTag

            # creating soil-pile interface
            set num [expr $j-1]
            set interfaceElemsFile "$outputdir/StructureFoundationInterfaceElements$num.dat"
            set connecFile         "$outputdir/StructureFoundationInterfaceConnections$num.dat"
            if {[file exists $interfaceElemsFile] == 1} { file delete $interfaceElemsFile }
            if {[file exists $connecFile] == 1} { file delete $connecFile }

            set interfaceElems {}
            set interfaceElems [generateInterfacePoints -beamEleRange $eleTag $eleTag -gPenalty -shape circle -nP $nPeri -nL $nLong -crdTransf $transfTag -radius $radius -penaltyParam 1.0e12 -file $interfaceElemsFile -connectivity $connecFile  ]
            set maxEleTag $interfaceElems
        }
        puts "Base columns and foundation elements are attached"

        # write pile nodes to a file
        set StrucFoundConnecNodesFile "$outputdir/StructureFoundationBeamNodes.dat"
        writeNodesinFile $StrucFoundConnecNodesFile $StrucFoundConnecNodes

        # write beam elements to a file
        set StrucFoundConnecElemsFile "$outputdir/StructureFoundationBeamElements.dat"
        writeElesinFile $StrucFoundConnecElemsFile $StrucFoundConnecElems
    }
}
barrier
# ============================================================================
# bulding PML layer
# ============================================================================
if {$pid == 0} {
    if {$HaveAbsorbingElements == "YES"} {
        puts "Creating Absorbing elements"
    } else {
        puts "Absorbing elements are skipped"
    }
}
if {$HaveAbsorbingElements == "YES" && $pid >= [expr  $structurecores + $regcores + $drmcores] } {
    # create PML material
    set gamma           0.5                    ;# --- Coefficient gamma, newmark gamma = 0.5
    set beta            0.25                   ;# --- Coefficient beta,  newmark beta  = 0.25
    set eta             [expr 1.0/12.]         ;# --- Coefficient eta,   newmark eta   = 1/12 
    set E               $E                     ;# --- Young's modulus
    set nu              $nu                    ;# --- Poisson's Ratio
    set rho             $rho                   ;# --- Density
    set EleType         6                      ;# --- Element type, See line
    set PML_L           $pmltotalthickness     ;# --- Thickness of the PML
    set afp             2.0                    ;# --- Coefficient m, typically m = 2
    set PML_Rcoef       1.0e-8                 ;# --- Coefficient R, typically R = 1e-8
    set RD_half_width_x [expr $llx/2.]         ;# --- Halfwidth of the regular domain in
    set RD_half_width_y [expr $lly/2.]         ;# --- Halfwidth of the regular domain in
    set RD_depth        [expr $llz/1.]         ;# --- Depth of the regular domain
    set pi              3.141593               ;# --- pi 

    set Damp_alpha      0.0    ;# --- Rayleigh damping coefficient alpha
    set Damp_beta       0.0    ;# --- Rayleigh damping coefficient beta

    
    if {$AbsorbingElements == "PML"} {
        puts "Absorbing elements are PML elements"
        model BasicBuilder -ndm 3 -ndf 9;
        set AbsorbingmatTag1 "$eta $beta $gamma $E $nu $rho $EleType $PML_L $afp $PML_Rcoef $RD_half_width_x $RD_half_width_y $RD_depth $Damp_alpha $Damp_beta"
        # set elementType "PMLVISCOUS"
        set elementType "PML"
        eval "source $meshdir/Nodes$pid.tcl"
        eval "source $meshdir/Elements$pid.tcl"

        # tie pml nodes to the regular nodes
        model BasicBuilder -ndm 3 -ndf 3;
        eval "source $meshdir/Boundary$pid.tcl"
    }
    if {$AbsorbingElements == "ASDA"} {
        puts "Absorbing elements are ASDA elements"
        model BasicBuilder -ndm 3 -ndf 3;
        set AbsorbingmatTag1 "$G $nu $rho";
        set elementType "ASDAbsorbingBoundary3D"
        # set elementType "PML"
        eval "source $meshdir/Nodes$pid.tcl"
        eval "source $meshdir/Elements$pid.tcl"
    }
    if {$AbsorbingElements == "Rayleigh"} {
        puts "Absorbing elements are normal elements"
        model BasicBuilder -ndm 3 -ndf 3;
        set AbsorbingmatTag1 "1 0.0 0.0 0.0";
        set elementType "stdBrick"
        eval "source $meshdir/Nodes$pid.tcl"
        eval "source $meshdir/Elements$pid.tcl"
        set rayleighalpha $Absorbing_rayleigh_alpha
        set rayleighbeta  $Absorbing_rayleigh_beta

    }
}
barrier


# ============================================================================
# creating fixities
# ============================================================================
if {$HaveAbsorbingElements == "YES"} {
    set totalThickness [expr $numdrmlayers*$drmthicknessx + $numpmllayers*$pmlthicknessx]
    if {$AbsorbingElements == "PML"} {
        fixX [expr -$lx/2. - $totalThickness] 1 1 1 1 1 1 1 1 1;
        fixX [expr  $lx/2. + $totalThickness] 1 1 1 1 1 1 1 1 1;
        fixY [expr -$ly/2. - $totalThickness] 1 1 1 1 1 1 1 1 1;
        fixY [expr  $ly/2. + $totalThickness] 1 1 1 1 1 1 1 1 1;
        fixZ [expr -$lz/1. - $totalThickness] 1 1 1 1 1 1 1 1 1;
    }
    if {$AbsorbingElements == "ASDA" || $AbsorbingElements == "Normal"} {
        fixX [expr -$lx/2. - $totalThickness] 1 1 1;
        fixX [expr  $lx/2. + $totalThickness] 1 1 1;
        fixY [expr -$ly/2. - $totalThickness] 1 1 1;
        fixY [expr  $ly/2. + $totalThickness] 1 1 1;
        fixZ [expr -$lz/1. - $totalThickness] 1 1 1;
    }
    puts "Boundary conditions are applied to the absorbing Layer nodes"
} else {
    set totalThickness [expr $numdrmlayers*$drmthicknessx]
    fixX [expr -$lx/2. - $totalThickness] 1 1 1;
    fixX [expr  $lx/2. + $totalThickness] 1 1 1;
    fixY [expr -$ly/2. - $totalThickness] 1 1 1;
    fixY [expr  $ly/2. + $totalThickness] 1 1 1;
    fixZ [expr -$lz/1. - $totalThickness] 1 1 1;
    puts "Boundary conditions are applied to the DRM Layer nodes"
}
barrier

# ============================================================================
# printing model information for debugging
# ============================================================================
# print "modelInfo$pid"


# ============================================================================
# Gravity analysis
# ============================================================================
domainChange
constraints      Plain
numberer         ParallelRCM
system           Mumps -ICNTL14 400
test             NormDispIncr 1e-4 10 2
algorithm        Linear -factorOnce 
# algorithm        ModifiedNewton -factoronce
# algorithm        Newton
integrator       Newmark 0.5 0.25
analysis         Transient


barrier
loadConst -time 0.0
wipeAnalysis
puts "Gravity analysis is done"
# ============================================================================
# Gravirty recorders
# ============================================================================
# if {$pid >= $structurecores && $pid < [expr $regcores + $structurecores]} {
#     eval "recorder Node -file $outputdir/GravityNodeDisp$pid.out  -node $recordList -dof 1 2 3 disp"
#     eval "recorder Node -file $outputdir/GravityNodeAccl$pid.out  -node $recordList -dof 1 2 3 accel"
#     eval "recorder Node -file $outputdir/GravityNodeVelo$pid.out  -node $recordList -dof 1 2 3 vel"
#     eval "recorder Element -file $outputdir/GravityElementStress$pid.out -ele $elerecordList stresses"
#     eval "recorder Element -file $outputdir/GravityElementStrain$pid.out -ele $elerecordList strains"

#     # print recordlist and elerecordlist to a file
#     set f [open "$outputdir/nodeOuputTags$pid.out" "w+"]
#     puts $f "$recordList"
#     close $f
#     set f [open "$outputdir/eleOuputTags$pid.out" "w+"]
#     puts $f "$elerecordList"
#     close $f
# }
# record
# record
# barrier
# remove recorders

# ============================================================================
# Post gravity settings
# ============================================================================
# ASDA Absorbing layer settings
if {$pid >= [expr $regcores +$structurecores] && $pid < [expr $regcores + $drmcores + $structurecores]} {
    if {$pid >= [expr  $structurecores + $regcores + $drmcores] && $AbsorbingElements == "ASDA" } {
        set abs_elements [getEleTags]
        eval "setParameter -val 1 -ele $abs_elements stage"
    }
}

# # set the initial displacements to zero
# foreach node [getNodeTags] {
#     foreach dof {1 2 3} {
#         setNodeAccel $node $dof 0.0 -commit
#         setNodeVel   $node $dof 0.0 -commit
#         setNodeDisp  $node $dof 0.0 -commit
#     }
# }
# ============================================================================
# loading 
# ============================================================================

# if {$pid < $regcores} {  
if {$pid>=$regcores && $pid < [expr $regcores + $drmcores] } {
    set prop1 "$DRM_factor $crd_scale $distance_tolerance $do_coordinate_transformation"; # factor crd_scale distance_tolerance do_coordinate_transformation
    set prop2 "$T00 $T01 $T02"; # T00, T01, T02
    set prop3 "$T10 $T11 $T12"; # T10, T11, T12
    set prop4 "$T20 $T21 $T22"; # T20, T21, T22
    set prop5 "$originX $originY $originZ"; # x00, x01, x02

    set DRMtag  1
    eval "pattern H5DRM $DRMtag $DRMFile $prop1 $prop2 $prop3 $prop4 $prop5"
}

# # ============================================================================
# # recorders
# # ============================================================================
# find the  nodetag with 0.0 0.0 0.0
if {$pid < $regcores} {
    set foundflag "No"
    foreach node [getNodeTags] {
        set coords [nodeCoord $node]
        set diff1 [expr abs([lindex $coords 0])]
        set diff2 [expr abs([lindex $coords 1])]
        set diff3 [expr abs([lindex $coords 2])]
        if { $diff1 < 1e-6 && $diff2 < 1e-6 && $diff3 < 1e-6 } {
            set recordNode $node
            set foundflag "Yes"
            break
        }
    }
    if {$foundflag == "Yes"} {
        recorder Node -file $outputdir/Disp$pid.out  -time -dT $Analysis_record_dt -node $recordNode -dof 1 2 3 disp
        recorder Node -file $outputdir/Accel$pid.out -time -dT $Analysis_record_dt -node $recordNode -dof 1 2 3 accel
        recorder Node -file $outputdir/Vel$pid.out   -time -dT $Analysis_record_dt -node $recordNode -dof 1 2 3 vel
    }
}
record

# ============================================================================
# Analysis 
# ============================================================================
domainChange

# constraints      Plain
constraints      Plain
numberer         ParallelRCM
system           Mumps -ICNTL14 400
test             NormDispIncr 1e-4 10 2
algorithm        Linear -factorOnce 
integrator       Newmark 0.5 0.25
analysis         Transient

if {$AbsorbingElements == "Rayleigh"} {
    puts "Rayleigh damping is applied"
    rayleigh $rayleighalpha $rayleighbeta 0.0 0.0
}
set startTime [clock milliseconds]

set dT $Analysis_dt
while { [getTime] < $Analysis_duration } {
    if {$pid ==0 } {puts "Time: [getTime]";}
    analyze 1 $dT
}

set endTime [clock milliseconds]
set elapsedTime [expr {$endTime - $startTime}]
puts "Elapsed time: [expr $elapsedTime/1000.] seconds in $pid"



# if {$pid == 0} {
#     puts "natural frequency of the pile: $f Hz (assuming cantilever beam)"
#     puts "wavelegth: [expr $Vs/$f] m"
#     puts "Vs: $Vs"
# }
wipeAnalysis
remove recorders
remove loadPattern 2