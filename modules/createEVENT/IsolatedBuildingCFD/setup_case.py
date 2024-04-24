"""
This script writes BC and initial condition, and setups the OpenFoam case 
directory.

"""
import numpy as np
import sys
import os
import json
import numpy as np
import foam_dict_reader as foam
from stl import mesh


def write_block_mesh_dict(input_json_path, template_dict_path, case_path):

    #Read JSON data    
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)
      
    # Returns JSON object as a dictionary
    mesh_data = json_data["blockMeshParameters"]
    geom_data = json_data['GeometricData']
    boundary_data = json_data['boundaryConditions']

    normalization_type = geom_data['normalizationType']
    origin = np.array(geom_data['origin'])
    scale =  geom_data['geometricScale']
    H = geom_data['buildingHeight']/scale #convert to model-scale
    
    Lx = geom_data['domainLength']
    Ly = geom_data['domainWidth']
    Lz = geom_data['domainHeight']
    Lf = geom_data['fetchLength']
    
    x_cells = mesh_data['xNumCells']
    y_cells = mesh_data['yNumCells']
    z_cells = mesh_data['zNumCells']
    
    x_grading = mesh_data['xGrading']
    y_grading = mesh_data['yGrading']
    z_grading = mesh_data['zGrading']

    bc_map = {"slip": 'wall', "cyclic": 'cyclic', "noSlip": 'wall', 
                     "symmetry": 'symmetry', "empty": 'empty', "TInf": 'patch', 
                     "MeanABL": 'patch', "Uniform": 'patch', "zeroPressureOutlet": 'patch',
                     "roughWallFunction": 'wall',"smoothWallFunction": 'wall'}



    inlet_type = bc_map[boundary_data['inletBoundaryCondition']]
    outlet_type = bc_map[boundary_data['outletBoundaryCondition']]
    ground_type = bc_map[boundary_data['groundBoundaryCondition']]  
    top_type = bc_map[boundary_data['topBoundaryCondition']]
    front_type = bc_map[boundary_data['sidesBoundaryCondition']]
    back_type = bc_map[boundary_data['sidesBoundaryCondition']]
    
    length_unit = json_data['lengthUnit']

    if normalization_type == "Relative":
        Lx = Lx*H
        Ly = Ly*H
        Lz = Lz*H
        Lf = Lf*H
        origin = origin*H
    
    x_min = -Lf - origin[0]
    y_min = -Ly/2.0 - origin[1]
    z_min =  0.0 - origin[2]

    x_max = x_min + Lx
    y_max = y_min + Ly
    z_max = z_min + Lz

    #Open the template blockMeshDict (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/blockMeshDictTemplate", "r")

    #Export to OpenFOAM probe format
    dict_lines = dict_file.readlines()
    dict_file.close()
    

    dict_lines[17] = "\txMin\t\t{:.4f};\n".format(x_min)
    dict_lines[18] = "\tyMin\t\t{:.4f};\n".format(y_min)
    dict_lines[19] = "\tzMin\t\t{:.4f};\n".format(z_min)

    dict_lines[20] = "\txMax\t\t{:.4f};\n".format(x_max)
    dict_lines[21] = "\tyMax\t\t{:.4f};\n".format(y_max)
    dict_lines[22] = "\tzMax\t\t{:.4f};\n".format(z_max)


    dict_lines[23] = "\txCells\t\t{:d};\n".format(x_cells)
    dict_lines[24] = "\tyCells\t\t{:d};\n".format(y_cells)
    dict_lines[25] = "\tzCells\t\t{:d};\n".format(z_cells)

    dict_lines[26] = "\txGrading\t{:.4f};\n".format(x_grading)
    dict_lines[27] = "\tyGrading\t{:.4f};\n".format(y_grading)
    dict_lines[28] = "\tzGrading\t{:.4f};\n".format(z_grading)

    convert_to_meters = 1.0

    if length_unit=='m':
        convert_to_meters = 1.0
    elif length_unit=='cm':
        convert_to_meters = 0.01
    elif length_unit=='mm':
        convert_to_meters = 0.001
    elif length_unit=='ft':
        convert_to_meters = 0.3048
    elif length_unit=='in':
        convert_to_meters = 0.0254

    dict_lines[31] = "convertToMeters {:.4f};\n".format(convert_to_meters)
    dict_lines[61] = "        type {};\n".format(inlet_type)
    dict_lines[70] = "        type {};\n".format(outlet_type)
    dict_lines[79] = "        type {};\n".format(ground_type)
    dict_lines[88] = "        type {};\n".format(top_type)
    dict_lines[97] = "        type {};\n".format(front_type)
    dict_lines[106] = "        type {};\n".format(back_type)

    
    write_file_name = case_path + "/system/blockMeshDict"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()



def write_building_stl_file(input_json_path, case_path):
    
    #Read JSON data    
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)
      
    geom_data = json_data['GeometricData']

    if geom_data["buildingShape"] == "Complex":
        import_building_stl_file(input_json_path, case_path)
        return  

    #Else create the STL file
    scale =  geom_data['geometricScale']
    length_unit =  json_data['lengthUnit']

    convert_to_meters = 1.0

    if length_unit=='m':
        convert_to_meters = 1.0
    elif length_unit=='cm':
        convert_to_meters = 0.01
    elif length_unit=='mm':
        convert_to_meters = 0.001
    elif length_unit=='ft':
        convert_to_meters = 0.3048
    elif length_unit=='in':
        convert_to_meters = 0.0254
    
    #Convert from full-scale to model-scale
    B = convert_to_meters*geom_data['buildingWidth']/scale
    D = convert_to_meters*geom_data['buildingDepth']/scale
    H = convert_to_meters*geom_data['buildingHeight']/scale
    
    normalization_type = geom_data['normalizationType']

    origin = np.array(geom_data['origin'])
    wind_dxn = geom_data['windDirection']

    if normalization_type == "Relative":
        origin = origin*H
    

    wind_dxn_rad = np.deg2rad(wind_dxn)
    epsilon = 1.0e-5 
     
    # Define the 8 vertices of the building
    vertices = np.array([[-D/2.0, -B/2.0, -epsilon],
                         [+D/2.0, -B/2.0, -epsilon],
                         [+D/2.0, +B/2.0, -epsilon],
                         [-D/2.0, +B/2.0, -epsilon],
                         [-D/2.0, -B/2.0, +H],
                         [+D/2.0, -B/2.0, +H],
                         [+D/2.0, +B/2.0, +H],
                         [-D/2.0, +B/2.0, +H]])

    n_vertices = np.shape(vertices)[0]

    #The default coordinate system is building center. 
    #Transform the preferred origin
    vertices = vertices - origin
    
    #Transform transform the vertices to account the wind direction. 
    trans_vertices = np.zeros((n_vertices, 3))
    trans_vertices[:,2] = vertices[:,2]
    
    t_matrix  = np.array([[np.cos(wind_dxn_rad), -np.sin(wind_dxn_rad)], 
                          [np.sin(wind_dxn_rad),  np.cos(wind_dxn_rad)]])
    
    for i in range(n_vertices):
        trans_vertices[i,0:2]  = np.matmul(t_matrix, vertices[i,0:2])
        
        
    # Define the 12 triangles composing the rectangular building
    faces = np.array([\
        [0,3,1],
        [1,3,2],
        [0,4,7],
        [0,7,3],
        [4,5,6],
        [4,6,7],
        [5,1,2],
        [5,2,6],
        [2,3,6],
        [3,7,6],
        [0,1,5],
        [0,5,4]])
    
    # Create the mesh
    bldg = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            bldg.vectors[i][j] = trans_vertices[f[j],:]
    
    # Write the mesh to file "building.stl"
    fmt = mesh.stl.Mode.ASCII # binary or ASCII format 
    bldg.save(case_path + '/constant/geometry/building.stl', mode=fmt)

def import_building_stl_file(input_json_path, case_path):
    #Read JSON data    
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    if json_data["GeometricData"]["buildingShape"] == "Simple":
        return  

    # Returns JSON object as a dictionary
    stl_path = json_data["GeometricData"]["importedSTLPath"]
    scale_factor = json_data["GeometricData"]["stlScaleFactor"]
    recenter = json_data["GeometricData"]["recenterToOrigin"]
    use_stl_dimension = json_data["GeometricData"]["useSTLDimensions"]
    account_wind_direction = json_data["GeometricData"]["accountWindDirection"]
    origin = np.array(json_data["GeometricData"]['origin'])
    wind_dxn = json_data["GeometricData"]['windDirection']
    wind_dxn_rad = np.deg2rad(wind_dxn)

    # Using an existing closed stl file:
    bldg_mesh = mesh.Mesh.from_file(stl_path)

    min_x = bldg_mesh.x.min()
    max_x = bldg_mesh.x.max()
    min_y = bldg_mesh.y.min()
    max_y = bldg_mesh.y.max()
    min_z = bldg_mesh.z.min()
    max_z = bldg_mesh.z.max()

    # if use_stl_dimension:
    # Data to be written
    stl_summary = {
        "xMin": float(min_x),
        "xMax": float(max_x),
        "yMin": float(min_y),
        "yMax": float(max_y),
        "zMin": float(min_z),
        "zMax": float(max_z)
    }

    # Serializing json
    json_object = json.dumps(stl_summary, indent=4)
    
    # Writing to sample.json
    with open(input_json_path + "/stlGeometrySummary.json", "w", encoding='utf-8') as outfile:
        outfile.write(json_object)
    
    #Translate the bottom center to origin
    if recenter:
        t = np.array([-((max_x - min_x)/2.0 + min_x), -((max_y - min_y)/2.0 + min_y), -min_z]) - origin/scale_factor
        bldg_mesh.translate(t)
    
    #Account wind direction by rotation
    if account_wind_direction:
        #Rotate about z-axis
        bldg_mesh.rotate(np.array([0, 0, 1.0]), wind_dxn_rad)

    # Scale the mesh 
    bldg_mesh.vectors *= scale_factor

    # Write the mesh to file "building.stl"
    fmt = mesh.stl.Mode.ASCII # binary or ASCII format 
    bldg_mesh.save(case_path + '/constant/geometry/building.stl', mode=fmt)

def write_surfaceFeaturesDict_file(input_json_path, template_dict_path, case_path):
    
  #Read JSON data    
  with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
      json_data =  json.load(json_file)
    
  # Returns JSON object as a dictionary
  domain_data = json_data["snappyHexMeshParameters"]
  building_stl_name = domain_data['buildingSTLName']

  #Open the template blockMeshDict (OpenFOAM file) for manipulation
  dict_file = open(template_dict_path + "/surfaceFeaturesDictTemplate", "r")

  #Export to OpenFOAM probe format
  dict_lines = dict_file.readlines()
  dict_file.close()
  
  
  #Write 'addLayers' switch    
  start_index = foam.find_keyword_line(dict_lines, "surfaces")
  dict_lines[start_index] = "surfaces  (\"{}.stl\");\n".format(building_stl_name)
  
  
  #Write edited dict to file
  write_file_name = case_path + "/system/surfaceFeaturesDict"
  
  if os.path.exists(write_file_name):
      os.remove(write_file_name)
  
  output_file = open(write_file_name, "w+")
  for line in dict_lines:
      output_file.write(line)
  output_file.close()




def write_snappy_hex_mesh_dict(input_json_path, template_dict_path, case_path):

    #Read JSON data    
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)
      
    # Returns JSON object as a dictionary
    mesh_data = json_data["snappyHexMeshParameters"]

    geom_data = json_data['GeometricData']

    scale =  geom_data['geometricScale']
    H = geom_data['buildingHeight']/scale #convert to model-scale
    
    Lx = geom_data['domainLength']
    Ly = geom_data['domainWidth']
    Lz = geom_data['domainHeight']
    Lf = geom_data['fetchLength']
    
    normalization_type = geom_data['normalizationType']
    origin = np.array(geom_data['origin'])
    
    building_stl_name = mesh_data['buildingSTLName']
    num_cells_between_levels = mesh_data['numCellsBetweenLevels']
    resolve_feature_angle = mesh_data['resolveFeatureAngle']
    num_processors = mesh_data['numProcessors']
    
    refinement_boxes = mesh_data['refinementBoxes']
    
    add_surface_refinement = mesh_data['addSurfaceRefinement']
    surface_refinement_level = mesh_data['surfaceRefinementLevel']
    surface_refinement_distance = mesh_data['surfaceRefinementDistance']
    refinement_surface_name = mesh_data['refinementSurfaceName']
    
    add_edge_refinement = mesh_data['addEdgeRefinement']
    edge_refinement_level = mesh_data['edgeRefinementLevel']
    refinement_edge_name = mesh_data['refinementEdgeName']
    
    add_prism_layers = mesh_data['addPrismLayers']
    number_of_prism_layers = mesh_data['numberOfPrismLayers']
    prism_layer_expansion_ratio = mesh_data['prismLayerExpansionRatio']
    final_prism_layer_thickness = mesh_data['finalPrismLayerThickness']
    prism_layer_surface_name = mesh_data['prismLayerSurfaceName']
    prism_layer_relative_size = "on"  

    if normalization_type == "Relative":
        Lx = Lx*H
        Ly = Ly*H
        Lz = Lz*H
        Lf = Lf*H
        origin = origin*H

        for i in range(len(refinement_boxes)):
            for j in range(2, 8, 1):
                refinement_boxes[i][j] = refinement_boxes[i][j]*H
                
        surface_refinement_distance = surface_refinement_distance*H
    
    x_min = -Lf - origin[0]
    y_min = -Ly/2.0 - origin[1]
    z_min =  0.0 - origin[2]

    x_max = x_min + Lx
    y_max = y_min + Ly
    z_max = z_min + Lz    
    
    inside_point  = [x_min + Lf/2.0, (y_min + y_max)/2.0, H]


    #Open the template blockMeshDict (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/snappyHexMeshDictTemplate", "r")

    #Export to OpenFOAM probe format
    dict_lines = dict_file.readlines()
    dict_file.close()
    
    
    #Write 'addLayers' switch    
    start_index = foam.find_keyword_line(dict_lines, "addLayers")
    dict_lines[start_index] = "addLayers\t{};\n".format("on" if add_prism_layers else "off")
    

    ###################### Edit Geometry Section ##############################
    
    #Add refinment box geometry
    start_index = foam.find_keyword_line(dict_lines, "geometry") + 2 
    added_part = ""
    n_boxes  = len(refinement_boxes)
    for i in range(n_boxes):
        added_part += "    {}\n".format(refinement_boxes[i][0])
        added_part += "    {\n"
        added_part += "         type searchableBox;\n"
        added_part += "         min ({:.4f} {:.4f} {:.4f});\n".format(refinement_boxes[i][2], refinement_boxes[i][3], refinement_boxes[i][4])
        added_part += "         max ({:.4f} {:.4f} {:.4f});\n".format(refinement_boxes[i][5], refinement_boxes[i][6], refinement_boxes[i][7])
        added_part += "    }\n"
        
    dict_lines.insert(start_index, added_part)
       
    #Add building stl geometry
    start_index = foam.find_keyword_line(dict_lines, "geometry") + 2 
    added_part = ""
    added_part += "    {}\n".format(building_stl_name)
    added_part += "    {\n"
    added_part += "         type triSurfaceMesh;\n"
    added_part += "         file \"{}.stl\";\n".format(building_stl_name)
    added_part += "    }\n"
    
    dict_lines.insert(start_index, added_part)
    
    
    ################# Edit castellatedMeshControls Section ####################

    #Write 'nCellsBetweenLevels'     
    start_index = foam.find_keyword_line(dict_lines, "nCellsBetweenLevels")
    dict_lines[start_index] = "    nCellsBetweenLevels {:d};\n".format(num_cells_between_levels)

    #Write 'resolveFeatureAngle'     
    start_index = foam.find_keyword_line(dict_lines, "resolveFeatureAngle")
    dict_lines[start_index] = "    resolveFeatureAngle {:d};\n".format(resolve_feature_angle)

    #Write 'insidePoint'     
    start_index = foam.find_keyword_line(dict_lines, "insidePoint")
    dict_lines[start_index] = "    insidePoint ({:.4f} {:.4f} {:.4f});\n".format(inside_point[0], inside_point[1], inside_point[2])


    #For compatability with OpenFOAM-9 and older
    start_index = foam.find_keyword_line(dict_lines, "locationInMesh")
    dict_lines[start_index] = "    locationInMesh ({:.4f} {:.4f} {:.4f});\n".format(inside_point[0], inside_point[1], inside_point[2])


    #Add refinment edge 
    if add_edge_refinement: 
        start_index = foam.find_keyword_line(dict_lines, "features") + 2 
        added_part  = ""
        added_part += "         {\n"
        added_part += "             file \"{}.eMesh\";\n".format(refinement_edge_name)
        added_part += "             level {};\n".format(edge_refinement_level)
        added_part += "         }\n"
        
        dict_lines.insert(start_index, added_part)
        
    #Add refinement surface
    if add_surface_refinement:         
        start_index = foam.find_keyword_line(dict_lines, "refinementSurfaces") + 2 
        added_part = ""
        added_part += "         {}\n".format(refinement_surface_name)
        added_part += "         {\n"
        added_part += "             level ({} {});\n".format(surface_refinement_level, surface_refinement_level)
        added_part += "             patchInfo\n"
        added_part += "             {\n"
        added_part += "                 type wall;\n"
        added_part += "             }\n"
        added_part += "         }\n"
        
        dict_lines.insert(start_index, added_part)
        
    #Add surface refinement around the building as a refinement region
    # if surface_refinement_level > refinement_boxes[-1][1]:
    added_part = ""
    added_part += "         {}\n".format(refinement_surface_name)
    added_part += "         {\n"
    added_part += "             mode   distance;\n"
    added_part += "             levels  (({:.4f} {}));\n".format(surface_refinement_distance, refinement_boxes[-1][1] + 1)
    added_part += "         }\n"
                
    start_index = foam.find_keyword_line(dict_lines, "refinementRegions") + 2 
    dict_lines.insert(start_index, added_part)
    
    #Add box refinements 
    added_part = ""
    for i in range(n_boxes):
        added_part += "         {}\n".format(refinement_boxes[i][0])
        added_part += "         {\n"
        added_part += "             mode   inside;\n"
        added_part += "             level  {};\n".format(refinement_boxes[i][1])
        added_part += "         }\n"
                
    start_index = foam.find_keyword_line(dict_lines, "refinementRegions") + 2 
    dict_lines.insert(start_index, added_part)
    
    
    ####################### Edit PrismLayer Section ##########################
    
    #Add surface layers (prism layers)
    added_part = ""
    added_part += "         \"{}\"\n".format(prism_layer_surface_name)
    added_part += "         {\n"
    added_part += "             nSurfaceLayers {};\n".format(number_of_prism_layers)
    added_part += "         }\n"
    
    start_index = foam.find_keyword_line(dict_lines, "layers") + 2 
    dict_lines.insert(start_index, added_part)

    #Write 'relativeSizes'     
    start_index = foam.find_keyword_line(dict_lines, "relativeSizes")
    dict_lines[start_index] = "    relativeSizes {};\n".format(prism_layer_relative_size)

    #Write 'expansionRatio'     
    start_index = foam.find_keyword_line(dict_lines, "expansionRatio")
    dict_lines[start_index] = "    expansionRatio {:.4f};\n".format(prism_layer_expansion_ratio)
    
    #Write 'finalLayerThickness'     
    start_index = foam.find_keyword_line(dict_lines, "finalLayerThickness")
    dict_lines[start_index] = "    finalLayerThickness {:.4f};\n".format(final_prism_layer_thickness)    
    
    
    
    #Write edited dict to file
    write_file_name = case_path + "/system/snappyHexMeshDict"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()



def write_U_file(input_json_path, template_dict_path, case_path):

    #Read JSON data    
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data["boundaryConditions"]    
    wind_data = json_data["windCharacteristics"]
    
      
    inlet_BC_type =  boundary_data['inletBoundaryCondition']
    top_BC_type = boundary_data['topBoundaryCondition']
    sides_BC_type = boundary_data['sidesBoundaryCondition']
    building_BC_type = boundary_data['buildingBoundaryCondition']
 
    wind_speed = wind_data['referenceWindSpeed']
    building_height = wind_data['referenceHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']


    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/UFileTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    ##################### Internal Field #########################
    #Initialize the internal fields frow a lower velocity to avoid Courant number 
    #instability when the solver starts. Now %10 of roof-height wind speed is set      
    start_index = foam.find_keyword_line(dict_lines, "internalField") 
    dict_lines[start_index] = "internalField   uniform ({:.4f} 0 0);\n".format(0.1*wind_speed)


    ###################### Inlet BC ##############################  
    #Write uniform
    start_index = foam.find_keyword_line(dict_lines, "inlet") + 2 

    if inlet_BC_type == "Uniform":    
        added_part = ""
        added_part += "\t type \t fixedValue;\n"
        added_part += "\t value \t uniform ({:.4f} 0 0);\n".format(wind_speed)
        
    if inlet_BC_type == "MeanABL":    
        added_part = ""
        added_part += "\t type \t atmBoundaryLayerInletVelocity;\n"
        added_part += "\t Uref \t {:.4f};\n".format(wind_speed)
        added_part += "\t Zref \t {:.4f};\n".format(building_height)
        added_part += "\t zDir \t (0.0 0.0 1.0);\n"
        added_part += "\t flowDir \t (1.0 0.0 0.0);\n"
        added_part += "\t z0 uniform \t {:.4e};\n".format(roughness_length)
        added_part += "\t zGround \t uniform 0.0;\n"
        
    if inlet_BC_type == "Place holder for TInf":    
        added_part = ""
        
    dict_lines.insert(start_index, added_part)

    ###################### Outlet BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "outlet") + 2 
    added_part = ""
    added_part += "\t type \t pressureInletOutletVelocity;\n"
    added_part += "\t value \t uniform ({:.4f} 0 0);\n".format(wind_speed)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Ground BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "ground") + 2 
    added_part = ""
    added_part += "\t type \t uniformFixedValue;\n"
    added_part += "\t value \t uniform (0 0 0);\n"
    added_part += "\t uniformValue \t constant (0 0 0);\n"
    
    dict_lines.insert(start_index, added_part)
    
    
    ###################### Top BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "top") + 2 
    added_part = ""
    added_part += "\t type    {};\n".format(top_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Front BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "front") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Back BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "back") + 2 
    added_part = ""
    added_part += "\t type    {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Building BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "building") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format("noSlip")
    
    dict_lines.insert(start_index, added_part)
    
    
    #Write edited dict to file
    write_file_name = case_path + "/0/U"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+", encoding='utf-8')
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    

def write_p_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data["boundaryConditions"]
      
    sides_BC_type = boundary_data['sidesBoundaryCondition']
    top_BC_type = boundary_data['topBoundaryCondition']


    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/pFileTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    
    #BC and initial condition
    p0 = 0.0; 


    ##################### Internal Field #########################
    
    start_index = foam.find_keyword_line(dict_lines, "internalField") 
    dict_lines[start_index] = "internalField   uniform {:.4f};\n".format(p0)


    ###################### Inlet BC ##############################  
    #Write uniform
    start_index = foam.find_keyword_line(dict_lines, "inlet") + 2 
    added_part = ""
    added_part += "\t type \t zeroGradient;\n"
    
    dict_lines.insert(start_index, added_part)

    ###################### Outlet BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "outlet") + 2 
    added_part = ""
    added_part += "\t type \t  uniformFixedValue;\n"
    added_part += "\t uniformValue \t constant {:.4f};\n".format(p0)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Ground BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "ground") + 2 
    added_part = ""
    added_part += "\t type \t zeroGradient;\n"
    
    dict_lines.insert(start_index, added_part)
    
    
    ###################### Top BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "top") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(top_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Front BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "front") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Back BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "back") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Building BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "building") + 2 
    added_part = ""
    added_part += "\t type  \t zeroGradient;\n"
    
    dict_lines.insert(start_index, added_part)
    
    
    #Write edited dict to file
    write_file_name = case_path + "/0/p"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    
def write_nut_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data["boundaryConditions"]
    wind_data = json_data["windCharacteristics"]
          
    sides_BC_type = boundary_data['sidesBoundaryCondition']
    top_BC_type = boundary_data['topBoundaryCondition']
    ground_BC_type = boundary_data['groundBoundaryCondition']
    building_BC_type = boundary_data['buildingBoundaryCondition']

    # wind_speed = wind_data['roofHeightWindSpeed']
    # building_height = wind_data['buildingHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']
    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/nutFileTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    
    #BC and initial condition
    nut0 = 0.0 

    ##################### Internal Field #########################
    
    start_index = foam.find_keyword_line(dict_lines, "internalField") 
    dict_lines[start_index] = "internalField   uniform {:.4f};\n".format(nut0)


    ###################### Inlet BC ##############################  
    #Write uniform
    start_index = foam.find_keyword_line(dict_lines, "inlet") + 2 
    added_part  = ""
    added_part += "\t type \t zeroGradient;\n"
    
    dict_lines.insert(start_index, added_part)

    ###################### Outlet BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "outlet") + 2 
    added_part = ""
    added_part += "\t type \t uniformFixedValue;\n"
    added_part += "\t uniformValue \t constant {:.4f};\n".format(nut0)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Ground BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "ground") + 2 
    
    if ground_BC_type == "noSlip": 
        added_part = ""
        added_part += "\t type \t zeroGradient;\n"
    
    if ground_BC_type == "roughWallFunction": 
        added_part = ""
        added_part += "\t type \t nutkAtmRoughWallFunction;\n"
        added_part += "\t z0  \t  uniform {:.4e};\n".format(roughness_length)
        added_part += "\t value \t uniform 0.0;\n"

    if ground_BC_type == "smoothWallFunction": 
        added_part = ""
        added_part += "\t type \t nutUSpaldingWallFunction;\n"
        added_part += "\t value \t uniform 0;\n"


    dict_lines.insert(start_index, added_part)
    
    
    ###################### Top BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "top") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(top_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Front BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "front") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Back BC ################################  
    
    start_index = foam.find_keyword_line(dict_lines, "back") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Building BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "building") + 2 
    
    if building_BC_type == "noSlip": 
        added_part = ""
        added_part += "\t type \t fixedValue;\n"
        added_part += "\t value \t uniform 0;\n"
    
    if building_BC_type == "smoothWallFunction": 
        added_part = ""
        added_part += "\t type \t nutUSpaldingWallFunction;\n"
        added_part += "\t value \t uniform 0;\n"
    
    if building_BC_type == "roughWallFunction": 
        added_part = ""
        added_part += "\t type \t nutkRoughWallFunction;\n"
        added_part += "\t Ks \t uniform 1e-5;\n"
        added_part += "\t Cs \t uniform 0.5;\n"
        added_part += "\t value \t uniform 0;\n"
    
    dict_lines.insert(start_index, added_part)
    
    #Write edited dict to file
    write_file_name = case_path + "/0/nut"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()

def write_epsilon_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data["boundaryConditions"]
    wind_data = json_data["windCharacteristics"]
      
    
    sides_BC_type = boundary_data['sidesBoundaryCondition']
    top_BC_type = boundary_data['topBoundaryCondition']
    ground_BC_type = boundary_data['groundBoundaryCondition']
    building_BC_type = boundary_data['buildingBoundaryCondition']

    wind_speed = wind_data['referenceWindSpeed']
    building_height = wind_data['referenceHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']
    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/epsilonFileTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    
    #BC and initial condition
    epsilon0 = 0.01 

    ##################### Internal Field #########################
    
    start_index = foam.find_keyword_line(dict_lines, "internalField") 
    dict_lines[start_index] = "internalField   uniform {:.4f};\n".format(epsilon0)


    ###################### Inlet BC ##############################  
    #Write uniform
    start_index = foam.find_keyword_line(dict_lines, "inlet") + 2 
    added_part  = ""
    added_part += "\t type \t atmBoundaryLayerInletEpsilon;\n"
    added_part += "\t Uref \t {:.4f};\n".format(wind_speed)
    added_part += "\t Zref \t {:.4f};\n".format(building_height)
    added_part += "\t zDir \t (0.0 0.0 1.0);\n"
    added_part += "\t flowDir \t (1.0 0.0 0.0);\n"
    added_part += "\t z0 \t  uniform {:.4e};\n".format(roughness_length)
    added_part += "\t zGround \t uniform 0.0;\n"
    
    dict_lines.insert(start_index, added_part)

    ###################### Outlet BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "outlet") + 2 
    added_part = ""
    added_part += "\t type \t inletOutlet;\n"
    added_part += "\t inletValue \t uniform {:.4f};\n".format(epsilon0)
    added_part += "\t value \t uniform {:.4f};\n".format(epsilon0)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Ground BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "ground") + 2 
    
    if ground_BC_type == "noSlip": 
        added_part = ""
        added_part += "\t type \t zeroGradient;\n"
    
    if ground_BC_type == "roughWallFunction": 
        added_part = ""
        added_part += "\t type \t epsilonWallFunction;\n"
        added_part += "\t Cmu \t {:.4f};\n".format(0.09)
        added_part += "\t kappa \t {:.4f};\n".format(0.41)
        added_part += "\t E \t {:.4f};\n".format(9.8)
        added_part += "\t value \t uniform {:.4f};\n".format(epsilon0)
    
    #Note:  Should be replaced with smooth wall function for epsilon,
    #       now the same with rough wall function.
    if ground_BC_type == "smoothWallFunction": 
        added_part = ""
        added_part += "\t type \t epsilonWallFunction;\n"
        added_part += "\t Cmu \t {:.4f};\n".format(0.09)
        added_part += "\t kappa \t {:.4f};\n".format(0.41)
        added_part += "\t E \t {:.4f};\n".format(9.8)
        added_part += "\t value \t uniform {:.4f};\n".format(epsilon0)
    dict_lines.insert(start_index, added_part)
    
    
    ###################### Top BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "top") + 2 
    added_part = ""
    added_part += "\t type  \t  {};\n".format(top_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Front BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "front") + 2 
    added_part = ""
    added_part += "\t type  \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Back BC ################################  
    
    start_index = foam.find_keyword_line(dict_lines, "back") + 2 
    added_part = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Building BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "building") + 2 
    
    if building_BC_type == "noSlip": 
        added_part = ""
        added_part += "\t type \t zeroGradient;\n"
    
    if building_BC_type == "roughWallFunction": 
        added_part = ""
        added_part += "\t type \t epsilonWallFunction;\n"
        added_part += "\t Cmu \t {:.4f};\n".format(0.09)
        added_part += "\t kappa \t {:.4f};\n".format(0.4)
        added_part += "\t E \t {:.4f};\n".format(9.8)
        added_part += "\t value \t uniform {:.4f};\n".format(epsilon0)

    if building_BC_type == "smoothWallFunction": 
        added_part = ""
        added_part += "\t type \t epsilonWallFunction;\n"
        added_part += "\t Cmu \t {:.4f};\n".format(0.09)
        added_part += "\t kappa \t {:.4f};\n".format(0.4)
        added_part += "\t E \t {:.4f};\n".format(9.8)
        added_part += "\t value \t uniform {:.4f};\n".format(epsilon0)


    dict_lines.insert(start_index, added_part)
    
    #Write edited dict to file
    write_file_name = case_path + "/0/epsilon"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()

def write_k_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data["boundaryConditions"]
    wind_data = json_data["windCharacteristics"]
      
    
    sides_BC_type = boundary_data['sidesBoundaryCondition']
    top_BC_type = boundary_data['topBoundaryCondition']
    ground_BC_type = boundary_data['groundBoundaryCondition']
    building_BC_type = boundary_data['buildingBoundaryCondition']

    wind_speed = wind_data['referenceWindSpeed']
    building_height = wind_data['referenceHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']
    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/kFileTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    
    #BC and initial condition (you may need to scale to model scale)
    # k0 = 1.3 #not in model scale
    
    I = 0.1 
    k0 = 1.5*(I*wind_speed)**2  

    ##################### Internal Field #########################
    
    start_index = foam.find_keyword_line(dict_lines, "internalField") 
    dict_lines[start_index] = "internalField \t uniform {:.4f};\n".format(k0)


    ###################### Inlet BC ##############################  
    #Write uniform
    start_index = foam.find_keyword_line(dict_lines, "inlet") + 2 
    added_part  = ""
    added_part += "\t type \t atmBoundaryLayerInletK;\n"
    added_part += "\t Uref \t {:.4f};\n".format(wind_speed)
    added_part += "\t Zref \t {:.4f};\n".format(building_height)
    added_part += "\t zDir \t (0.0 0.0 1.0);\n"
    added_part += "\t flowDir \t (1.0 0.0 0.0);\n"
    added_part += "\t z0 \t uniform {:.4e};\n".format(roughness_length)
    added_part += "\t zGround \t uniform 0.0;\n"
    
    dict_lines.insert(start_index, added_part)

    ###################### Outlet BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "outlet") + 2 
    added_part = ""
    added_part += "\t type \t inletOutlet;\n"
    added_part += "\t inletValue \t uniform {:.4f};\n".format(k0)
    added_part += "\t value \t uniform {:.4f};\n".format(k0)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Ground BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "ground") + 2 
    
    if ground_BC_type == "noSlip": 
        added_part = ""
        added_part += "\t type \t zeroGradient;\n"
    
    if ground_BC_type == "smoothWallFunction": 
        added_part = ""
        added_part += "\t type \t kqRWallFunction;\n"
        added_part += "\t value \t uniform {:.4f};\n".format(0.0)

    if ground_BC_type == "roughWallFunction": 
        added_part = ""
        added_part += "\t type \t kqRWallFunction;\n"
        added_part += "\t value \t uniform {:.4f};\n".format(0.0)

    dict_lines.insert(start_index, added_part)
    
    
    ###################### Top BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "top") + 2 
    added_part = ""
    added_part += "\t type  \t {};\n".format(top_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Front BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "front") + 2 
    added_part  = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Back BC ################################  
    
    start_index = foam.find_keyword_line(dict_lines, "back") + 2 
    added_part  = ""
    added_part += "\t type \t {};\n".format(sides_BC_type)
    
    dict_lines.insert(start_index, added_part)
    
    ###################### Building BC ##############################  
    
    start_index = foam.find_keyword_line(dict_lines, "building") + 2 
    
    if building_BC_type == "noSlip": 
        added_part = ""
        added_part += "\t type \t zeroGradient;\n"
    
    if building_BC_type == "smoothWallFunction": 
        added_part = ""
        added_part += "\t type \t kqRWallFunction;\n"
        added_part += "\t value \t uniform {:.6f};\n".format(k0)
    
    #Note:  should be replaced with k wall function for rough walls 
    #       now it's the same with smooth wall function.
    if building_BC_type == "roughWallFunction": 
        added_part = ""
        added_part += "\t type \t kqRWallFunction;\n"
        added_part += "\t value \t uniform {:.6f};\n".format(k0)

    dict_lines.insert(start_index, added_part)
    
    #Write edited dict to file
    write_file_name = case_path + "/0/k"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    
    
def write_controlDict_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    ns_data = json_data["numericalSetup"]
    rm_data = json_data["resultMonitoring"]
          
    solver_type = ns_data['solverType']
    duration = ns_data['duration']
    time_step = ns_data['timeStep']
    max_courant_number = ns_data['maxCourantNumber']
    adjust_time_step = ns_data['adjustTimeStep']
    
    
    num_stories = rm_data['numStories']
    floor_height = rm_data['floorHeight']
    center_of_rotation = rm_data['centerOfRotation']
    story_load_write_interval = rm_data['storyLoadWriteInterval']
    monitor_base_load = rm_data['monitorBaseLoad']
    monitor_surface_pressure = rm_data['monitorSurfacePressure']
    pressure_sampling_points = rm_data['pressureSamplingPoints']
    pressure_write_interval = rm_data['pressureWriteInterval']
    
    # Need to change this for      
    max_delta_t = 10*time_step
    
    write_interval = 1000
    purge_write =  3
    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/controlDictTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    #Write application type 
    start_index = foam.find_keyword_line(dict_lines, "application") 
    dict_lines[start_index] = "application \t{};\n".format(solver_type)
    
    #Write end time 
    start_index = foam.find_keyword_line(dict_lines, "endTime") 
    dict_lines[start_index] = "endTime \t{:.6f};\n".format(duration)
    
    #Write time step time 
    start_index = foam.find_keyword_line(dict_lines, "deltaT") 
    dict_lines[start_index] = "deltaT \t{:.6f};\n".format(time_step)
 
    #Write adjustable time step or not  
    start_index = foam.find_keyword_line(dict_lines, "adjustTimeStep") 
    dict_lines[start_index] = "adjustTimeStep \t\t{};\n".format("yes" if adjust_time_step else "no")
 
    #Write writeInterval  
    start_index = foam.find_keyword_line(dict_lines, "writeInterval") 
    dict_lines[start_index] = "writeInterval \t{};\n".format(write_interval)
    
    #Write maxCo  
    start_index = foam.find_keyword_line(dict_lines, "maxCo") 
    dict_lines[start_index] = "maxCo \t{:.2f};\n".format(max_courant_number)
    
    #Write maximum time step  
    start_index = foam.find_keyword_line(dict_lines, "maxDeltaT") 
    dict_lines[start_index] = "maxDeltaT \t{:.6f};\n".format(max_delta_t)
       

    #Write purge write interval  
    start_index = foam.find_keyword_line(dict_lines, "purgeWrite") 
    dict_lines[start_index] = "purgeWrite \t{};\n".format(purge_write)

    ########################### Function Objects ##############################
     
    #Find function object location  
    start_index = foam.find_keyword_line(dict_lines, "functions") + 2

    #Write story loads functionObjects  
    added_part = "    #includeFunc  storyForces\n"
    dict_lines.insert(start_index, added_part)

    #Write base loads functionObjects
    if monitor_base_load:
        added_part = "    #includeFunc  baseForces\n"
        dict_lines.insert(start_index, added_part)
    
    #Write pressure sampling points 
    if monitor_surface_pressure:
        added_part = "    #includeFunc  pressureSamplingPoints\n"
        dict_lines.insert(start_index, added_part)
    

    #Write edited dict to file
    write_file_name = case_path + "/system/controlDict"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    
def write_fvSolution_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    ns_data = json_data["numericalSetup"]
      
    json_file.close()
    
    num_non_orthogonal_correctors = ns_data['numNonOrthogonalCorrectors']
    num_correctors = ns_data['numCorrectors']
    num_outer_correctors = ns_data['numOuterCorrectors']
        
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/fvSolutionTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()

    
    #Write simpleFoam options  
    start_index = foam.find_keyword_line(dict_lines, "SIMPLE") + 2   
    added_part = ""
    added_part += "    nNonOrthogonalCorrectors \t{};\n".format(num_non_orthogonal_correctors)
    dict_lines.insert(start_index, added_part)

    
    #Write pimpleFoam options  
    start_index = foam.find_keyword_line(dict_lines, "PIMPLE") + 2   
    added_part = ""
    added_part += "    nOuterCorrectors \t{};\n".format(num_outer_correctors)
    added_part += "    nCorrectors \t{};\n".format(num_correctors)
    added_part += "    nNonOrthogonalCorrectors \t{};\n".format(num_non_orthogonal_correctors)
    dict_lines.insert(start_index, added_part)


    #Write pisoFoam options  
    start_index = foam.find_keyword_line(dict_lines, "PISO") + 2   
    added_part = ""
    added_part += "    nCorrectors \t{};\n".format(num_correctors)
    added_part += "    nNonOrthogonalCorrectors \t{};\n".format(num_non_orthogonal_correctors)
    dict_lines.insert(start_index, added_part)
   
   
    #Write edited dict to file
    write_file_name = case_path + "/system/fvSolution"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()    


def write_pressure_probes_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    rm_data = json_data["resultMonitoring"]
      

    pressure_sampling_points = rm_data['pressureSamplingPoints']
    pressure_write_interval = rm_data['pressureWriteInterval']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/probeTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    

    #Write writeInterval 
    start_index = foam.find_keyword_line(dict_lines, "writeInterval") 
    dict_lines[start_index] = "writeInterval \t{};\n".format(pressure_write_interval)
    
    
    #Write fields to be montored 
    start_index = foam.find_keyword_line(dict_lines, "fields") 
    dict_lines[start_index] = "fields \t\t(p);\n"
    
    start_index = foam.find_keyword_line(dict_lines, "probeLocations") + 2

    added_part = ""
    
    for i in range(len(pressure_sampling_points)):
        added_part += " ({:.6f} {:.6f} {:.6f})\n".format(pressure_sampling_points[i][0], pressure_sampling_points[i][1], pressure_sampling_points[i][2])
    
    dict_lines.insert(start_index, added_part)

    #Write edited dict to file
    write_file_name = case_path + "/system/pressureSamplingPoints"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    
    
  
def write_base_forces_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    air_density = 1.0

    # Returns JSON object as a dictionary
    rm_data = json_data["resultMonitoring"]   

    num_stories = rm_data['numStories']
    floor_height = rm_data['floorHeight']
    center_of_rotation = rm_data['centerOfRotation']
    base_load_write_interval = rm_data['baseLoadWriteInterval']
    monitor_base_load = rm_data['monitorBaseLoad']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/baseForcesTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    

    #Write writeInterval 
    start_index = foam.find_keyword_line(dict_lines, "writeInterval") 
    dict_lines[start_index] = "writeInterval \t{};\n".format(base_load_write_interval)    
    
    #Write patch name to intergrate forces on 
    start_index = foam.find_keyword_line(dict_lines, "patches") 
    dict_lines[start_index] = "patches \t({});\n".format("building")
    
    #Write air density to rhoInf 
    start_index = foam.find_keyword_line(dict_lines, "rhoInf") 
    dict_lines[start_index] = "rhoInf \t\t{:.4f};\n".format(air_density)
    
    #Write center of rotation
    start_index = foam.find_keyword_line(dict_lines, "CofR") 
    dict_lines[start_index] = "CofR \t\t({:.4f} {:.4f} {:.4f});\n".format(center_of_rotation[0], center_of_rotation[1], center_of_rotation[2])
    

    #Write edited dict to file
    write_file_name = case_path + "/system/baseForces"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    
def write_story_forces_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    air_density = 1.0

    # Returns JSON object as a dictionary
    rm_data = json_data["resultMonitoring"]    

    num_stories = rm_data['numStories']
    floor_height = rm_data['floorHeight']
    center_of_rotation = rm_data['centerOfRotation']
    story_load_write_interval = rm_data['storyLoadWriteInterval']
    monitor_base_load = rm_data['monitorBaseLoad']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/storyForcesTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    

    #Write writeInterval 
    start_index = foam.find_keyword_line(dict_lines, "writeInterval") 
    dict_lines[start_index] = "writeInterval \t{};\n".format(story_load_write_interval)    
    
    #Write patch name to intergrate forces on 
    start_index = foam.find_keyword_line(dict_lines, "patches") 
    dict_lines[start_index] = "patches \t({});\n".format("building")
    
    #Write air density to rhoInf 
    start_index = foam.find_keyword_line(dict_lines, "rhoInf") 
    dict_lines[start_index] = "rhoInf \t\t{:.4f};\n".format(air_density)
    
    #Write center of rotation
    start_index = foam.find_keyword_line(dict_lines, "CofR") 
    dict_lines[start_index] = "CofR \t\t({:.4f} {:.4f} {:.4f});\n".format(center_of_rotation[0], center_of_rotation[1], center_of_rotation[2])
    
    #Number of stories  as nBins
    start_index = foam.find_keyword_line(dict_lines, "nBin") 
    dict_lines[start_index] = "    nBin \t{};\n".format(num_stories)
    
    #Write story direction
    start_index = foam.find_keyword_line(dict_lines, "direction") 
    dict_lines[start_index] = "    direction \t({:.4f} {:.4f} {:.4f});\n".format(0, 0, 1.0)

    #Write edited dict to file
    write_file_name = case_path + "/system/storyForces"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    
    
def write_momentumTransport_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    turb_data = json_data["turbulenceModeling"]
      
    simulation_type = turb_data['simulationType']
    RANS_type = turb_data['RANSModelType']
    LES_type = turb_data['LESModelType']
    DES_type = turb_data['DESModelType']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/momentumTransportTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    

    #Write type of the simulation 
    start_index = foam.find_keyword_line(dict_lines, "simulationType") 
    dict_lines[start_index] = "simulationType \t{};\n".format("RAS" if simulation_type=="RANS" else simulation_type)
    
    if simulation_type=="RANS":
        #Write RANS model type 
        start_index = foam.find_keyword_line(dict_lines, "RAS") + 2
        added_part = "    model \t{};\n".format(RANS_type)
        dict_lines.insert(start_index, added_part)
        
    elif simulation_type=="LES":
        #Write LES SGS model type 
        start_index = foam.find_keyword_line(dict_lines, "LES") + 2
        added_part = "    model \t{};\n".format(LES_type)
        dict_lines.insert(start_index, added_part)
    
    elif simulation_type=="DES":
        #Write DES model type 
        start_index = foam.find_keyword_line(dict_lines, "LES") + 2
        added_part = "    model \t{};\n".format(DES_type)
        dict_lines.insert(start_index, added_part)

    #Write edited dict to file
    write_file_name = case_path + "/constant/momentumTransport"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    
def write_physicalProperties_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    wc_data = json_data["windCharacteristics"]
      
    
    kinematic_viscosity = wc_data['kinematicViscosity']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/physicalPropertiesTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    

    #Write type of the simulation 
    start_index = foam.find_keyword_line(dict_lines, "nu") 
    dict_lines[start_index] = "nu\t\t[0 2 -1 0 0 0 0] {:.4e};\n".format(kinematic_viscosity)


    #Write edited dict to file
    write_file_name = case_path + "/constant/physicalProperties"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()
    

def write_transportProperties_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    wc_data = json_data["windCharacteristics"]
      
    
    kinematic_viscosity = wc_data['kinematicViscosity']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/transportPropertiesTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    

    #Write type of the simulation 
    start_index = foam.find_keyword_line(dict_lines, "nu") 
    dict_lines[start_index] = "nu\t\t[0 2 -1 0 0 0 0] {:.3e};\n".format(kinematic_viscosity)


    #Write edited dict to file
    write_file_name = case_path + "/constant/transportProperties"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()

def write_fvSchemes_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json", 'r', encoding='utf-8') as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    turb_data = json_data["turbulenceModeling"]
      
    
    simulation_type = turb_data['simulationType']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/fvSchemesTemplate{}".format(simulation_type), "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    

    #Write edited dict to file
    write_file_name = case_path + "/system/fvSchemes"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()    
    
def write_decomposeParDict_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json") as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    ns_data = json_data["numericalSetup"]
      
    num_processors = ns_data['numProcessors']

    
    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/decomposeParDictTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    #Write number of sub-domains
    start_index = foam.find_keyword_line(dict_lines, "numberOfSubdomains") 
    dict_lines[start_index] = "numberOfSubdomains\t{};\n".format(num_processors)
    
    #Write method of decomposition
    start_index = foam.find_keyword_line(dict_lines, "decomposer") 
    dict_lines[start_index] = "decomposer\t\t{};\n".format("scotch")

    #Write method of decomposition for OF-V9 and lower compatability
    start_index = foam.find_keyword_line(dict_lines, "method") 
    dict_lines[start_index] = "method\t\t{};\n".format("scotch")
    

    #Write edited dict to file
    write_file_name = case_path + "/system/decomposeParDict"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()    
    
def write_DFSRTurbDict_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json") as json_file:
        json_data =  json.load(json_file)
    
    fmax = 200.0

    # Returns JSON object as a dictionary
    wc_data = json_data["windCharacteristics"]
    ns_data = json_data["numericalSetup"]
      
    wind_speed = wc_data['referenceWindSpeed']
    duration = ns_data['duration']
    
    #Generate a little longer duration to be safe
    duration = duration*1.010

    #Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + "/DFSRTurbDictTemplate", "r")

    dict_lines = dict_file.readlines()
    dict_file.close()
    
    #Write the end time
    start_index = foam.find_keyword_line(dict_lines, "endTime") 
    dict_lines[start_index] = "endTime\t\t\t{:.4f};\n".format(duration)
    
    #Write patch name
    start_index = foam.find_keyword_line(dict_lines, "patchName") 
    dict_lines[start_index] = "patchName\t\t\"{}\";\n".format("inlet")

    #Write cohUav 
    start_index = foam.find_keyword_line(dict_lines, "cohUav") 
    dict_lines[start_index] = "cohUav\t\t\t{:.4f};\n".format(wind_speed)
    
    #Write fmax 
    start_index = foam.find_keyword_line(dict_lines, "fMax") 
    dict_lines[start_index] = "fMax\t\t\t{:.4f};\n".format(fmax)  
    
    #Write time step 
    start_index = foam.find_keyword_line(dict_lines, "timeStep") 
    dict_lines[start_index] = "timeStep\t\t{:.4f};\n".format(1.0/fmax)

    #Write edited dict to file
    write_file_name = case_path + "/constant/DFSRTurbDict"
    
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
    
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)
    output_file.close()    
    

if __name__ == '__main__':    
    
    input_args = sys.argv

    # Set filenames
    input_json_path = sys.argv[1]
    template_dict_path = sys.argv[2]
    case_path = sys.argv[3]
    
    
    # input_json_path = "/home/abiy/Documents/WE-UQ/LocalWorkDir/IsolatedBuildingCFD/constant/simCenter/input"
    # template_dict_path = "/home/abiy/SimCenter/SourceCode/NHERI-SimCenter/SimCenterBackendApplications/applications/createEVENT/IsolatedBuildingCFD/templateOF10Dicts"
    # case_path = "/home/abiy/Documents/WE-UQ/LocalWorkDir/IsolatedBuildingCFD"
    
    # data_path = os.getcwd()
    # script_path = os.path.dirname(os.path.realpath(__file__))
    
    
    #Create case director
    # set up goes here 

    
    
    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json") as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    turb_data = json_data["turbulenceModeling"]
       
    simulation_type = turb_data['simulationType']
    RANS_type = turb_data['RANSModelType']
    LES_type = turb_data['LESModelType']
    
    #Write blockMesh
    write_block_mesh_dict(input_json_path, template_dict_path, case_path)

    #Create and write the building .stl file
    #Also, import STL file if the shape is complex, the check is done inside the function
    write_building_stl_file(input_json_path, case_path)
    
    #Create and write the SnappyHexMeshDict file
    write_snappy_hex_mesh_dict(input_json_path, template_dict_path, case_path)
    
    #Create and write the surfaceFeaturesDict file
    write_surfaceFeaturesDict_file(input_json_path, template_dict_path, case_path)
    
    #Write files in "0" directory
    write_U_file(input_json_path, template_dict_path, case_path)
    write_p_file(input_json_path, template_dict_path, case_path)
    write_nut_file(input_json_path, template_dict_path, case_path)
    write_k_file(input_json_path, template_dict_path, case_path)
    
    if simulation_type == "RANS" and RANS_type=="kEpsilon":
        write_epsilon_file(input_json_path, template_dict_path, case_path)

    #Write control dict
    write_controlDict_file(input_json_path, template_dict_path, case_path)
    
    #Write results to be monitored
    write_base_forces_file(input_json_path, template_dict_path, case_path)
    write_story_forces_file(input_json_path, template_dict_path, case_path)
    write_pressure_probes_file(input_json_path, template_dict_path, case_path)
    
    #Write fvSolution dict
    write_fvSolution_file(input_json_path, template_dict_path, case_path)

    #Write fvSchemes dict
    write_fvSchemes_file(input_json_path, template_dict_path, case_path)

    #Write momentumTransport dict
    write_momentumTransport_file(input_json_path, template_dict_path, case_path)
    
    #Write physicalProperties dict
    write_physicalProperties_file(input_json_path, template_dict_path, case_path)
    
    #Write transportProperties (physicalProperties in OF-10) dict for OpenFOAM-9 and below
    write_transportProperties_file(input_json_path, template_dict_path, case_path)
    
    #Write decomposeParDict
    write_decomposeParDict_file(input_json_path, template_dict_path, case_path)
    
    #Write DFSRTurb dict
    write_DFSRTurbDict_file(input_json_path, template_dict_path, case_path)
    
