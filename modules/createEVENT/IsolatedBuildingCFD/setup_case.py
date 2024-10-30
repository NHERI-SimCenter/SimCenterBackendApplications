"""This script writes BC and initial condition, and setups the OpenFoam case
directory.

"""  # noqa: INP001, D205, D404

import json
import os
import sys

import foam_dict_reader as foam
import numpy as np
from stl import mesh


def write_block_mesh_dict(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    mesh_data = json_data['blockMeshParameters']
    geom_data = json_data['GeometricData']
    boundary_data = json_data['boundaryConditions']

    normalization_type = geom_data['normalizationType']
    origin = np.array(geom_data['origin'])
    scale = geom_data['geometricScale']
    H = geom_data['buildingHeight'] / scale  # convert to model-scale  # noqa: N806

    Lx = geom_data['domainLength']  # noqa: N806
    Ly = geom_data['domainWidth']  # noqa: N806
    Lz = geom_data['domainHeight']  # noqa: N806
    Lf = geom_data['fetchLength']  # noqa: N806

    x_cells = mesh_data['xNumCells']
    y_cells = mesh_data['yNumCells']
    z_cells = mesh_data['zNumCells']

    x_grading = mesh_data['xGrading']
    y_grading = mesh_data['yGrading']
    z_grading = mesh_data['zGrading']

    bc_map = {
        'slip': 'wall',
        'cyclic': 'cyclic',
        'noSlip': 'wall',
        'symmetry': 'symmetry',
        'empty': 'empty',
        'TInf': 'patch',
        'MeanABL': 'patch',
        'Uniform': 'patch',
        'zeroPressureOutlet': 'patch',
        'roughWallFunction': 'wall',
        'smoothWallFunction': 'wall',
    }

    inlet_type = bc_map[boundary_data['inletBoundaryCondition']]
    outlet_type = bc_map[boundary_data['outletBoundaryCondition']]
    ground_type = bc_map[boundary_data['groundBoundaryCondition']]
    top_type = bc_map[boundary_data['topBoundaryCondition']]
    front_type = bc_map[boundary_data['sidesBoundaryCondition']]
    back_type = bc_map[boundary_data['sidesBoundaryCondition']]

    length_unit = json_data['lengthUnit']

    if normalization_type == 'Relative':
        Lx = Lx * H  # noqa: N806
        Ly = Ly * H  # noqa: N806
        Lz = Lz * H  # noqa: N806
        Lf = Lf * H  # noqa: N806
        origin = origin * H

    x_min = -Lf - origin[0]
    y_min = -Ly / 2.0 - origin[1]
    z_min = 0.0 - origin[2]

    x_max = x_min + Lx
    y_max = y_min + Ly
    z_max = z_min + Lz

    # Open the template blockMeshDict (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/blockMeshDictTemplate')  # noqa: SIM115, PTH123

    # Export to OpenFOAM probe format
    dict_lines = dict_file.readlines()
    dict_file.close()

    dict_lines[17] = f'\txMin\t\t{x_min:.4f};\n'
    dict_lines[18] = f'\tyMin\t\t{y_min:.4f};\n'
    dict_lines[19] = f'\tzMin\t\t{z_min:.4f};\n'

    dict_lines[20] = f'\txMax\t\t{x_max:.4f};\n'
    dict_lines[21] = f'\tyMax\t\t{y_max:.4f};\n'
    dict_lines[22] = f'\tzMax\t\t{z_max:.4f};\n'

    dict_lines[23] = f'\txCells\t\t{x_cells:d};\n'
    dict_lines[24] = f'\tyCells\t\t{y_cells:d};\n'
    dict_lines[25] = f'\tzCells\t\t{z_cells:d};\n'

    dict_lines[26] = f'\txGrading\t{x_grading:.4f};\n'
    dict_lines[27] = f'\tyGrading\t{y_grading:.4f};\n'
    dict_lines[28] = f'\tzGrading\t{z_grading:.4f};\n'

    convert_to_meters = 1.0

    if length_unit == 'm':
        convert_to_meters = 1.0
    elif length_unit == 'cm':
        convert_to_meters = 0.01
    elif length_unit == 'mm':
        convert_to_meters = 0.001
    elif length_unit == 'ft':
        convert_to_meters = 0.3048
    elif length_unit == 'in':
        convert_to_meters = 0.0254

    dict_lines[31] = f'convertToMeters {convert_to_meters:.4f};\n'
    dict_lines[61] = f'        type {inlet_type};\n'
    dict_lines[70] = f'        type {outlet_type};\n'
    dict_lines[79] = f'        type {ground_type};\n'
    dict_lines[88] = f'        type {top_type};\n'
    dict_lines[97] = f'        type {front_type};\n'
    dict_lines[106] = f'        type {back_type};\n'

    write_file_name = case_path + '/system/blockMeshDict'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_building_stl_file(input_json_path, case_path):  # noqa: C901, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    geom_data = json_data['GeometricData']

    if geom_data['buildingShape'] == 'Complex':
        import_building_stl_file(input_json_path, case_path)
        return

    # Else create the STL file
    scale = geom_data['geometricScale']
    length_unit = json_data['lengthUnit']

    convert_to_meters = 1.0

    if length_unit == 'm':
        convert_to_meters = 1.0
    elif length_unit == 'cm':
        convert_to_meters = 0.01
    elif length_unit == 'mm':
        convert_to_meters = 0.001
    elif length_unit == 'ft':
        convert_to_meters = 0.3048
    elif length_unit == 'in':
        convert_to_meters = 0.0254

    # Convert from full-scale to model-scale
    B = convert_to_meters * geom_data['buildingWidth'] / scale  # noqa: N806
    D = convert_to_meters * geom_data['buildingDepth'] / scale  # noqa: N806
    H = convert_to_meters * geom_data['buildingHeight'] / scale  # noqa: N806

    normalization_type = geom_data['normalizationType']

    origin = np.array(geom_data['origin'])
    wind_dxn = geom_data['windDirection']

    if normalization_type == 'Relative':
        origin = origin * H

    wind_dxn_rad = np.deg2rad(wind_dxn)
    epsilon = 1.0e-5

    # Define the 8 vertices of the building
    vertices = np.array(
        [
            [-D / 2.0, -B / 2.0, -epsilon],
            [+D / 2.0, -B / 2.0, -epsilon],
            [+D / 2.0, +B / 2.0, -epsilon],
            [-D / 2.0, +B / 2.0, -epsilon],
            [-D / 2.0, -B / 2.0, +H],
            [+D / 2.0, -B / 2.0, +H],
            [+D / 2.0, +B / 2.0, +H],
            [-D / 2.0, +B / 2.0, +H],
        ]
    )

    n_vertices = np.shape(vertices)[0]

    # The default coordinate system is building center.
    # Transform the preferred origin
    vertices = vertices - origin

    # Transform transform the vertices to account the wind direction.
    trans_vertices = np.zeros((n_vertices, 3))
    trans_vertices[:, 2] = vertices[:, 2]

    t_matrix = np.array(
        [
            [np.cos(wind_dxn_rad), -np.sin(wind_dxn_rad)],
            [np.sin(wind_dxn_rad), np.cos(wind_dxn_rad)],
        ]
    )

    for i in range(n_vertices):
        trans_vertices[i, 0:2] = np.matmul(t_matrix, vertices[i, 0:2])

    # Define the 12 triangles composing the rectangular building
    faces = np.array(
        [
            [0, 3, 1],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [4, 5, 6],
            [4, 6, 7],
            [5, 1, 2],
            [5, 2, 6],
            [2, 3, 6],
            [3, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
        ]
    )

    # Create the mesh
    bldg = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            bldg.vectors[i][j] = trans_vertices[f[j], :]

    # Write the mesh to file "building.stl"
    fmt = mesh.stl.Mode.ASCII  # binary or ASCII format
    bldg.save(case_path + '/constant/geometry/building.stl', mode=fmt)


def import_building_stl_file(input_json_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    if json_data['GeometricData']['buildingShape'] == 'Simple':
        return

    # Returns JSON object as a dictionary
    stl_path = json_data['GeometricData']['importedSTLPath']
    scale_factor = json_data['GeometricData']['stlScaleFactor']
    recenter = json_data['GeometricData']['recenterToOrigin']
    use_stl_dimension = json_data['GeometricData']['useSTLDimensions']  # noqa: F841
    account_wind_direction = json_data['GeometricData']['accountWindDirection']
    origin = np.array(json_data['GeometricData']['origin'])
    wind_dxn = json_data['GeometricData']['windDirection']
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
        'xMin': float(min_x),
        'xMax': float(max_x),
        'yMin': float(min_y),
        'yMax': float(max_y),
        'zMin': float(min_z),
        'zMax': float(max_z),
    }

    # Serializing json
    json_object = json.dumps(stl_summary, indent=4)

    # Writing to sample.json
    with open(  # noqa: PTH123
        input_json_path + '/stlGeometrySummary.json', 'w', encoding='utf-8'
    ) as outfile:
        outfile.write(json_object)

    # Translate the bottom center to origin
    if recenter:
        t = (
            np.array(
                [
                    -((max_x - min_x) / 2.0 + min_x),
                    -((max_y - min_y) / 2.0 + min_y),
                    -min_z,
                ]
            )
            - origin / scale_factor
        )
        bldg_mesh.translate(t)

    # Account wind direction by rotation
    if account_wind_direction:
        # Rotate about z-axis
        bldg_mesh.rotate(np.array([0, 0, 1.0]), wind_dxn_rad)

    # Scale the mesh
    bldg_mesh.vectors *= scale_factor

    # Write the mesh to file "building.stl"
    fmt = mesh.stl.Mode.ASCII  # binary or ASCII format
    bldg_mesh.save(case_path + '/constant/geometry/building.stl', mode=fmt)


def write_surfaceFeaturesDict_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    domain_data = json_data['snappyHexMeshParameters']
    building_stl_name = domain_data['buildingSTLName']

    # Open the template blockMeshDict (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/surfaceFeaturesDictTemplate')  # noqa: SIM115, PTH123

    # Export to OpenFOAM probe format
    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write 'addLayers' switch
    start_index = foam.find_keyword_line(dict_lines, 'surfaces')
    dict_lines[start_index] = f'surfaces  ("{building_stl_name}.stl");\n'

    # Write edited dict to file
    write_file_name = case_path + '/system/surfaceFeaturesDict'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_snappy_hex_mesh_dict(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    mesh_data = json_data['snappyHexMeshParameters']

    geom_data = json_data['GeometricData']

    scale = geom_data['geometricScale']
    H = geom_data['buildingHeight'] / scale  # convert to model-scale  # noqa: N806

    Lx = geom_data['domainLength']  # noqa: N806
    Ly = geom_data['domainWidth']  # noqa: N806
    Lz = geom_data['domainHeight']  # noqa: N806
    Lf = geom_data['fetchLength']  # noqa: N806

    normalization_type = geom_data['normalizationType']
    origin = np.array(geom_data['origin'])

    building_stl_name = mesh_data['buildingSTLName']
    num_cells_between_levels = mesh_data['numCellsBetweenLevels']
    resolve_feature_angle = mesh_data['resolveFeatureAngle']
    num_processors = mesh_data['numProcessors']  # noqa: F841

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
    prism_layer_relative_size = 'on'

    if normalization_type == 'Relative':
        Lx = Lx * H  # noqa: N806
        Ly = Ly * H  # noqa: N806
        Lz = Lz * H  # noqa: N806
        Lf = Lf * H  # noqa: N806
        origin = origin * H

        for i in range(len(refinement_boxes)):
            for j in range(2, 8, 1):
                refinement_boxes[i][j] = refinement_boxes[i][j] * H

        surface_refinement_distance = surface_refinement_distance * H

    x_min = -Lf - origin[0]
    y_min = -Ly / 2.0 - origin[1]
    z_min = 0.0 - origin[2]

    x_max = x_min + Lx  # noqa: F841
    y_max = y_min + Ly
    z_max = z_min + Lz  # noqa: F841

    inside_point = [x_min + Lf / 2.0, (y_min + y_max) / 2.0, H]

    # Open the template blockMeshDict (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/snappyHexMeshDictTemplate')  # noqa: SIM115, PTH123

    # Export to OpenFOAM probe format
    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write 'addLayers' switch
    start_index = foam.find_keyword_line(dict_lines, 'addLayers')
    dict_lines[start_index] = 'addLayers\t{};\n'.format(
        'on' if add_prism_layers else 'off'
    )

    # Edit Geometry Section ##############################

    # Add refinement box geometry
    start_index = foam.find_keyword_line(dict_lines, 'geometry') + 2
    added_part = ''
    n_boxes = len(refinement_boxes)
    for i in range(n_boxes):
        added_part += f'    {refinement_boxes[i][0]}\n'
        added_part += '    {\n'
        added_part += '         type searchableBox;\n'
        added_part += f'         min ({refinement_boxes[i][2]:.4f} {refinement_boxes[i][3]:.4f} {refinement_boxes[i][4]:.4f});\n'
        added_part += f'         max ({refinement_boxes[i][5]:.4f} {refinement_boxes[i][6]:.4f} {refinement_boxes[i][7]:.4f});\n'
        added_part += '    }\n'

    dict_lines.insert(start_index, added_part)

    # Add building stl geometry
    start_index = foam.find_keyword_line(dict_lines, 'geometry') + 2
    added_part = ''
    added_part += f'    {building_stl_name}\n'
    added_part += '    {\n'
    added_part += '         type triSurfaceMesh;\n'
    added_part += f'         file "{building_stl_name}.stl";\n'
    added_part += '    }\n'

    dict_lines.insert(start_index, added_part)

    # Edit castellatedMeshControls Section ####################

    # Write 'nCellsBetweenLevels'
    start_index = foam.find_keyword_line(dict_lines, 'nCellsBetweenLevels')
    dict_lines[start_index] = (
        f'    nCellsBetweenLevels {num_cells_between_levels:d};\n'
    )

    # Write 'resolveFeatureAngle'
    start_index = foam.find_keyword_line(dict_lines, 'resolveFeatureAngle')
    dict_lines[start_index] = f'    resolveFeatureAngle {resolve_feature_angle:d};\n'

    # Write 'insidePoint'
    start_index = foam.find_keyword_line(dict_lines, 'insidePoint')
    dict_lines[start_index] = (
        f'    insidePoint ({inside_point[0]:.4f} {inside_point[1]:.4f} {inside_point[2]:.4f});\n'
    )

    # For compatibility with OpenFOAM-9 and older
    start_index = foam.find_keyword_line(dict_lines, 'locationInMesh')
    dict_lines[start_index] = (
        f'    locationInMesh ({inside_point[0]:.4f} {inside_point[1]:.4f} {inside_point[2]:.4f});\n'
    )

    # Add refinement edge
    if add_edge_refinement:
        start_index = foam.find_keyword_line(dict_lines, 'features') + 2
        added_part = ''
        added_part += '         {\n'
        added_part += f'             file "{refinement_edge_name}.eMesh";\n'
        added_part += f'             level {edge_refinement_level};\n'
        added_part += '         }\n'

        dict_lines.insert(start_index, added_part)

    # Add refinement surface
    if add_surface_refinement:
        start_index = foam.find_keyword_line(dict_lines, 'refinementSurfaces') + 2
        added_part = ''
        added_part += f'         {refinement_surface_name}\n'
        added_part += '         {\n'
        added_part += f'             level ({surface_refinement_level} {surface_refinement_level});\n'
        added_part += '             patchInfo\n'
        added_part += '             {\n'
        added_part += '                 type wall;\n'
        added_part += '             }\n'
        added_part += '         }\n'

        dict_lines.insert(start_index, added_part)

    # Add surface refinement around the building as a refinement region
    # if surface_refinement_level > refinement_boxes[-1][1]:
    added_part = ''
    added_part += f'         {refinement_surface_name}\n'
    added_part += '         {\n'
    added_part += '             mode   distance;\n'
    added_part += f'             levels  (({surface_refinement_distance:.4f} {refinement_boxes[-1][1] + 1}));\n'
    added_part += '         }\n'

    start_index = foam.find_keyword_line(dict_lines, 'refinementRegions') + 2
    dict_lines.insert(start_index, added_part)

    # Add box refinements
    added_part = ''
    for i in range(n_boxes):
        added_part += f'         {refinement_boxes[i][0]}\n'
        added_part += '         {\n'
        added_part += '             mode   inside;\n'
        added_part += f'             level  {refinement_boxes[i][1]};\n'
        added_part += '         }\n'

    start_index = foam.find_keyword_line(dict_lines, 'refinementRegions') + 2
    dict_lines.insert(start_index, added_part)

    # Edit PrismLayer Section ##########################
    # Add surface layers (prism layers)
    added_part = ''
    added_part += f'         "{prism_layer_surface_name}"\n'
    added_part += '         {\n'
    added_part += f'             nSurfaceLayers {number_of_prism_layers};\n'
    added_part += '         }\n'

    start_index = foam.find_keyword_line(dict_lines, 'layers') + 2
    dict_lines.insert(start_index, added_part)

    # Write 'relativeSizes'
    start_index = foam.find_keyword_line(dict_lines, 'relativeSizes')
    dict_lines[start_index] = f'    relativeSizes {prism_layer_relative_size};\n'

    # Write 'expansionRatio'
    start_index = foam.find_keyword_line(dict_lines, 'expansionRatio')
    dict_lines[start_index] = (
        f'    expansionRatio {prism_layer_expansion_ratio:.4f};\n'
    )

    # Write 'finalLayerThickness'
    start_index = foam.find_keyword_line(dict_lines, 'finalLayerThickness')
    dict_lines[start_index] = (
        f'    finalLayerThickness {final_prism_layer_thickness:.4f};\n'
    )

    # Write edited dict to file
    write_file_name = case_path + '/system/snappyHexMeshDict'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_U_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data['boundaryConditions']
    wind_data = json_data['windCharacteristics']

    inlet_BC_type = boundary_data['inletBoundaryCondition']  # noqa: N806
    top_BC_type = boundary_data['topBoundaryCondition']  # noqa: N806
    sides_BC_type = boundary_data['sidesBoundaryCondition']  # noqa: N806
    building_BC_type = boundary_data['buildingBoundaryCondition']  # noqa: N806, F841

    wind_speed = wind_data['referenceWindSpeed']
    building_height = wind_data['referenceHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/UFileTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Internal Field #########################
    # Initialize the internal fields frow a lower velocity to avoid Courant number
    # instability when the solver starts. Now %10 of roof-height wind speed is set
    start_index = foam.find_keyword_line(dict_lines, 'internalField')
    dict_lines[start_index] = 'internalField   uniform (0 0 0);\n'

    # Inlet BC ##############################
    # Write uniform
    start_index = foam.find_keyword_line(dict_lines, 'inlet') + 2

    if inlet_BC_type == 'Uniform':
        added_part = ''
        added_part += '\t type \t fixedValue;\n'
        added_part += f'\t value \t uniform ({wind_speed:.4f} 0 0);\n'

    if inlet_BC_type == 'MeanABL':
        added_part = ''
        added_part += '\t type \t atmBoundaryLayerInletVelocity;\n'
        added_part += f'\t Uref \t {wind_speed:.4f};\n'
        added_part += f'\t Zref \t {building_height:.4f};\n'
        added_part += '\t zDir \t (0.0 0.0 1.0);\n'
        added_part += '\t flowDir \t (1.0 0.0 0.0);\n'
        added_part += f'\t z0 uniform \t {roughness_length:.4e};\n'
        added_part += '\t zGround \t uniform 0.0;\n'

    if inlet_BC_type == 'TInf':
        added_part = ''
        added_part += '\t type \t turbulentDFMInlet;\n'
        added_part += '\t filterType \t exponential;\n'
        added_part += f'\t filterFactor \t {4};\n'
        added_part += f'\t value \t uniform ({wind_speed:.4f} 0 0);\n'
        added_part += '\t periodicInY \t {};\n'.format('true')
        added_part += '\t periodicInZ \t {};\n'.format('false')
        added_part += '\t constMeanU \t {};\n'.format('true')
        added_part += f'\t Uref \t {wind_speed:.4f};\n'

    dict_lines.insert(start_index, added_part)

    # Outlet BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'outlet') + 2
    added_part = ''
    added_part += '\t type \t inletOutlet;\n'
    added_part += '\t inletValue \t uniform (0 0 0);\n'
    added_part += '\t value \t uniform (0 0 0);\n'

    dict_lines.insert(start_index, added_part)

    # Ground BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'ground') + 2
    added_part = ''
    added_part += '\t type \t uniformFixedValue;\n'
    added_part += '\t value \t uniform (0 0 0);\n'
    added_part += '\t uniformValue \t constant (0 0 0);\n'

    dict_lines.insert(start_index, added_part)

    # Top BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'top') + 2
    added_part = ''
    added_part += f'\t type    {top_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Front BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'front') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Back BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'back') + 2
    added_part = ''
    added_part += f'\t type    {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Building BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'building') + 2
    added_part = ''
    added_part += '\t type \t {};\n'.format('noSlip')

    dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/0/U'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+', encoding='utf-8')  # noqa: SIM115, PTH123

    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_p_file(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data['boundaryConditions']

    sides_BC_type = boundary_data['sidesBoundaryCondition']  # noqa: N806
    top_BC_type = boundary_data['topBoundaryCondition']  # noqa: N806

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/pFileTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # BC and initial condition
    p0 = 0.0
    # Internal Field #########################

    start_index = foam.find_keyword_line(dict_lines, 'internalField')
    dict_lines[start_index] = f'internalField   uniform {p0:.4f};\n'

    # Inlet BC ##############################
    # Write uniform
    start_index = foam.find_keyword_line(dict_lines, 'inlet') + 2
    added_part = ''
    added_part += '\t type \t zeroGradient;\n'

    dict_lines.insert(start_index, added_part)

    # Outlet BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'outlet') + 2
    added_part = ''
    added_part += '\t type \t  uniformFixedValue;\n'
    added_part += f'\t uniformValue \t constant {p0:.4f};\n'

    dict_lines.insert(start_index, added_part)

    # Ground BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'ground') + 2
    added_part = ''
    added_part += '\t type \t zeroGradient;\n'

    dict_lines.insert(start_index, added_part)

    # Top BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'top') + 2
    added_part = ''
    added_part += f'\t type \t {top_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Front BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'front') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Back BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'back') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Building BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'building') + 2
    added_part = ''
    added_part += '\t type  \t zeroGradient;\n'

    dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/0/p'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_nut_file(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data['boundaryConditions']
    wind_data = json_data['windCharacteristics']

    sides_BC_type = boundary_data['sidesBoundaryCondition']  # noqa: N806
    top_BC_type = boundary_data['topBoundaryCondition']  # noqa: N806
    ground_BC_type = boundary_data['groundBoundaryCondition']  # noqa: N806
    building_BC_type = boundary_data['buildingBoundaryCondition']  # noqa: N806

    # wind_speed = wind_data['roofHeightWindSpeed']
    # building_height = wind_data['buildingHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/nutFileTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # BC and initial condition
    nut0 = 0.0

    # Internal Field #########################

    start_index = foam.find_keyword_line(dict_lines, 'internalField')
    dict_lines[start_index] = f'internalField   uniform {nut0:.4f};\n'

    # Inlet BC ##############################
    # Write uniform
    start_index = foam.find_keyword_line(dict_lines, 'inlet') + 2
    added_part = ''
    added_part += '\t type \t zeroGradient;\n'

    dict_lines.insert(start_index, added_part)

    # Outlet BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'outlet') + 2
    added_part = ''
    added_part += '\t type \t uniformFixedValue;\n'
    added_part += f'\t uniformValue \t constant {nut0:.4f};\n'

    dict_lines.insert(start_index, added_part)

    # Ground BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'ground') + 2

    if ground_BC_type == 'noSlip':
        added_part = ''
        added_part += '\t type \t zeroGradient;\n'

    if ground_BC_type == 'roughWallFunction':
        added_part = ''
        added_part += '\t type \t nutkAtmRoughWallFunction;\n'
        added_part += f'\t z0  \t  uniform {roughness_length:.4e};\n'
        added_part += '\t value \t uniform 0.0;\n'

    if ground_BC_type == 'smoothWallFunction':
        added_part = ''
        added_part += '\t type \t nutUSpaldingWallFunction;\n'
        added_part += '\t value \t uniform 0;\n'

    dict_lines.insert(start_index, added_part)

    # Top BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'top') + 2
    added_part = ''
    added_part += f'\t type \t {top_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Front BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'front') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Back BC ################################

    start_index = foam.find_keyword_line(dict_lines, 'back') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Building BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'building') + 2

    if building_BC_type == 'noSlip':
        added_part = ''
        added_part += '\t type \t fixedValue;\n'
        added_part += '\t value \t uniform 0;\n'

    if building_BC_type == 'smoothWallFunction':
        added_part = ''
        added_part += '\t type \t nutUSpaldingWallFunction;\n'
        added_part += '\t value \t uniform 0;\n'

    if building_BC_type == 'roughWallFunction':
        added_part = ''
        added_part += '\t type \t nutkRoughWallFunction;\n'
        added_part += '\t Ks \t uniform 1e-5;\n'
        added_part += '\t Cs \t uniform 0.5;\n'
        added_part += '\t value \t uniform 0;\n'

    dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/0/nut'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)
    output_file.close()


def write_epsilon_file(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data['boundaryConditions']
    wind_data = json_data['windCharacteristics']

    sides_BC_type = boundary_data['sidesBoundaryCondition']  # noqa: N806
    top_BC_type = boundary_data['topBoundaryCondition']  # noqa: N806
    ground_BC_type = boundary_data['groundBoundaryCondition']  # noqa: N806
    building_BC_type = boundary_data['buildingBoundaryCondition']  # noqa: N806

    wind_speed = wind_data['referenceWindSpeed']
    building_height = wind_data['referenceHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/epsilonFileTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # BC and initial condition
    epsilon0 = 0.01

    # Internal Field #########################

    start_index = foam.find_keyword_line(dict_lines, 'internalField')
    dict_lines[start_index] = f'internalField   uniform {epsilon0:.4f};\n'

    # Inlet BC ##############################
    # Write uniform
    start_index = foam.find_keyword_line(dict_lines, 'inlet') + 2
    added_part = ''
    added_part += '\t type \t atmBoundaryLayerInletEpsilon;\n'
    added_part += f'\t Uref \t {wind_speed:.4f};\n'
    added_part += f'\t Zref \t {building_height:.4f};\n'
    added_part += '\t zDir \t (0.0 0.0 1.0);\n'
    added_part += '\t flowDir \t (1.0 0.0 0.0);\n'
    added_part += f'\t z0 \t  uniform {roughness_length:.4e};\n'
    added_part += '\t zGround \t uniform 0.0;\n'

    dict_lines.insert(start_index, added_part)

    # Outlet BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'outlet') + 2
    added_part = ''
    added_part += '\t type \t inletOutlet;\n'
    added_part += f'\t inletValue \t uniform {epsilon0:.4f};\n'
    added_part += f'\t value \t uniform {epsilon0:.4f};\n'

    dict_lines.insert(start_index, added_part)

    # Ground BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'ground') + 2

    if ground_BC_type == 'noSlip':
        added_part = ''
        added_part += '\t type \t zeroGradient;\n'

    if ground_BC_type == 'roughWallFunction':
        added_part = ''
        added_part += '\t type \t epsilonWallFunction;\n'
        added_part += f'\t Cmu \t {0.09:.4f};\n'
        added_part += f'\t kappa \t {0.41:.4f};\n'
        added_part += f'\t E \t {9.8:.4f};\n'
        added_part += f'\t value \t uniform {epsilon0:.4f};\n'

    # Note:  Should be replaced with smooth wall function for epsilon,
    #       now the same with rough wall function.
    if ground_BC_type == 'smoothWallFunction':
        added_part = ''
        added_part += '\t type \t epsilonWallFunction;\n'
        added_part += f'\t Cmu \t {0.09:.4f};\n'
        added_part += f'\t kappa \t {0.41:.4f};\n'
        added_part += f'\t E \t {9.8:.4f};\n'
        added_part += f'\t value \t uniform {epsilon0:.4f};\n'
    dict_lines.insert(start_index, added_part)

    # Top BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'top') + 2
    added_part = ''
    added_part += f'\t type  \t  {top_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Front BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'front') + 2
    added_part = ''
    added_part += f'\t type  \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Back BC ################################

    start_index = foam.find_keyword_line(dict_lines, 'back') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Building BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'building') + 2

    if building_BC_type == 'noSlip':
        added_part = ''
        added_part += '\t type \t zeroGradient;\n'

    if building_BC_type == 'roughWallFunction':
        added_part = ''
        added_part += '\t type \t epsilonWallFunction;\n'
        added_part += f'\t Cmu \t {0.09:.4f};\n'
        added_part += f'\t kappa \t {0.4:.4f};\n'
        added_part += f'\t E \t {9.8:.4f};\n'
        added_part += f'\t value \t uniform {epsilon0:.4f};\n'

    if building_BC_type == 'smoothWallFunction':
        added_part = ''
        added_part += '\t type \t epsilonWallFunction;\n'
        added_part += f'\t Cmu \t {0.09:.4f};\n'
        added_part += f'\t kappa \t {0.4:.4f};\n'
        added_part += f'\t E \t {9.8:.4f};\n'
        added_part += f'\t value \t uniform {epsilon0:.4f};\n'

    dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/0/epsilon'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123

    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_k_file(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data['boundaryConditions']
    wind_data = json_data['windCharacteristics']

    sides_BC_type = boundary_data['sidesBoundaryCondition']  # noqa: N806
    top_BC_type = boundary_data['topBoundaryCondition']  # noqa: N806
    ground_BC_type = boundary_data['groundBoundaryCondition']  # noqa: N806
    building_BC_type = boundary_data['buildingBoundaryCondition']  # noqa: N806

    wind_speed = wind_data['referenceWindSpeed']
    building_height = wind_data['referenceHeight']
    roughness_length = wind_data['aerodynamicRoughnessLength']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/kFileTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # BC and initial condition (you may need to scale to model scale)
    # k0 = 1.3 #not in model scale

    I = 0.1  # noqa: N806, E741
    k0 = 1.5 * (I * wind_speed) ** 2

    # Internal Field #########################

    start_index = foam.find_keyword_line(dict_lines, 'internalField')
    dict_lines[start_index] = f'internalField \t uniform {k0:.4f};\n'

    # Inlet BC ##############################
    # Write uniform
    start_index = foam.find_keyword_line(dict_lines, 'inlet') + 2
    added_part = ''
    added_part += '\t type \t atmBoundaryLayerInletK;\n'
    added_part += f'\t Uref \t {wind_speed:.4f};\n'
    added_part += f'\t Zref \t {building_height:.4f};\n'
    added_part += '\t zDir \t (0.0 0.0 1.0);\n'
    added_part += '\t flowDir \t (1.0 0.0 0.0);\n'
    added_part += f'\t z0 \t uniform {roughness_length:.4e};\n'
    added_part += '\t zGround \t uniform 0.0;\n'

    dict_lines.insert(start_index, added_part)

    # Outlet BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'outlet') + 2
    added_part = ''
    added_part += '\t type \t inletOutlet;\n'
    added_part += f'\t inletValue \t uniform {k0:.4f};\n'
    added_part += f'\t value \t uniform {k0:.4f};\n'

    dict_lines.insert(start_index, added_part)

    # Ground BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'ground') + 2

    if ground_BC_type == 'noSlip':
        added_part = ''
        added_part += '\t type \t zeroGradient;\n'

    if ground_BC_type == 'smoothWallFunction':
        added_part = ''
        added_part += '\t type \t kqRWallFunction;\n'
        added_part += f'\t value \t uniform {0.0:.4f};\n'

    if ground_BC_type == 'roughWallFunction':
        added_part = ''
        added_part += '\t type \t kqRWallFunction;\n'
        added_part += f'\t value \t uniform {0.0:.4f};\n'

    dict_lines.insert(start_index, added_part)

    # Top BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'top') + 2
    added_part = ''
    added_part += f'\t type  \t {top_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Front BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'front') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Back BC ################################

    start_index = foam.find_keyword_line(dict_lines, 'back') + 2
    added_part = ''
    added_part += f'\t type \t {sides_BC_type};\n'

    dict_lines.insert(start_index, added_part)

    # Building BC ##############################

    start_index = foam.find_keyword_line(dict_lines, 'building') + 2

    if building_BC_type == 'noSlip':
        added_part = ''
        added_part += '\t type \t zeroGradient;\n'

    if building_BC_type == 'smoothWallFunction':
        added_part = ''
        added_part += '\t type \t kqRWallFunction;\n'
        added_part += f'\t value \t uniform {k0:.6f};\n'

    # Note:  should be replaced with k wall function for rough walls
    #       now it's the same with smooth wall function.
    if building_BC_type == 'roughWallFunction':
        added_part = ''
        added_part += '\t type \t kqRWallFunction;\n'
        added_part += f'\t value \t uniform {k0:.6f};\n'

    dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/0/k'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_controlDict_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    ns_data = json_data['numericalSetup']
    rm_data = json_data['resultMonitoring']

    solver_type = ns_data['solverType']
    duration = ns_data['duration']
    time_step = ns_data['timeStep']
    max_courant_number = ns_data['maxCourantNumber']
    adjust_time_step = ns_data['adjustTimeStep']

    monitor_base_load = rm_data['monitorBaseLoad']
    monitor_surface_pressure = rm_data['monitorSurfacePressure']

    monitor_vtk_planes = rm_data['monitorVTKPlane']
    vtk_planes = rm_data['vtkPlanes']
  # noqa: W293
    # Need to change this for  # noqa: W291
    max_delta_t = 10*time_step
  # noqa: W293
    #Write 10 times
    write_frequency = 10.0
    write_interval_time = duration / write_frequency
    write_interval_count = int(write_interval_time / time_step)
    purge_write = 3

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/controlDictTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write application type
    start_index = foam.find_keyword_line(dict_lines, 'application')
    dict_lines[start_index] = f'application \t{solver_type};\n'

    # Write end time
    start_index = foam.find_keyword_line(dict_lines, 'endTime')
    dict_lines[start_index] = f'endTime \t{duration:.6f};\n'

    # Write time step time
    start_index = foam.find_keyword_line(dict_lines, 'deltaT')
    dict_lines[start_index] = f'deltaT \t{time_step:.6f};\n'

    # Write writeControl
    start_index = foam.find_keyword_line(dict_lines, 'writeControl')
    if solver_type == 'pimpleFoam' and adjust_time_step:
        dict_lines[start_index] = 'writeControl \t{};\n'.format('adjustableRunTime')
    else:
        dict_lines[start_index] = 'writeControl \t\t{};\n'.format('timeStep')

    # Write adjustable time step or not
    start_index = foam.find_keyword_line(dict_lines, 'adjustTimeStep')
    dict_lines[start_index] = 'adjustTimeStep \t\t{};\n'.format(
        'yes' if adjust_time_step else 'no'
    )

    # Write writeInterval
    start_index = foam.find_keyword_line(dict_lines, 'writeInterval')
    if solver_type == 'pimpleFoam' and adjust_time_step:
        dict_lines[start_index] = f'writeInterval \t{write_interval_time:.6f};\n'
    else:
        dict_lines[start_index] = f'writeInterval \t{write_interval_count};\n'

    # Write maxCo
    start_index = foam.find_keyword_line(dict_lines, 'maxCo')
    dict_lines[start_index] = f'maxCo \t{max_courant_number:.2f};\n'

    # Write maximum time step
    start_index = foam.find_keyword_line(dict_lines, 'maxDeltaT')
    dict_lines[start_index] = f'maxDeltaT \t{max_delta_t:.6f};\n'

    # Write purge write interval
    start_index = foam.find_keyword_line(dict_lines, 'purgeWrite')
    dict_lines[start_index] = f'purgeWrite \t{purge_write};\n'

    # Function Objects ##############################

    # Find function object location
    start_index = foam.find_keyword_line(dict_lines, 'functions') + 2

    # Write story loads functionObjects
    added_part = '    #includeFunc  storyForces\n'
    dict_lines.insert(start_index, added_part)

    # Write base loads functionObjects
    if monitor_base_load:
        added_part = '    #includeFunc  baseForces\n'
        dict_lines.insert(start_index, added_part)

    #Write VTK sampling sampling points  # noqa: W291
    if monitor_vtk_planes:
        added_part = ""
        for pln in vtk_planes:
            added_part += "    #includeFunc  {}\n".format(pln["name"])
        dict_lines.insert(start_index, added_part)
  # noqa: W293


    #Write edited dict to file
    write_file_name = case_path + "/system/controlDict"
  # noqa: W293
    if os.path.exists(write_file_name):
        os.remove(write_file_name)
  # noqa: W293
    output_file = open(write_file_name, "w+")
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_fvSolution_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    ns_data = json_data['numericalSetup']

    json_file.close()

    num_non_orthogonal_correctors = ns_data['numNonOrthogonalCorrectors']
    num_correctors = ns_data['numCorrectors']
    num_outer_correctors = ns_data['numOuterCorrectors']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/fvSolutionTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write simpleFoam options
    start_index = foam.find_keyword_line(dict_lines, 'SIMPLE') + 2
    added_part = ''
    added_part += (
        f'    nNonOrthogonalCorrectors \t{num_non_orthogonal_correctors};\n'
    )
    dict_lines.insert(start_index, added_part)

    # Write pimpleFoam options
    start_index = foam.find_keyword_line(dict_lines, 'PIMPLE') + 2
    added_part = ''
    added_part += f'    nOuterCorrectors \t{num_outer_correctors};\n'
    added_part += f'    nCorrectors \t{num_correctors};\n'
    added_part += (
        f'    nNonOrthogonalCorrectors \t{num_non_orthogonal_correctors};\n'
    )
    dict_lines.insert(start_index, added_part)

    # Write pisoFoam options
    start_index = foam.find_keyword_line(dict_lines, 'PISO') + 2
    added_part = ''
    added_part += f'    nCorrectors \t{num_correctors};\n'
    added_part += (
        f'    nNonOrthogonalCorrectors \t{num_non_orthogonal_correctors};\n'
    )
    dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/system/fvSolution'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_generated_pressure_probes_file(  # noqa: D103
    input_json_path,
    template_dict_path,
    case_path,
):
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    rm_data = json_data['resultMonitoring']
    ns_data = json_data['numericalSetup']
    solver_type = ns_data['solverType']
    time_step = ns_data['timeStep']
    adjust_time_step = ns_data['adjustTimeStep']
    monitor_surface_pressure = rm_data['monitorSurfacePressure']

    if monitor_surface_pressure:
        generated_sampling_points = rm_data['generatedPressureSamplingPoints']
        pressure_write_interval = rm_data['pressureWriteInterval']

        # Open the template file (OpenFOAM file) for manipulation
        dict_file = open(template_dict_path + '/probeTemplate')  # noqa: SIM115, PTH123

        dict_lines = dict_file.readlines()
        dict_file.close()

        # Write writeControl
        start_index = foam.find_keyword_line(dict_lines, 'writeControl')
        if solver_type == 'pimpleFoam' and adjust_time_step:
            dict_lines[start_index] = 'writeControl \t{};\n'.format(
                'adjustableRunTime'
            )
        else:
            dict_lines[start_index] = 'writeControl \t{};\n'.format('timeStep')

        # Write writeInterval
        start_index = foam.find_keyword_line(dict_lines, 'writeInterval')
        if solver_type == 'pimpleFoam' and adjust_time_step:
            dict_lines[start_index] = (
                f'writeInterval \t{pressure_write_interval * time_step:.6f};\n'
            )
        else:
            dict_lines[start_index] = f'writeInterval \t{pressure_write_interval};\n'

        # Write fields to be motored
        start_index = foam.find_keyword_line(dict_lines, 'fields')
        dict_lines[start_index] = 'fields \t\t(p);\n'

        start_index = foam.find_keyword_line(dict_lines, 'probeLocations') + 2

        added_part = ''

        for i in range(len(generated_sampling_points)):
            added_part += f' ({generated_sampling_points[i][0]:.6f} {generated_sampling_points[i][1]:.6f} {generated_sampling_points[i][2]:.6f})\n'

        dict_lines.insert(start_index, added_part)

        # Write edited dict to file
        write_file_name = case_path + '/system/generatedPressureSamplingPoints'

        if os.path.exists(write_file_name):  # noqa: PTH110
            os.remove(write_file_name)  # noqa: PTH107

        output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
        for line in dict_lines:
            output_file.write(line)

        output_file.close()


def write_imported_pressure_probes_file(  # noqa: D103
    input_json_path,
    template_dict_path,
    case_path,
):
    # Read JSON data
    with open(input_json_path + '/IsolatedBuildingCFD.json') as json_file:  # noqa: PTH123
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    rm_data = json_data['resultMonitoring']
    ns_data = json_data['numericalSetup']
    solver_type = ns_data['solverType']
    time_step = ns_data['timeStep']
    adjust_time_step = ns_data['adjustTimeStep']
    monitor_surface_pressure = rm_data['monitorSurfacePressure']

    if monitor_surface_pressure:
        imported_sampling_points = rm_data['importedPressureSamplingPoints']
        pressure_write_interval = rm_data['pressureWriteInterval']

        # Open the template file (OpenFOAM file) for manipulation
        dict_file = open(template_dict_path + '/probeTemplate')  # noqa: SIM115, PTH123

        dict_lines = dict_file.readlines()
        dict_file.close()

        # Write writeInterval
        start_index = foam.find_keyword_line(dict_lines, 'writeInterval')
        dict_lines[start_index] = f'writeInterval \t{pressure_write_interval};\n'

        # Write writeControl
        start_index = foam.find_keyword_line(dict_lines, 'writeControl')
        if solver_type == 'pimpleFoam' and adjust_time_step:
            dict_lines[start_index] = 'writeControl \t{};\n'.format(
                'adjustableRunTime'
            )
        else:
            dict_lines[start_index] = 'writeControl \t{};\n'.format('timeStep')

        # Write writeInterval
        start_index = foam.find_keyword_line(dict_lines, 'writeInterval')
        if solver_type == 'pimpleFoam' and adjust_time_step:
            dict_lines[start_index] = (
                f'writeInterval \t{pressure_write_interval * time_step:.6f};\n'
            )
        else:
            dict_lines[start_index] = f'writeInterval \t{pressure_write_interval};\n'

        # Write fields to be motored
        start_index = foam.find_keyword_line(dict_lines, 'fields')
        dict_lines[start_index] = 'fields \t\t(p);\n'

        start_index = foam.find_keyword_line(dict_lines, 'probeLocations') + 2

        added_part = ''

        for i in range(len(imported_sampling_points)):
            added_part += f' ({imported_sampling_points[i][0]:.6f} {imported_sampling_points[i][1]:.6f} {imported_sampling_points[i][2]:.6f})\n'

        dict_lines.insert(start_index, added_part)

        # Write edited dict to file
        write_file_name = case_path + '/system/importedPressureSamplingPoints'

        if os.path.exists(write_file_name):  # noqa: PTH110
            os.remove(write_file_name)  # noqa: PTH107

        output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
        for line in dict_lines:
            output_file.write(line)

        output_file.close()


def write_base_forces_file(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    air_density = 1.0

    # Returns JSON object as a dictionary
    rm_data = json_data['resultMonitoring']

    num_stories = rm_data['numStories']  # noqa: F841
    floor_height = rm_data['floorHeight']  # noqa: F841
    center_of_rotation = rm_data['centerOfRotation']
    base_load_write_interval = rm_data['baseLoadWriteInterval']
    monitor_base_load = rm_data['monitorBaseLoad']  # noqa: F841

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/baseForcesTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write writeInterval
    start_index = foam.find_keyword_line(dict_lines, 'writeInterval')
    dict_lines[start_index] = f'writeInterval \t{base_load_write_interval};\n'

    # Write patch name to integrate forces on
    start_index = foam.find_keyword_line(dict_lines, 'patches')
    dict_lines[start_index] = 'patches \t({});\n'.format('building')

    # Write air density to rhoInf
    start_index = foam.find_keyword_line(dict_lines, 'rhoInf')
    dict_lines[start_index] = f'rhoInf \t\t{air_density:.4f};\n'

    # Write center of rotation
    start_index = foam.find_keyword_line(dict_lines, 'CofR')
    dict_lines[start_index] = (
        f'CofR \t\t({center_of_rotation[0]:.4f} {center_of_rotation[1]:.4f} {center_of_rotation[2]:.4f});\n'
    )

    # Write edited dict to file
    write_file_name = case_path + '/system/baseForces'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123

    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_story_forces_file(input_json_path, template_dict_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    air_density = 1.0

    # Returns JSON object as a dictionary
    rm_data = json_data['resultMonitoring']

    num_stories = rm_data['numStories']
    floor_height = rm_data['floorHeight']  # noqa: F841
    center_of_rotation = rm_data['centerOfRotation']
    story_load_write_interval = rm_data['storyLoadWriteInterval']
    monitor_base_load = rm_data['monitorBaseLoad']  # noqa: F841

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/storyForcesTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write writeInterval
    start_index = foam.find_keyword_line(dict_lines, 'writeInterval')
    dict_lines[start_index] = f'writeInterval \t{story_load_write_interval};\n'

    # Write patch name to integrate forces on
    start_index = foam.find_keyword_line(dict_lines, 'patches')
    dict_lines[start_index] = 'patches \t({});\n'.format('building')

    # Write air density to rhoInf
    start_index = foam.find_keyword_line(dict_lines, 'rhoInf')
    dict_lines[start_index] = f'rhoInf \t\t{air_density:.4f};\n'

    # Write center of rotation
    start_index = foam.find_keyword_line(dict_lines, 'CofR')
    dict_lines[start_index] = (
        f'CofR \t\t({center_of_rotation[0]:.4f} {center_of_rotation[1]:.4f} {center_of_rotation[2]:.4f});\n'
    )

    # Number of stories  as nBins
    start_index = foam.find_keyword_line(dict_lines, 'nBin')
    dict_lines[start_index] = f'    nBin \t{num_stories};\n'

    # Write story direction
    start_index = foam.find_keyword_line(dict_lines, 'direction')
    dict_lines[start_index] = f'    direction \t({0:.4f} {0:.4f} {1.0:.4f});\n'

    # Write edited dict to file
    write_file_name = case_path + '/system/storyForces'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123

    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_momentumTransport_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    turb_data = json_data['turbulenceModeling']

    simulation_type = turb_data['simulationType']
    RANS_type = turb_data['RANSModelType']  # noqa: N806
    LES_type = turb_data['LESModelType']  # noqa: N806
    DES_type = turb_data['DESModelType']  # noqa: N806

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/momentumTransportTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write type of the simulation
    start_index = foam.find_keyword_line(dict_lines, 'simulationType')
    dict_lines[start_index] = 'simulationType \t{};\n'.format(
        'RAS' if simulation_type == 'RANS' else simulation_type
    )

    if simulation_type == 'RANS':
        # Write RANS model type
        start_index = foam.find_keyword_line(dict_lines, 'RAS') + 2
        added_part = f'    model \t{RANS_type};\n'
        dict_lines.insert(start_index, added_part)

    elif simulation_type == 'LES':
        # Write LES SGS model type
        start_index = foam.find_keyword_line(dict_lines, 'LES') + 2
        added_part = f'    model \t{LES_type};\n'
        dict_lines.insert(start_index, added_part)

    elif simulation_type == 'DES':
        # Write DES model type
        start_index = foam.find_keyword_line(dict_lines, 'LES') + 2
        added_part = f'    model \t{DES_type};\n'
        dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/constant/momentumTransport'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_physicalProperties_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    wc_data = json_data['windCharacteristics']

    kinematic_viscosity = wc_data['kinematicViscosity']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/physicalPropertiesTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write type of the simulation
    start_index = foam.find_keyword_line(dict_lines, 'nu')
    dict_lines[start_index] = f'nu\t\t[0 2 -1 0 0 0 0] {kinematic_viscosity:.4e};\n'

    # Write edited dict to file
    write_file_name = case_path + '/constant/physicalProperties'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)
    output_file.close()


def write_transportProperties_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    wc_data = json_data['windCharacteristics']

    kinematic_viscosity = wc_data['kinematicViscosity']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/transportPropertiesTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write type of the simulation
    start_index = foam.find_keyword_line(dict_lines, 'nu')
    dict_lines[start_index] = f'nu\t\t[0 2 -1 0 0 0 0] {kinematic_viscosity:.3e};\n'

    # Write edited dict to file
    write_file_name = case_path + '/constant/transportProperties'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_fvSchemes_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    turb_data = json_data['turbulenceModeling']

    simulation_type = turb_data['simulationType']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + f'/fvSchemesTemplate{simulation_type}')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write edited dict to file
    write_file_name = case_path + '/system/fvSchemes'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_decomposeParDict_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(input_json_path + '/IsolatedBuildingCFD.json') as json_file:  # noqa: PTH123
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    ns_data = json_data['numericalSetup']

    num_processors = ns_data['numProcessors']

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/decomposeParDictTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write number of sub-domains
    start_index = foam.find_keyword_line(dict_lines, 'numberOfSubdomains')
    dict_lines[start_index] = f'numberOfSubdomains\t{num_processors};\n'

    # Write method of decomposition
    start_index = foam.find_keyword_line(dict_lines, 'decomposer')
    dict_lines[start_index] = 'decomposer\t\t{};\n'.format('scotch')

    # Write method of decomposition for OF-V9 and lower compatibility
    start_index = foam.find_keyword_line(dict_lines, 'method')
    dict_lines[start_index] = 'method\t\t{};\n'.format('scotch')

    # Write edited dict to file
    write_file_name = case_path + '/system/decomposeParDict'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_DFSRTurbDict_file(input_json_path, template_dict_path, case_path):  # noqa: N802, D103
    # Read JSON data
    with open(input_json_path + '/IsolatedBuildingCFD.json') as json_file:  # noqa: PTH123
        json_data = json.load(json_file)

    fMax = 200.0  # noqa: N806

    # Returns JSON object as a dictionary
    wc_data = json_data['windCharacteristics']
    ns_data = json_data['numericalSetup']

    wind_speed = wc_data['referenceWindSpeed']
    duration = ns_data['duration']

    # Generate a little longer duration to be safe
    duration = duration * 1.010

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(template_dict_path + '/DFSRTurbDictTemplate')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()

    # Write the end time
    start_index = foam.find_keyword_line(dict_lines, 'endTime')
    dict_lines[start_index] = f'endTime\t\t\t{duration:.4f};\n'

    # Write patch name
    start_index = foam.find_keyword_line(dict_lines, 'patchName')
    dict_lines[start_index] = 'patchName\t\t"{}";\n'.format('inlet')

    # Write cohUav
    start_index = foam.find_keyword_line(dict_lines, 'cohUav')
    dict_lines[start_index] = f'cohUav\t\t\t{wind_speed:.4f};\n'

    # Write fMax
    start_index = foam.find_keyword_line(dict_lines, 'fMax')
    dict_lines[start_index] = f'fMax\t\t\t{fMax:.4f};\n'

    # Write time step
    start_index = foam.find_keyword_line(dict_lines, 'timeStep')
    dict_lines[start_index] = f'timeStep\t\t{1.0 / fMax:.4f};\n'

    # Write edited dict to file
    write_file_name = case_path + '/constant/DFSRTurbDict'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def write_boundary_data_files(input_json_path, case_path):
    """This functions writes wind profile files in "constant/boundaryData/inlet"
    if TInf options are used for the simulation.
    """  # noqa: D205, D401, D404
    # Read JSON data
    with open(input_json_path + '/IsolatedBuildingCFD.json') as json_file:  # noqa: PTH123
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    boundary_data = json_data['boundaryConditions']
    geom_data = json_data['GeometricData']
    scale = 1.0 / float(geom_data['geometricScale'])
    norm_type = geom_data['normalizationType']
    building_height = scale * geom_data['buildingHeight']

    if boundary_data['inletBoundaryCondition'] == 'TInf':
        wind_profiles = np.array(boundary_data['inflowProperties']['windProfiles'])

        bd_path = case_path + '/constant/boundaryData/inlet/'

        # Write points file
        n_pts = np.shape(wind_profiles)[0]
        points = np.zeros((n_pts, 3))

        origin = np.array(geom_data['origin'])

        Ly = geom_data['domainWidth']  # noqa: N806
        Lf = geom_data['fetchLength']  # noqa: N806

        if norm_type == 'Relative':
            Ly *= building_height  # noqa: N806
            Lf *= building_height  # noqa: N806

        x_min = -Lf - origin[0]
        y_min = -Ly / 2.0 - origin[1]
        y_max = y_min + Ly

        points[:, 0] = x_min
        points[:, 1] = (y_min + y_max) / 2.0
        points[:, 2] = wind_profiles[:, 0]

        # Shift the last element of the y coordinate
        # a bit to make planer interpolation easier
        points[-1:, 1] = y_max

        foam.write_foam_field(points, bd_path + 'points')

        # Write wind speed file as a scalar field
        foam.write_scalar_field(wind_profiles[:, 1], bd_path + 'U')

        # Write Reynolds stress profile (6 columns -> it's a symmetric tensor field)
        foam.write_foam_field(wind_profiles[:, 2:8], bd_path + 'R')

        # Write length scale file (8 columns -> it's a tensor field)
        foam.write_foam_field(wind_profiles[:, 8:17], bd_path + 'L')


<<<<<<< HEAD
def write_vtk_plane_file(input_json_path, template_dict_path, case_path):

    #Read JSON data
    with open(input_json_path + "/IsolatedBuildingCFD.json") as json_file:
        json_data =  json.load(json_file)

    # Returns JSON object as a dictionary
    rm_data = json_data["resultMonitoring"]  # noqa: W291
    ns_data = json_data["numericalSetup"]
    solver_type = ns_data['solverType']
    time_step = ns_data['timeStep']


    vtk_planes = rm_data['vtkPlanes']
    write_interval = rm_data['vtkWriteInterval']

    if rm_data['monitorVTKPlane'] == False:
        return  # noqa: W291
  # noqa: W293
    if len(vtk_planes)==0:  # noqa: W291
        return

    #Write dict files for wind profiles
    for pln in vtk_planes:
        #Open the template file (OpenFOAM file) for manipulation
        dict_file = open(template_dict_path + "/vtkPlaneTemplate", "r")

        dict_lines = dict_file.readlines()
        dict_file.close()
  # noqa: W293
        #Write writeControl  # noqa: W291
        start_index = foam.find_keyword_line(dict_lines, "writeControl")  # noqa: W291
        if solver_type=="pimpleFoam":
            dict_lines[start_index] = "    writeControl \t{};\n".format("adjustableRunTime")
        else:
            dict_lines[start_index] = "    writeControl \t{};\n".format("timeStep")  # noqa: W291

        #Write writeInterval
        start_index = foam.find_keyword_line(dict_lines, "writeInterval")  # noqa: W291
        if solver_type=="pimpleFoam":
            dict_lines[start_index] = "    writeInterval \t{:.6f};\n".format(write_interval*time_step)
        else:
            dict_lines[start_index] = "    writeInterval \t{};\n".format(write_interval)

        #Write start and end time for the section  # noqa: W291
        start_time = pln['startTime']
        end_time = pln['endTime']
        start_index = foam.find_keyword_line(dict_lines, "timeStart")  # noqa: W291
        dict_lines[start_index] = "    timeStart \t\t{:.6f};\n".format(start_time)  # noqa: W291

        start_index = foam.find_keyword_line(dict_lines, "timeEnd")  # noqa: W291
        dict_lines[start_index] = "    timeEnd \t\t{:.6f};\n".format(end_time)  # noqa: W291

        #Write name of the profile  # noqa: W291
        name = pln["name"]
        start_index = foam.find_keyword_line(dict_lines, "planeName")  # noqa: W291
        dict_lines[start_index] = "{}\n".format(name)  # noqa: W291

        #Write field type  # noqa: W291
        field_type = pln["field"]
        start_index = foam.find_keyword_line(dict_lines, "fields")  # noqa: W291

        if field_type=="Velocity":
            dict_lines[start_index] = "    fields \t\t({});\n".format("U")
        if field_type=="Pressure":
            dict_lines[start_index] = "    fields \t\t({});\n".format("p")

        #Write normal and point coordinates
        point_x = pln["pointX"]
        point_y = pln["pointY"]
        point_z = pln["pointZ"]

        normal_axis = pln["normalAxis"]

        start_index = foam.find_keyword_line(dict_lines, "point")  # noqa: W291
        dict_lines[start_index] = "\t    point\t\t({:.6f} {:.6f} {:.6f});\n".format(point_x, point_y, point_z)

        start_index = foam.find_keyword_line(dict_lines, "normal")  # noqa: W291
        if normal_axis=="X":  # noqa: W291
            dict_lines[start_index] = "\t    normal\t\t({} {} {});\n".format(1, 0, 0)
        if normal_axis=="Y":  # noqa: W291
            dict_lines[start_index] = "\t    normal\t\t({} {} {});\n".format(0, 1, 0)
        if normal_axis=="Z":  # noqa: W291
            dict_lines[start_index] = "\t    normal\t\t({} {} {});\n".format(0, 0, 1)

        #Write edited dict to file
        write_file_name = case_path + "/system/" + name
  # noqa: W293
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
  # noqa: W293
        output_file = open(write_file_name, "w+")
        for line in dict_lines:
            output_file.write(line)
        output_file.close()


if __name__ == '__main__':  # noqa: W291
  # noqa: W293
=======
if __name__ == '__main__':
>>>>>>> upstream/master
    input_args = sys.argv

    # Set filenames
    input_json_path = sys.argv[1]
    template_dict_path = sys.argv[2]
    case_path = sys.argv[3]

    # Read JSON data
    with open(input_json_path + '/IsolatedBuildingCFD.json') as json_file:  # noqa: PTH123
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    turb_data = json_data['turbulenceModeling']

    simulation_type = turb_data['simulationType']
    RANS_type = turb_data['RANSModelType']
    LES_type = turb_data['LESModelType']

    # Write blockMesh
    write_block_mesh_dict(input_json_path, template_dict_path, case_path)

    # Create and write the building .stl file
    # Also, import STL file if the shape is complex, the check is done inside the function
    write_building_stl_file(input_json_path, case_path)

    # Create and write the SnappyHexMeshDict file
    write_snappy_hex_mesh_dict(input_json_path, template_dict_path, case_path)

    # Create and write the surfaceFeaturesDict file
    write_surfaceFeaturesDict_file(input_json_path, template_dict_path, case_path)

    # Write files in "0" directory
    write_U_file(input_json_path, template_dict_path, case_path)
    write_p_file(input_json_path, template_dict_path, case_path)
    write_nut_file(input_json_path, template_dict_path, case_path)
    write_k_file(input_json_path, template_dict_path, case_path)

    if simulation_type == 'RANS' and RANS_type == 'kEpsilon':
        write_epsilon_file(input_json_path, template_dict_path, case_path)

    # Write control dict
    write_controlDict_file(input_json_path, template_dict_path, case_path)

    # Write results to be monitored
    write_base_forces_file(input_json_path, template_dict_path, case_path)
    write_story_forces_file(input_json_path, template_dict_path, case_path)
<<<<<<< HEAD
    write_generated_pressure_probes_file(input_json_path, template_dict_path, case_path)
    write_imported_pressure_probes_file(input_json_path, template_dict_path, case_path)
    write_vtk_plane_file(input_json_path, template_dict_path, case_path)
=======
    write_generated_pressure_probes_file(
        input_json_path, template_dict_path, case_path
    )
    write_imported_pressure_probes_file(
        input_json_path, template_dict_path, case_path
    )
>>>>>>> upstream/master

    # Write fvSolution dict
    write_fvSolution_file(input_json_path, template_dict_path, case_path)

    # Write fvSchemes dict
    write_fvSchemes_file(input_json_path, template_dict_path, case_path)

    # Write momentumTransport dict
    write_momentumTransport_file(input_json_path, template_dict_path, case_path)

    # Write physicalProperties dict
    write_physicalProperties_file(input_json_path, template_dict_path, case_path)

    # Write transportProperties (physicalProperties in OF-10) dict for OpenFOAM-9 and below
    write_transportProperties_file(input_json_path, template_dict_path, case_path)

    # Write decomposeParDict
    write_decomposeParDict_file(input_json_path, template_dict_path, case_path)

    # #Write DFSRTurb dict
    # write_DFSRTurbDict_file(input_json_path, template_dict_path, case_path)

    # Write TInf files
    write_boundary_data_files(input_json_path, case_path)
<<<<<<< HEAD


  # noqa: W293
=======
>>>>>>> upstream/master
