# -*- coding: utf-8 -*-  # noqa: INP001, D100, UP009
# Copyright (c) 2016-2017, The Regents of the University of California (Regents).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
# THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
# PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

#
# Contributors:
# Abiy Melaku

import json
import os
import sys
import foam_dict_reader as foam
import numpy as np
from stl import mesh
# from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
# import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString


def modify_controlDict_file(case_path):  # noqa: N802, D103

    # Open the template file (OpenFOAM file) for manipulation
    dict_file = open(case_path + '/system/controlDict')  # noqa: SIM115, PTH123

    dict_lines = dict_file.readlines()
    dict_file.close()


    # Function Objects ##############################

    # Find function object location
    start_index = foam.find_keyword_line(dict_lines, 'functions') + 2

    # Write story loads functionObjects
    added_part = '    #includeFunc  componentPressureSamplingPoints\n'
    dict_lines.insert(start_index, added_part)

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

def write_component_probes_file(  # noqa: D103
    input_json_path,
    compt_json_path,
    template_dict_path,
    probe_points,
    case_path,
):
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        json_data = json.load(json_file)

    with open(compt_json_path, encoding='utf-8') as json_file:
        compt_json_data = json.load(json_file)

    sample_time_interval = compt_json_data['sampleTimeInterval']


    # Returns JSON object as a dictionary
    ns_data = json_data['numericalSetup']
    solver_type = ns_data['solverType']
    time_step = ns_data['timeStep']
    adjust_time_step = ns_data['adjustTimeStep']


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
            f'writeInterval \t{sample_time_interval * time_step:.6f};\n'
        )
    else:
        dict_lines[start_index] = f'writeInterval \t{sample_time_interval};\n'

    # Write fields to be motored
    start_index = foam.find_keyword_line(dict_lines, 'fields')
    dict_lines[start_index] = 'fields \t\t(p);\n'

    start_index = foam.find_keyword_line(dict_lines, 'probeLocations') + 2

    added_part = ''

    for i in range(np.shape(probe_points)[0]):
        added_part += f' ({probe_points[i,0]:.6f} {probe_points[i,1]:.6f} {probe_points[i,2]:.6f})\n'

    dict_lines.insert(start_index, added_part)

    # Write edited dict to file
    write_file_name = case_path + '/system/componentPressureSamplingPoints'

    if os.path.exists(write_file_name):  # noqa: PTH110
        os.remove(write_file_name)  # noqa: PTH107

    output_file = open(write_file_name, 'w+')  # noqa: SIM115, PTH123
    for line in dict_lines:
        output_file.write(line)

    output_file.close()


def create_component_geometry(input_json_path, compt_json_path, template_dict_path, case_path):  
    
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        iso_json_data = json.load(json_file)

    with open(compt_json_path, encoding='utf-8') as json_file:
        compt_json_data = json.load(json_file)

    geom_data = iso_json_data['GeometricData']
    scale = 1.0/geom_data['geometricScale']
    wind_dxn = 0.0
    thickness = 0.02*scale*geom_data["buildingWidth"]
    compts = compt_json_data['components']
    
    if compt_json_data['considerWindDirection']:
        wind_dxn = geom_data['windDirection']

    block_meshes = []
    block_names = []
    probe_points = []
    probe_areas = []

    #Write stl files for each component
    for compt_i in compts:
        geoms = compt_i["geometries"]
        sampling_density = compt_i["samplingDensity"]
        mesh_obj = []
        
        for geom_i in geoms:
            if geom_i["shape"] == "circle":
                mesh_obj, points, areas = make_circular_component_geometry(wind_dxn, sampling_density, thickness, scale, compt_i["componentId"], geom_i, case_path)
            elif geom_i["shape"] == "polygon":
                mesh_obj, points, areas = make_polygon_component_geometry(wind_dxn, sampling_density,thickness, scale, compt_i["componentId"], geom_i, case_path)
            elif geom_i["shape"] == "rectangle":
                mesh_obj, points, areas = make_rectangular_component_geometry(wind_dxn, sampling_density,thickness, scale, compt_i["componentId"], geom_i, case_path)
            
            block_meshes.append(mesh_obj)
            block_names.append("component_" + str(compt_i["componentId"]) + "_geometry_" + str(geom_i["geometryId"]))
            probe_points.append(points)
            probe_areas.append(areas)
            
    # Save the STL file
    stl_file_name = case_path + '/constant/geometry/components/components_geometry.stl'
    points_file_name = case_path + '/constant/geometry/components/probe_points.txt'
    indexes_file_name = case_path + '/constant/geometry/components/probe_indexes.txt'
    areas_file_name = case_path + '/constant/geometry/components/probe_areas.txt'

    combine_and_write_named_stl(block_meshes=block_meshes,block_names=block_names, output_file=stl_file_name)
    
    n_elements = len(probe_points)
    compt_end_indx = [0]*n_elements

    for i in range(n_elements):
        if i == 0:
            compt_end_indx[i] = len(probe_points[i])
        else:
            compt_end_indx[i] = compt_end_indx[i-1] + len(probe_points[i])
        
    np.savetxt(indexes_file_name, compt_end_indx, fmt='%d', delimiter='\t')

    points = np.array([point for sublist in probe_points for point in sublist])
    areas  = np.array([area for sublist in probe_areas for area in sublist])

    write_points_to_file(points, points_file_name)
    write_areas_to_file(areas, areas_file_name)
    
    write_component_probes_file(
        input_json_path,
        compt_json_path,
        template_dict_path,
        points,
        case_path)
    
    modify_controlDict_file(case_path)

def rotate_to_normal(vertices, normal):
    """
    Rotates the vertices to ensure they are perpendicular to the given normal vector.

    :param vertices: Array of vertex coordinates.
    :param normal: Target normal vector.
    :return: Rotated vertices.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1])  # Disk is initially aligned along the Z-axis

    # Calculate the rotation axis and angle
    rotation_axis = np.cross(z_axis, normal)
    if np.allclose(rotation_axis, 0):  # No rotation needed
        rotation_matrix = np.eye(3)
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.dot(z_axis, normal))
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = (
            np.eye(3) +
            np.sin(rotation_angle) * K +
            (1 - np.cos(rotation_angle)) * np.dot(K, K)
        )
    # Apply the rotation
    return np.dot(vertices, rotation_matrix.T)


def rotation_matrix_z(angle_deg):
    """
    Generate a rotation matrix for rotating points around the Z-axis.

    Parameters:
        angle_deg (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The 3x3 rotation matrix for Z-axis rotation.
    """
    angle_rad = np.radians(angle_deg)
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1],
    ])


def bounding_rectangle_3d(points):
    """
    Calculates the bounding rectangle of a 3D polygon on its plane.

    Parameters:
        points (np.ndarray): A NumPy array of shape (n, 3), where n is the number of points.

    Returns:
        np.ndarray: A NumPy array of shape (4, 3) containing the vertices of the bounding rectangle in 3D.
    """
    # Step 1: Calculate the normal of the polygon's plane
    centroid = np.mean(points, axis=0)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal = normal / np.linalg.norm(normal)  # Normalize the normal

    # Step 2: Find two orthogonal vectors on the plane
    arbitrary_vector = np.array([1, 0, 0]) if normal[0] == 0 else np.array([0, 1, 0])
    plane_x = np.cross(normal, arbitrary_vector)
    plane_x = plane_x / np.linalg.norm(plane_x)  # Normalize
    plane_y = np.cross(normal, plane_x)  # Orthogonal to both normal and plane_x

    # Step 3: Project points onto the 2D plane defined by (plane_x, plane_y)
    projected_points = np.dot(points - centroid, np.column_stack((plane_x, plane_y)))

    # Step 4: Compute the 2D bounding box
    min_x, min_y = np.min(projected_points, axis=0)
    max_x, max_y = np.max(projected_points, axis=0)

    # Step 5: Map the bounding box back to 3D
    rectangle_2d = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    bounding_rectangle_3d = np.dot(rectangle_2d, np.column_stack((plane_x, plane_y)).T) + centroid

    return bounding_rectangle_3d

def calculate_spacing(points, density):
    """
    Calculate the spacing of grid points based on the smallest non-zero dimension of a bounding box.

    Parameters:
        bbox_min (tuple or list): The minimum coordinates of the bounding box (x_min, y_min, z_min).
        bbox_max (tuple or list): The maximum coordinates of the bounding box (x_max, y_max, z_max).
        density (int): The number of grid points along the smallest non-zero dimension.

    Returns:
        float: The calculated spacing for the grid points.
    """
    
    tolerance = 1.0e-6
    
    # Step 1: Get the bounding rectangle in 3D
    bounding_rectangle = bounding_rectangle_3d(points)

    # Step 2: Calculate the edge lengths of the bounding rectangle
    edge1_length = np.linalg.norm(bounding_rectangle[1] - bounding_rectangle[0])
    edge2_length = np.linalg.norm(bounding_rectangle[2] - bounding_rectangle[1])

    # Step 3: Find the smallest non-zero dimension
    dimensions = np.array([edge1_length, edge2_length])
    smallest_dimension = np.min(dimensions[dimensions > tolerance])
    
    # Calculate the spacing
    spacing = smallest_dimension/density
    
    return spacing

def make_circular_component_geometry(wind_dxn, sampling_density, thickness, geom_scale, compt_id, geom_json, case_path): 

    radius = geom_json["radius"]
    origin = np.array(geom_json["origin"])
    normal = np.array(geom_json["normal"])
    global_origin = np.array([0.0, 0.0, 0.0])

    num_points = 100  # Number of points approximating the circle
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.array([[radius * np.cos(angle), radius * np.sin(angle), 0] for angle in angles])

    # Rotate the points to align with the normal vector
    points = rotate_to_normal(points, normal) + origin

    # Scale the points
    points = scale_vertices(points, global_origin, geom_scale) 
    
    
    #Account wind direction    
    rot_matrix = rotation_matrix_z(wind_dxn)
    points = np.dot(points, rot_matrix.T)

    bbox_vertices, bbox_min, bbox_max = compute_bounding_box(points)
        
    spacing = calculate_spacing(points, sampling_density)   

    # Compute normal and basis
    normal = calculate_normal(points)

    probes_orig =  generate_grid_on_plane_with_bounding_box(points[0], normal, bbox_min, bbox_max, spacing=spacing)  
    
    probe_points = []
    
    for i in range(np.shape(probes_orig)[0]):
        if is_point_in_polygon(probes_orig[i], points):
            probe_points.append(probes_orig[i])
    
    probe_points = np.array(probe_points)
    

    # Generate the Voronoi diagram and clip it
    clipped_edges, probe_areas = create_voronoi_cells_3d(probe_points, normal, points[0], points)
    
    # write_areas_to_file(cell_areas, areas_file_name)

    # Plot the result
    # plot_voronoi_3d(clipped_edges, probes_final, points)


    # Offset top and bottom circles to create thickness
    top_circle = points + normal * (thickness / 2)
    bottom_circle = points - normal * (thickness / 2)

    # Create faces for the sides of the disk (sidewalls)
    faces = []
    for i in range(num_points):
        next_i = (i + 1) % num_points
        # Two triangles for each side face
        faces.append([top_circle[i], bottom_circle[i], bottom_circle[next_i]])
        faces.append([top_circle[i], bottom_circle[next_i], top_circle[next_i]])

    # Convert faces to a numpy array
    faces = np.array(faces)

    # Create the STL mesh
    disk_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        disk_mesh.vectors[i] = f
        
    
    
    return disk_mesh, probe_points, probe_areas


def make_polygon_component_geometry(wind_dxn, sampling_density, thickness, geom_scale, compt_id, geom_json, case_path): 
    
    """
    Generate an extruded geometry from a planar polygon and save as an STL file.
    """
    points = np.array(geom_json["points"])
    global_origin = np.array([0.0, 0.0, 0.0])

    # Ensure the points form a closed loop
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Scale vertices
    points = scale_vertices(points, global_origin, geom_scale)

    #Account wind direction    
    rot_matrix = rotation_matrix_z(wind_dxn)
    points = np.dot(points, rot_matrix.T)

    bbox_vertices, bbox_min, bbox_max = compute_bounding_box(points)
    
    
    spacing = calculate_spacing(points, sampling_density)   


    # Compute normal and basis
    normal = calculate_normal(points)

    probes_orig =  generate_grid_on_plane_with_bounding_box(points[0], normal, bbox_min, bbox_max, spacing=spacing)  
    
    probe_points = []
    
    for i in range(np.shape(probes_orig)[0]):
        if is_point_in_polygon(probes_orig[i], points):
            probe_points.append(probes_orig[i])
    
    probe_points = np.array(probe_points)

    # Generate the Voronoi diagram and clip it
    clipped_edges, probe_areas = create_voronoi_cells_3d(probe_points, normal, points[0], points)

    # Plot the result
    # plot_voronoi_3d(clipped_edges, probes_final, points)


    faces = []
    # Sidewalls
    for i in range(len(points) - 1):
        p1_top = points[i] + normal * (thickness / 2)
        p2_top = points[i + 1] + normal * (thickness / 2)
        p1_bottom = points[i] - normal * (thickness / 2)
        p2_bottom = points[i + 1] - normal * (thickness / 2)

        # Two triangles per side face
        faces.append([p1_bottom, p1_top, p2_top])
        faces.append([p1_bottom, p2_top, p2_bottom])

    # Convert faces to numpy array
    faces = np.array(faces)

    # Create the STL mesh
    extruded_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, triangle in enumerate(faces):
        extruded_mesh.vectors[i] = triangle

    
    return extruded_mesh, probe_points, probe_areas



def make_rectangular_component_geometry(wind_dxn, sampling_density, thickness, geom_scale, compt_id, geom_json, case_path): 
    
    width = geom_json["width"]  # Rectangle width
    height = geom_json["height"]  # Rectangle height
    origin = np.array(geom_json["origin"])
    normal = np.array(geom_json["normal"])
    global_origin = np.array([0.0, 0.0, 0.0])
    n_sides = 4

    # Define rectangle vertices
    half_width = width / 2
    half_height = height / 2
    points = np.array([
        [-half_width, -half_height,  0],
        [ half_width, -half_height, 0],
        [half_width, half_height, 0],
        [-half_width, half_height, 0]
    ])

    # Rotate the rectangle points to align with the normal vector
    points = rotate_to_normal(points, normal) + origin

    # Scale the points
    points = scale_vertices(points, global_origin, geom_scale)
    
    #Account wind direction    
    rot_matrix = rotation_matrix_z(wind_dxn)
    points = np.dot(points, rot_matrix.T)
    
    bbox_vertices, bbox_min, bbox_max = compute_bounding_box(points)
    
    spacing = calculate_spacing(points, sampling_density)   

    # Compute normal and basis
    normal = calculate_normal(points)

    probes_orig =  generate_grid_on_plane_with_bounding_box(points[0], normal, bbox_min, bbox_max, spacing=spacing)  
    
    probe_points = []
    
    for i in range(np.shape(probes_orig)[0]):
        if is_point_in_polygon(probes_orig[i], points):
            probe_points.append(probes_orig[i])
    
    probe_points = np.array(probe_points)

    # Generate the Voronoi diagram and clip it
    clipped_edges, probe_areas = create_voronoi_cells_3d(probe_points, normal, points[0], points)
    
 
    # Plot the result
    # plot_voronoi_3d(clipped_edges, probes_final, points)

    # Offset top and bottom rectangles to create thickness
    top_rectangle = points + normal * (thickness / 2)
    bottom_rectangle = points - normal * (thickness / 2)

    # Create sidewall faces
    faces = []
    for i in range(n_sides):  
        next_i = (i + 1) % 4
        # Two triangles for each side face
        faces.append([top_rectangle[i], bottom_rectangle[i], bottom_rectangle[next_i]])
        faces.append([top_rectangle[i], bottom_rectangle[next_i], top_rectangle[next_i]])

    # Convert faces to a numpy array
    faces = np.array(faces)

    # Create the STL mesh
    rectangle_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        rectangle_mesh.vectors[i] = f
    
    
    return rectangle_mesh, probe_points, probe_areas


def write_points_to_file(points, filename="points.txt"):
    """
    Write a set of 3D points into a CSV file with headers X, Y, Z.

    Parameters:
        points (np.ndarray): Nx3 array of 3D points (X, Y, Z).
        filename (str): Output CSV file name.
    """
    # Ensure the input is a valid Nx3 array
    if points.shape[1] != 3:
        raise ValueError("Input points must have shape (N, 3)")

    # Save the points to a CSV file with headers
    # header = "X,Y,Z"
    np.savetxt(filename, points, delimiter="\t",comments="", fmt="%.6e")


def write_areas_to_file(areas, filename="areas.txt"):
    """
    Write a set of 3D points into a CSV file with headers X, Y, Z.

    Parameters:
        areas (np.ndarray): Nx1.
        filename (str): Output txt file name.
    """

    # Save the points to a CSV file with headers
    np.savetxt(filename, areas, delimiter="\t", comments="", fmt="%.6e")


def generate_grid_on_plane_with_bounding_box(base_point, normal, bbox_min, bbox_max, spacing=1.0):
    """
    Generate a regular grid of points on a 3D plane within a 3D bounding box.

    Parameters:
        base_point (np.ndarray): A point on the plane (3D coordinate, shape (3,)).
        normal (np.ndarray): Normal vector of the plane (shape (3,)).
        bbox_min (np.ndarray): Minimum corner of the 3D bounding box (shape (3,)).
        bbox_max (np.ndarray): Maximum corner of the 3D bounding box (shape (3,)).
        spacing (float): Distance between grid points.

    Returns:
        grid_points (np.ndarray): Nx3 array of grid points on the 3D plane within the bounding box.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Step 1: Create an orthonormal basis (u, v) for the plane
    # Find an arbitrary vector not parallel to the normal
    arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arbitrary_vector)
    u /= np.linalg.norm(u)  # Normalize u
    v = np.cross(normal, u)  # Compute v as perpendicular to both normal and u

    # Step 2: Determine grid range in u-v plane
    # Project the bounding box corners onto the plane
    bbox_min = np.array(bbox_min)
    bbox_max = np.array(bbox_max)

    # Find grid limits in the u and v directions
    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ])

    # Project corners onto the u-v plane
    uv_corners = np.dot(corners - base_point, np.vstack([u, v]).T)
    
    offset_distance = spacing/2.0

    
    uv_min = np.min(uv_corners, axis=0) + offset_distance
    uv_max = np.max(uv_corners, axis=0) - offset_distance
    
    # Center alignment adjustment: Calculate the center of the bounding box in u-v space
    uv_center = (uv_min + uv_max)/2.0

    # Step 3: Generate grid points in the u-v plane
    
    npoints_u = int((uv_max[0] - uv_min[0])/spacing)
    npoints_v = int((uv_max[1] - uv_min[1])/spacing)
                    
    u_coords = np.linspace(uv_min[0], uv_max[0], npoints_u, endpoint=True)
    v_coords = np.linspace(uv_min[1], uv_max[1], npoints_v, endpoint=True)

    # Adjust coordinates to ensure they are symmetric around the center
    u_offset = (uv_center[0] - (u_coords[0] + u_coords[-1]) / 2)
    v_offset = (uv_center[1] - (v_coords[0] + v_coords[-1]) / 2)
    
    
    u_coords += u_offset
    v_coords += v_offset

    
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    grid_uv = np.vstack([u_grid.ravel(), v_grid.ravel()]).T

    # Step 4: Map grid points back to 3D space
    grid_points = np.array([base_point + u * uv[0] + v * uv[1] for uv in grid_uv])

    # Step 5: Filter points to ensure they lie within the bounding box
    mask = np.all((grid_points >= bbox_min) & (grid_points <= bbox_max), axis=1)
    grid_points_filtered = grid_points[mask]

    return grid_points_filtered





# Function to scale geometry around the origin
def scale_vertices(vertices, origin, scale_factor):
    """
    Scales the vertices relative to the given origin by the scale factor.

    :param vertices: Array of vertex coordinates.
    :param origin: Origin point as a numpy array.
    :param scale_factor: Scaling factor.
    :return: Scaled vertices.
    """
    return (vertices - origin) * scale_factor + origin


# Function to calculate the normal vector from polygon points
def calculate_normal(points):
    """
    Calculates the normal vector of the polygon from its vertices using the cross product.

    :param points: Array of polygon points (must be at least 3 points).
    :return: Normal vector of the polygon.
    """
    # Calculate two vectors in the plane of the polygon
    vec1 = points[1] - points[0]
    vec2 = points[2] - points[0]
    
    # Calculate the normal using the cross product
    normal = np.cross(vec1, vec2)
    
    # Normalize the normal vector
    return normal / np.linalg.norm(normal)

def compute_bounding_box(points):
    """
    Compute the axis-aligned bounding box (AABB) for a set of 3D points.

    Parameters:
        points (np.ndarray): Nx3 array of 3D points.

    Returns:
        bbox_vertices (np.ndarray): 8 vertices of the bounding box.
    """
    # Compute the minimum and maximum corners of the bounding box
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    
    # Define the 8 vertices of the bounding box
    bbox_vertices = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],  # 0: Min corner
        [bbox_max[0], bbox_min[1], bbox_min[2]],  # 1
        [bbox_max[0], bbox_max[1], bbox_min[2]],  # 2
        [bbox_min[0], bbox_max[1], bbox_min[2]],  # 3
        [bbox_min[0], bbox_min[1], bbox_max[2]],  # 4
        [bbox_max[0], bbox_min[1], bbox_max[2]],  # 5
        [bbox_max[0], bbox_max[1], bbox_max[2]],  # 6: Max corner
        [bbox_min[0], bbox_max[1], bbox_max[2]],  # 7
    ])
    return bbox_vertices, bbox_min, bbox_max

def save_bounding_box_as_stl(bbox_vertices, filename="bounding_box.stl"):
    """
    Save the bounding box as an STL file using numpy-stl.

    Parameters:
        bbox_vertices (np.ndarray): 8 vertices of the bounding box.
        filename (str): Output STL file name.
    """
    # Define the 12 triangular faces of the bounding box using vertex indices
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Side face 1
        [1, 2, 6], [1, 6, 5],  # Side face 2
        [2, 3, 7], [2, 7, 6],  # Side face 3
        [3, 0, 4], [3, 4, 7],  # Side face 4
    ])

    # Create an empty mesh with 12 faces
    bbox_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    # Assign the vertices to each face
    for i, face in enumerate(faces):
        for j in range(3):
            bbox_mesh.vectors[i][j] = bbox_vertices[face[j]]

    # Save the mesh to an STL file
    bbox_mesh.save(filename)

def is_point_in_polygon(probe, polygon, tol=1e-6):
    """
    Check if a probe point lies inside a polygon defined in 3D space.

    Parameters:
        probe (np.ndarray): The 3D probe point to check (shape (3,)).
        polygon (np.ndarray): Nx3 array of vertices defining the polygon in 3D space.
        tol (float): Tolerance for floating-point comparisons.

    Returns:
        bool: True if the point lies inside the polygon, False otherwise.
    """
    # Step 1: Check if the point lies on the plane
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])  # Plane normal
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    
    # Plane equation: (point - base_point) · normal = 0
    plane_distance = np.dot(probe - polygon[0], normal)
    if abs(plane_distance) > tol:
        return False  # The point is not on the plane

    # Step 2: Project the polygon and the probe point onto a 2D plane
    # Define orthonormal basis (u, v) on the plane
    arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arbitrary_vector)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project points to 2D UV space
    polygon_2d = np.dot(polygon - polygon[0], np.vstack([u, v]).T)
    probe_2d = np.dot(probe - polygon[0], np.vstack([u, v]).T)

    # Step 3: Check if the point lies inside the 2D polygon using ray-casting
    return is_point_in_2d_polygon(probe_2d, polygon_2d)

def is_point_in_2d_polygon(point, polygon):
    """
    Check if a 2D point lies inside a 2D polygon using the ray-casting algorithm.

    Parameters:
        point (np.ndarray): The 2D point to check (shape (2,)).
        polygon (np.ndarray): Nx2 array of vertices defining the 2D polygon.

    Returns:
        bool: True if the point lies inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    # Iterate through each edge of the polygon
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]  # Next vertex (wrap around)

        # Check if the ray intersects the edge
        if (yi > y) != (yj > y):  # Edge crosses the ray
            intersect_x = xi + (xj - xi) * (y - yi) / (yj - yi)  # X-coordinate of intersection
            if x < intersect_x:
                inside = not inside  # Toggle the inside flag

    return inside


def combine_and_write_named_stl(block_meshes, block_names, output_file):
    """
    Combine multiple STL meshes into a single ASCII STL file, 
    keeping each object separate with its own name and block.

    Parameters:
        input_files (list of str): List of input STL file paths.
        output_file (str): Path to the output STL file.
    """
    # Open the output file for writing in ASCII mode
    with open(output_file, 'w') as out_file:

        # Loop through each input STL file
        for i, msh in enumerate(block_meshes):
            
            # Use file name (without path and extension) as the block name
            block_name = block_names[i]
            out_file.write(f"solid {block_name}\n")  # Start a new solid block with a name
            
            # Write each facet (triangle) to the ASCII STL file
            for triangle in msh.vectors:
                # Calculate face normal inline
                v1, v2, v3 = triangle
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm != 0:
                    normal = normal / norm  # Normalize the normal vector
                else:
                    normal = np.array([0.0, 0.0, 0.0])

                out_file.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                out_file.write("    outer loop\n")
                
                for vertex in triangle:
                    out_file.write(f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                out_file.write("    endloop\n")
                out_file.write("  endfacet\n")
            
            out_file.write(f"endsolid {block_name}\n")  # End the block for the current solid
        
    print(f"Merged STL file written to: {output_file}")



def create_voronoi_cells_3d(points, plane_normal, plane_point, clip_polygon, offset_distance=0.5):
    """
    Generate Voronoi cells for each input point, clip them to a polygon boundary,
    ensure one bounded Voronoi cell per point in 3D space, and calculate their areas.

    Parameters:
        points (np.ndarray): Nx3 array of 3D points for the Voronoi diagram.
        plane_normal (np.ndarray): Normal vector of the 3D plane.
        plane_point (np.ndarray): A point on the 3D plane.
        clip_polygon (np.ndarray): Mx3 array of 3D vertices defining the clipping polygon.
        offset_distance (float): Distance to offset the polygon outward.

    Returns:
        voronoi_cells_3d (list): List of clipped Voronoi cells (edges) in 3D space.
        cell_areas (list): List of areas for each Voronoi cell.
    """
    # Step 1: Define the plane and create orthonormal basis
    normal = plane_normal / np.linalg.norm(plane_normal)
    arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, arbitrary_vector)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Basis transformation matrix
    basis = np.vstack([u, v, normal]).T

    # Step 2: Project points and clipping polygon to 2D
    points_2d = np.dot(points - plane_point, basis[:, :2])
    clip_polygon_2d = np.dot(clip_polygon - plane_point, basis[:, :2])
    clip_polygon_shape = Polygon(clip_polygon_2d)

    # Step 3: Offset the polygon outward
    offset_polygon_shape = clip_polygon_shape.buffer(offset_distance, join_style=2)

    # Step 4: Generate artificial points along the offset polygon
    boundary_points = np.array(offset_polygon_shape.exterior.coords[:-1])  # Exclude closing point
    all_points_2d = np.vstack([points_2d, boundary_points])

    # Step 5: Compute the Voronoi diagram in 2D
    vor = Voronoi(all_points_2d)

    # Step 6: Extract and clip Voronoi regions
    voronoi_cells = []
    cell_areas = []  # List to store the areas of the Voronoi cells
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue  # Skip regions with infinite edges

        # Get the vertices of the Voronoi region
        region_vertices = vor.vertices[region]
        region_polygon = Polygon(region_vertices)

        # Clip the region polygon to the original boundary
        clipped_region = region_polygon.intersection(clip_polygon_shape)

        # Only process valid clipped regions
        if isinstance(clipped_region, Polygon):
            # Store the area of the clipped region
            cell_areas.append(clipped_region.area)

            # Map the clipped region back to 3D as a set of edges
            edges = np.array(clipped_region.exterior.coords)
            edges_3d = [plane_point + u * coord[0] + v * coord[1] for coord in edges]
            voronoi_cells.append(np.array(edges_3d))
        else:
            cell_areas.append(0.0)  # If no valid clipped region exists, area is 0

    return voronoi_cells, cell_areas


# def plot_voronoi_3d(clipped_edges_3d, points, clip_polygon):
#     """
#     Plot the clipped Voronoi diagram on the 3D plane.

#     Parameters:
#         clipped_edges_3d (list): List of clipped Voronoi edges in 3D space.
#         points (np.ndarray): Nx3 array of original 3D points.
#         clip_polygon (np.ndarray): Mx3 array of 3D vertices defining the clipping polygon.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the original points
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', label='Points')

#     # Plot the clipping polygon
#     clip_polygon_closed = np.vstack([clip_polygon, clip_polygon[0]])

#     # Plot the Voronoi edges
#     for edge in clipped_edges_3d:
#         edge = np.atleast_2d(edge)  # Ensure edge is at least 2D
#         if edge.shape[0] < 2:  # Skip if edge has fewer than 2 points
#             continue
#         ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color='green')
    
#     ax.plot(clip_polygon_closed[:, 0], clip_polygon_closed[:, 1], clip_polygon_closed[:, 2], color='blue', label='Clipping Polygon')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.legend()
#     plt.show()


if __name__ == '__main__':  
    
    input_args = sys.argv

    # # Set filenames
    # input_json_path = sys.argv[1]
    # compt_json_path = sys.argv[2]
    # template_dict_path = sys.argv[3]
    # case_path = sys.argv[4]

    input_json_path = "C:\\Users\\fanta\\Documents\\WE-UQ\\LocalWorkDir\\IsolatedBuildingCFD\\constant\\simCenter\\input\\"
    compt_json_path = "C:\\Users\\fanta\\SimCenter\\WBS_Items\\PBWE\\CC_Pressure_EDP_Example_TPU_V1.json"
    template_dict_path = "C:\\Users\\fanta\\SimCenter\\SourceCode\\NHERI-SimCenter\\SimCenterBackendApplications\\modules\\createEVENT\\IsolatedBuildingCFD\\templateOF10Dicts\\"
    case_path = "C:\\Users\\fanta\\Documents\\WE-UQ\\LocalWorkDir\\IsolatedBuildingCFD"

    # Write the component's geometry
    create_component_geometry(input_json_path, compt_json_path, template_dict_path, case_path)
    
    
    

