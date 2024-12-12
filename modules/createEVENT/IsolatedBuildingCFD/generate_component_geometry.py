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



def create_component_geometry(input_json_path, compt_json_path, case_path):  # noqa: D103
    # Read JSON data
    with open(  # noqa: PTH123
        input_json_path + '/IsolatedBuildingCFD.json', encoding='utf-8'
    ) as json_file:
        iso_json_data = json.load(json_file)

    with open(compt_json_path, encoding='utf-8') as json_file:
        compt_json_data = json.load(json_file)

    geom_data = iso_json_data['GeometricData']
    scale = 1.0/geom_data['geometricScale']
    thickness = 0.01*scale*geom_data["buildingWidth"]

    compts = compt_json_data['components']

    #Write stl files for each component
    for compt_i in compts:
        geoms = compt_i["geometries"]

        for geom_i in geoms:
            if geom_i["shape"] == "circle":
                make_circular_component_geometry(thickness, scale, compt_i["componentId"], geom_i, case_path)
            elif geom_i["shape"] == "polygon":
                make_polygon_component_geometry(thickness, scale, compt_i["componentId"], geom_i, case_path)


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


def make_circular_component_geometry(thickness, geom_scale, compt_id, geom_json, case_path): 

    geom_id = geom_json["geometryId"]
    radius = geom_json["radius"]
    origin = np.array(geom_json["origin"])
    normal = np.array(geom_json["normal"])
    global_origin = np.array([0.0, 0.0, 0.0])

    num_points = 100  # Number of points approximating the circle
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_points = np.array([[radius * np.cos(angle), radius * np.sin(angle), 0] for angle in angles])

    # Rotate the points to align with the normal vector
    circle_points = rotate_to_normal(circle_points, normal) + origin

    # Scale the points
    circle_points = scale_vertices(circle_points, global_origin, geom_scale) 


    # Offset top and bottom circles to create thickness
    top_circle = circle_points + normal * (thickness / 2)
    bottom_circle = circle_points - normal * (thickness / 2)

    # Create faces for the sides of the disk (sidewalls)
    faces = []
    for i in range(num_points):
        next_i = (i + 1) % num_points
        # Two triangles for each side face
        faces.append([top_circle[i], bottom_circle[i], bottom_circle[next_i]])
        faces.append([top_circle[i], bottom_circle[next_i], top_circle[next_i]])

    # Create top and bottom faces (triangulated fan)
    for i in range(1, num_points - 1):
        # Top face
        faces.append([top_circle[0], top_circle[i], top_circle[i + 1]])
        # Bottom face (reversed order)
        faces.append([bottom_circle[0], bottom_circle[i + 1], bottom_circle[i]])

    # Convert faces to a numpy array
    faces = np.array(faces)

    # Create the STL mesh
    disk_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        disk_mesh.vectors[i] = f

    # Save the STL file
    stl_file_name = case_path + '/constant/geometry/components/component_{}_geometry_{}.stl'.format(compt_id, geom_id)
    
    fmt = mesh.stl.Mode.ASCII  # ASCII format
    print(stl_file_name)
    disk_mesh.save(stl_file_name, mode=fmt)


def make_polygon_component_geometry(thickness, geom_scale, compt_id, geom_json, case_path): 

    geom_id = geom_json["geometryId"]
    points = np.array(geom_json["points"])
    global_origin = np.array([0.0, 0.0, 0.0])

    normal = calculate_normal(points)

    # Scale the vertices
    points = scale_vertices(points, np.array(global_origin), geom_scale)

    # Create the top and bottom polygons by shifting along the Z-axis (thickness)
    top_polygon = points + normal * (thickness / 2)
    bottom_polygon = points - normal * (thickness / 2)

    # Create faces for the sides of the disk (sidewalls)
    faces = []
    num_points = len(points)
    
    # Side faces (connecting corresponding vertices of top and bottom polygons)
    for i in range(num_points):
        next_i = (i + 1) % num_points
        # Two triangles for each side face
        faces.append([top_polygon[i], bottom_polygon[i], bottom_polygon[next_i]])
        faces.append([top_polygon[i], bottom_polygon[next_i], top_polygon[next_i]])

    # Create top and bottom faces (from perimeter vertices)
    for i in range(num_points - 2):
        # Top face: First vertex to each subsequent pair of vertices
        faces.append([top_polygon[0], top_polygon[i + 1], top_polygon[i + 2]])
        # Bottom face: First vertex to each subsequent pair of vertices (inverted order)
        faces.append([bottom_polygon[0], bottom_polygon[i + 2], bottom_polygon[i + 1]])

    # Convert faces to a numpy array
    faces = np.array(faces)

    # Create a mesh object for the STL file
    polygon_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    # Assign faces to the mesh
    for i, f in enumerate(faces):
        polygon_mesh.vectors[i] = f


    # Save the STL file
    stl_file_name = case_path + '/constant/geometry/components/component_{}_geometry_{}.stl'.format(compt_id, geom_id)
    
    fmt = mesh.stl.Mode.ASCII  # ASCII format
    print(stl_file_name)
    polygon_mesh.save(stl_file_name, mode=fmt)


def make_rectangular_component_geometry(thickness, geom_scale, compt_id, geom_json, case_path): 

    geom_id = geom_json["geometryId"]
    points = np.array(geom_json["points"])
    global_origin = np.array([0.0, 0.0, 0.0])

    normal = calculate_normal(points)

    # Scale the vertices
    points = scale_vertices(points, np.array(global_origin), geom_scale)

    # Create the top and bottom polygons by shifting along the Z-axis (thickness)
    top_polygon = points + normal * (thickness / 2)
    bottom_polygon = points - normal * (thickness / 2)

    # Create faces for the sides of the disk (sidewalls)
    faces = []
    num_points = len(points)
    
    # Side faces (connecting corresponding vertices of top and bottom polygons)
    for i in range(num_points):
        next_i = (i + 1) % num_points
        # Two triangles for each side face
        faces.append([top_polygon[i], bottom_polygon[i], bottom_polygon[next_i]])
        faces.append([top_polygon[i], bottom_polygon[next_i], top_polygon[next_i]])

    # Create top and bottom faces (from perimeter vertices)
    for i in range(num_points - 2):
        # Top face: First vertex to each subsequent pair of vertices
        faces.append([top_polygon[0], top_polygon[i + 1], top_polygon[i + 2]])
        # Bottom face: First vertex to each subsequent pair of vertices (inverted order)
        faces.append([bottom_polygon[0], bottom_polygon[i + 2], bottom_polygon[i + 1]])

    # Convert faces to a numpy array
    faces = np.array(faces)

    # Create a mesh object for the STL file
    polygon_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    # Assign faces to the mesh
    for i, f in enumerate(faces):
        polygon_mesh.vectors[i] = f


    # Save the STL file
    stl_file_name = case_path + '/constant/geometry/components/component_{}_geometry_{}.stl'.format(compt_id, geom_id)
    
    fmt = mesh.stl.Mode.ASCII  # ASCII format
    print(stl_file_name)
    polygon_mesh.save(stl_file_name, mode=fmt)


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



# Define a function to normalize a vector
def normalize(vector):
    return vector / np.linalg.norm(vector)


def transform_vertices(vertices, origin, normal):
    """
    Rotates the vertices to align with the given normal vector and translates them to the origin.

    :param vertices: Array of vertex coordinates in the XY plane.
    :param origin: Target origin for the transformed vertices.
    :param normal: Target normal vector.
    :return: Transformed vertices.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1])

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
    # Apply the rotation and translate to the origin
    return np.dot(vertices, rotation_matrix.T) + origin


if __name__ == '__main__':  
    
    input_args = sys.argv

    # Set filenames
    input_json_path = sys.argv[1]
    compt_json_path = sys.argv[2]
    case_path = sys.argv[3]

    # Write the component's geometry
    create_component_geometry(input_json_path, compt_json_path, case_path)

