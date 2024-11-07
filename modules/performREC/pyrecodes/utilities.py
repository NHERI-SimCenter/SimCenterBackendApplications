import json
import importlib
import shapely

def read_json_file(file_name: str) -> dict:
    """Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_name (str): The name of the JSON file to be read.

    Returns:
        dict: The dictionary representation of the JSON data.
    """
    with open(file_name, 'r') as file:
        file = json.load(file)
    return file

def get_class(module_name: str, class_name: str, folder_name: str) -> object:
    """Imports a class from a file"""
    class_module = importlib.import_module(f'pyrecodes.{folder_name}.{module_name}')
    target_class = getattr(class_module, class_name)
    return target_class

def create_locality_polygon(bounding_box: list) -> shapely.Polygon:
    return shapely.Polygon([(lat, long) for long, lat in bounding_box])

def component_inside_bounding_box(component_location: list, polygon: shapely.Polygon) -> bool:
    component_centroid = shapely.Point(component_location[0], component_location[1])
    return component_centroid.within(polygon)

def format_locality_id(locality_string) -> int:
    return int(locality_string.split(' ')[-1])