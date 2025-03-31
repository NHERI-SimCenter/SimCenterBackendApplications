"""
Python code to generate list of all python modules needed by various python files in subdir

wriiten: chatGPT with ppromots from fmk!
"""

import os
import ast
import argparse
import sys
from collections import defaultdict
from stdlib_list import stdlib_list


def find_python_files(directory):
    """
    Recursively locate all Python files within a directory.

    :param directory: The root directory to begin the search.
    :return: A dictionary mapping subdirectory paths to lists of Python file paths.
    """
    python_files = defaultdict(list)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files[root].append(os.path.join(root, file))
    return python_files

def extract_imports(file_path, stdlib_modules, project_modules):
    """
    Parse a Python file to extract all imported modules, excluding standard library modules
    and those defined within the project directory.

    :param file_path: Path to the Python file.
    :param stdlib_modules: Set of standard library module names.
    :param project_modules: Set of module names defined within the project directory.
    :return: A set of imported module names excluding standard and project-defined modules.
    """
    imported_modules = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read(), filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in stdlib_modules and module_name not in project_modules:
                            imported_modules.add(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in stdlib_modules and module_name not in project_modules:
                            imported_modules.add(module_name)
        except (SyntaxError, UnicodeDecodeError):
            print(f"Skipping file due to parse error: {file_path}")
    return imported_modules

def get_project_modules(directory):
    """
    Identify all module names defined within the project directory.

    :param directory: The root directory of the project.
    :return: A set of module names defined within the project directory.
    """
    project_modules = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                module_name = os.path.splitext(file)[0]
                project_modules.add(module_name)
        for subdir in os.listdir(root):
            subdir_path = os.path.join(root, subdir)
            if os.path.isdir(subdir_path) and any(fname.endswith('.py') for fname in os.listdir(subdir_path)):
                project_modules.add(subdir)
    return project_modules

def main(directory):
    """
    Main function to find all Python files in each subdirectory, extract imports,
    and display the results, excluding standard library modules and those defined
    within the project directory.

    :param directory: The root directory to analyze.
    """
    stdlib_modules = set(stdlib_list())
    project_modules = get_project_modules(directory)
    all_imports = defaultdict(set)
    python_files = find_python_files(directory)
    for subdir, files in python_files.items():
        for file_path in files:
            imports = extract_imports(file_path, stdlib_modules, project_modules)
            all_imports[subdir].update(imports)

    for subdir, modules in all_imports.items():
        print(f"\nSubdirectory: {subdir}")
        if modules:
            for module in sorted(modules):
                print(f"  {module}")
        else:
            print("  No non-standard, external imports found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Python files to list imported modules, excluding standard library and project-defined modules.")
    parser.add_argument('directory', type=str, help="Path to the directory to analyze.")
    args = parser.parse_args()
    main(args.directory)
