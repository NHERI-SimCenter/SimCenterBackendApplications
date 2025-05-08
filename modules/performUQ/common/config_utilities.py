"""
Utility functions for handling configuration files.

Includes loading, saving, merging settings, and writing default configurations
and schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

    from pydantic import BaseModel


def load_settings_from_config(config_path: str | Path) -> dict:
    """
    Load a configuration file in JSON format.

    Args:
        config_path (str | Path): Path to the configuration file.

    Returns
    -------
        dict: The loaded configuration as a dictionary.

    Raises
    ------
        FileNotFoundError: If the specified file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config file '{config_path}' does not exist."
        raise FileNotFoundError(msg)
    with config_path.open() as f:
        return json.load(f)


def save_used_settings_as_config(
    config: dict,
    output_path: str | Path = 'config_used.json',
    logger: logging.Logger | None = None,
):
    """
    Save the provided configuration dictionary to a JSON file.

    Args:
        config (dict): The configuration dictionary to save.
        output_path (str | Path): The file path where the configuration will be saved.
                                  Defaults to 'config_used.json'.
    """
    with Path(output_path).open('w') as f:
        json.dump(config, f, indent=4, sort_keys=True)
    if logger is not None:
        logger.info(f"Saved used configuration to '{Path(output_path).resolve()}'.")


def merge_settings(defaults: dict, config: dict, cli: dict) -> dict:
    """
    Merge default settings, configuration file settings, and command-line arguments.

    Args:
        defaults (dict): Default settings.
        config (dict): Settings from a configuration file.
        cli (dict): Settings from command-line arguments.

    Returns
    -------
        dict: Merged settings with priority given to CLI, then config, then defaults.
    """
    merged = defaults.copy()
    for key in merged:
        if key in config and config[key] is not None:
            merged[key] = config[key]
        elif key in cli and cli[key] is not None:
            merged[key] = cli[key]
    return merged


def write_default_config_and_schema(
    default_settings: BaseModel,
    config_filename='config_default.json',
    schema_filename='config_schema.json',
    logger: logging.Logger | None = None,
):
    """
    Write the default settings and JSON schema to files.

    Parameters
    ----------
    config_filename : str
        Filename for saving the default settings.
    schema_filename : str
        Filename for saving the JSON schema.

    Returns
    -------
    tuple[str, str]
        Absolute paths to the saved config and schema files.
    """
    config_path = Path(config_filename).resolve()
    schema_path = Path(schema_filename).resolve()

    # Write default config
    config_path.write_text(default_settings.model_dump_json(indent=4))

    # Write schema
    schema = default_settings.model_json_schema()
    schema_path.write_text(json.dumps(schema, indent=4, sort_keys=True))

    if logger is not None:
        logger.info(f'Default configuration saved to: {config_path}')
        logger.info(f'JSON schema saved to:           {schema_path}')

    return str(config_path), str(schema_path)
