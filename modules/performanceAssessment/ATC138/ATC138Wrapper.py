#  # noqa: INP001, D100
# Copyright (c) 2026 The Regents of the University of California
# Copyright (c) 2026 Leland Stanford Junior University
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this software. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnoczay

"""SimCenter wrapper for the ATC-138 Functional Recovery Assessment.

This script bridges the SimCenter simulation workflow (sWHALE) and the ``atc138``
Python package.  It reads configuration from the AIM file, stages the
required input files from the Pelicun damage-and-loss results into an
``ATC138_input/`` directory, and invokes ``atc138.driver.run_analysis``.

The wrapper is executed by sWHALE from the working directory that already
contains the Pelicun outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

from atc138.driver import run_analysis

# Pelicun output files required by the ATC-138 Pelicun converter.
REQUIRED_PELICUN_FILES = (
    'CMP_QNT.csv',
    'DL_summary.csv',
    'DMG_sample.csv',
    'DV_repair_sample.csv',
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    """Read and return a JSON file as a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path) as fh:
        return json.load(fh)


def _extract_app_data(aim_data: dict) -> dict:
    """Extract the ATC-138 ApplicationData from the AIM structure.

    Args:
        aim_data: Parsed AIM JSON content.

    Returns:
        The ``ApplicationData`` dictionary for the Performance application,
        or an empty dict if the key path is absent.
    """
    return (
        aim_data
        .get('Applications', {})
        .get('Performance', {})
        .get('ApplicationData', {})
    )


def _extract_csv_from_zip(zip_path: Path, dest_path: Path) -> None:
    """Extract the first CSV file found inside a ZIP archive.

    Args:
        zip_path: Path to the ZIP archive.
        dest_path: Destination file path for the extracted CSV.

    Raises:
        FileNotFoundError: If the ZIP contains no CSV files.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
        if not csv_names:
            msg = f'No CSV file found inside {zip_path.name}'
            raise FileNotFoundError(msg)
        with zf.open(csv_names[0]) as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())


def _stage_pelicun_file(
    filename: str, work_dir: Path, input_dir: Path
) -> None:
    """Locate a Pelicun output (ZIP or CSV) and copy it to the input dir.

    Checks for a ``.zip`` archive first (the default Pelicun output format),
    then falls back to a plain ``.csv``.

    Args:
        filename: Base filename (e.g. ``'DMG_sample.csv'``, ``'AIM.json'``).
        work_dir: Working directory containing Pelicun outputs.
        input_dir: ATC-138 input directory to copy the file into.

    Raises:
        FileNotFoundError: If neither the ZIP nor the CSV is found.
    """
    stem = Path(filename).stem  # e.g. 'DMG_sample' from 'DMG_sample.csv'
    zip_path = work_dir / (stem + '.zip')
    csv_path = work_dir / filename

    if zip_path.exists():
        _extract_csv_from_zip(zip_path, input_dir / filename)
    elif csv_path.exists():
        shutil.copy2(csv_path, input_dir / filename)
    else:
        msg = (
            f'Required Pelicun output {filename} not found in {work_dir} '
            f'(checked both .csv and .zip)'
        )
        raise FileNotFoundError(msg)


def _copy_if_exists(src: str, dest: Path) -> bool:
    """Copy a file to *dest* if the source path is non-empty and exists.

    Args:
        src: Source file path as a string (may be empty).
        dest: Destination path.

    Returns:
        True if the file was copied, False otherwise.
    """
    if src and Path(src).exists():
        shutil.copy2(src, dest)
        return True
    return False


def generate_summary(output_dir: Path) -> None:
    """Post-process recovery_outputs.json into a compact summary.json.

    Extracts building-level recovery days for the three recovery phases
    (Reoccupancy, Functional Recovery, Full Recovery) and computes
    summary statistics (mean, std, min, percentiles, max) for each.

    Args:
        output_dir: Path to the ``ATC138_output/`` directory that
            contains ``recovery_outputs.json``.
    """
    import numpy as np

    data = _read_json(output_dir / 'recovery_outputs.json')

    recovery = data['recovery']
    reoc = np.array(recovery['reoccupancy']['building_level']['recovery_day'])
    func = np.array(recovery['functional']['building_level']['recovery_day'])
    full = np.array(
        data['building_repair_schedule']['full']['repair_complete_day']['per_story']
    ).max(axis=-1)

    def _stats(a: np.ndarray) -> dict:
        return {
            'mean': round(float(a.mean()), 2),
            'std': round(float(a.std()), 2),
            'min': round(float(a.min()), 2),
            '0.10%': round(float(np.percentile(a, 10)), 2),
            '50%': round(float(np.percentile(a, 50)), 2),
            '90%': round(float(np.percentile(a, 90)), 2),
            'max': round(float(a.max()), 2),
        }

    summary = {
        'Reoccupancy [days]': _stats(reoc),
        'Functional Recovery [days]': _stats(func),
        'Full Recovery [days]': _stats(full),
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f'ATC-138 | Summary written to {summary_path}')


def generate_default_tenant_unit_list(
    general_inputs: dict, num_stories: int, output_path: Path
) -> None:
    """Create a tenant_unit_list.csv with one tenant unit per story.

    Each unit is assigned the full story plan area, a perimeter area
    derived from edge lengths and story height, and a default occupancy
    type (``occupancy_id=1``).

    Args:
        general_inputs: Parsed ``general_inputs.json`` content.
        num_stories: Number of above-ground stories.
        output_path: Destination CSV path.
    """
    length_1 = general_inputs.get('length_side_1_ft', 100.0)
    length_2 = general_inputs.get('length_side_2_ft', 100.0)
    story_ht = general_inputs.get('typ_story_ht_ft', 13.0)

    story_area = length_1 * length_2
    perim_area = 2 * (length_1 + length_2) * story_ht

    with open(output_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['id', 'story', 'area', 'perim_area', 'occupancy_id'])
        for story in range(1, num_stories + 1):
            writer.writerow([story, story, story_area, perim_area, 1])

    print(f'Generated default tenant_unit_list.csv ({num_stories} tenant units)')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
        general_inputs_path: Path,
        tenant_unit_list_path: Path = None,
        optional_inputs_path: Path = None
) -> None:
    """Run the ATC-138 functional recovery assessment.

    Reads configuration from the AIM file, stages all required inputs
    into ``ATC138_input/``, and calls ``atc138.driver.run_analysis``.
    The script assumes it is executed from the working directory that
    already contains the Pelicun output files.

    Args:
        general_inputs_path: Path to the ``general_inputs.json`` file.
        tenant_unit_list_path: Path to the ``tenant_unit_list.csv`` file.
        optional_inputs_path: Path to the ``optional_inputs.json`` file.
    """
    print('ATC-138 | Starting Functional Recovery Assessment')

    aim_path = Path('AIM.json')
    aim_data = _read_json(aim_path)

    work_dir = Path.cwd()
    input_dir = work_dir / 'ATC138_input'
    output_dir = work_dir / 'ATC138_output'

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage general_inputs.json -------------------------------------------

    if not general_inputs_path or not Path(general_inputs_path).exists():
        msg = (
            f'general_inputs.json not found at '
            f'{general_inputs_path or "<not specified>"}'
        )
        raise FileNotFoundError(msg)

    shutil.copy2(general_inputs_path, input_dir / 'general_inputs.json')
    general_inputs = _read_json(Path(general_inputs_path))

    # --- Stage Pelicun output files ------------------------------------------

    # copy the CMP_QNT.csv file from the templatedir to the workdir
    shutil.copy2('templatedir/CMP_QNT.csv','CMP_QNT.csv')

    for filename in REQUIRED_PELICUN_FILES:
        _stage_pelicun_file(filename, work_dir, input_dir)

    # --- Stage tenant unit list (auto-generate if not provided) --------------

    if not _copy_if_exists(
        str(tenant_unit_list_path),
        input_dir / 'tenant_unit_list.csv',
    ):
        num_stories = int(general_inputs.get('number_of_stories', 1))
        generate_default_tenant_unit_list(
            general_inputs, num_stories, input_dir / 'tenant_unit_list.csv'
        )

    # --- Stage optional assessment inputs (skip silently if absent) ----------

    _copy_if_exists(
        str(optional_inputs_path),
        input_dir / 'optional_inputs.json',
    )

    # --- Run the analysis ----------------------------------------------------

    print(f'ATC-138 | Input directory:  {input_dir}')
    print(f'ATC-138 | Output directory: {output_dir}')

    run_analysis(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    # --- Post-process results into summary.json ------------------------------

    generate_summary(output_dir)

    print('ATC-138 | Assessment complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ATC-138 Functional Recovery Wrapper for SimCenter',
    )
    parser.add_argument(
        '--generalInputsPath',
        type=Path,
        required=True,
        help='Path to the general_inputs JSON file',
    )
    parser.add_argument(
        '--tenantUnitsListPath',
        type=Path,
        default=None,
        help='Path to the tenant_units CSV file',
    )
    parser.add_argument(
        '--optionalInputsPath',
        type=Path,
        default=None,
        help='Path to the optional_inputs JSON file',
    )

    args = parser.parse_args()

    main(
        general_inputs_path = args.generalInputsPath,
        tenant_unit_list_path = args.tenantUnitsListPath,
        optional_inputs_path = args.optionalInputsPath
    )
