"""This script create blockMeshDict for OpenFoam given vertices and boundary type.

code creates pressure probes for the main simulation. Three types of
probes are created.

"""  # noqa: CPY001, D404, INP001

import json
import sys

import CWE as cwe  # noqa: N811
import numpy as np


def write_wind_profiles(case_path):  # noqa: ANN001, ANN201, D103
    inf_path = (
        case_path + '/constant/boundaryData/windProfile/sampledData/verticalProfile/'
    )

    inf = cwe.VelocityData('cfd', inf_path, start_time=None, end_time=None)

    # Read JSON data for turbulence model
    wc_json_file = open(case_path + '/constant/simCenter/windCharacteristics.json')  # noqa: PLW1514, PTH123, SIM115

    # Returns JSON object as a dictionary
    wind_data = json.load(wc_json_file, 'r', encoding='utf-8')
    wc_json_file.close()

    building_height = wind_data['buildingHeight']

    # Wind profile z, Uav, Iu, Lu
    prof = np.zeros((len(inf.z), 4))
    prof[:, 0] = inf.z
    prof[:, 1] = inf.Uav
    prof[:, 2] = inf.I[:, 0]
    prof[:, 3] = inf.L[:, 0]

    # Wind velocity at roof height
    H_loc = np.argmin(np.abs(inf.z - building_height))  # noqa: N806

    # U, v, w in at roof height
    Uh = inf.U[H_loc, :, :].T  # noqa: N806

    s_uh = []

    for i in range(3):
        f, s = cwe.psd(Uh[:, i], 0.0025, 8)
        s_uh.append(np.abs(s))

    s_uh.insert(0, f)

    Suhout = np.asarray(s_uh, dtype=np.float32).T  # noqa: N806

    write_precision = 6
    fmt = f'%.{write_precision}e'

    prof_path = case_path + '/constant/simCenter/output/windProfiles.txt'
    s_uh_path = case_path + '/constant/simCenter/output/Suh.txt'

    np.savetxt(prof_path, prof, fmt=fmt)
    np.savetxt(s_uh_path, Suhout, fmt=fmt)


def write_wind_loads(case_path):  # noqa: ANN001, ANN201, D103
    # Write base forces
    base_forces_path = case_path + '/postProcessing/baseForces/0/forces.dat'
    base_o, base_t, base_f, base_m = cwe.read_forces_OF10(base_forces_path)  # noqa: F841

    base_forces = np.zeros((len(base_t), 3))

    base_forces[:, 0:2] = base_f[:, 0:2]
    base_forces[:, 2] = base_m[:, 2]

    # Write story forces
    story_forces_path = case_path + '/postProcessing/storyForces/0/forces_bins.dat'
    story_coord, story_t, story_f, story_m = cwe.read_bin_forces_OF10(  # noqa: F841
        story_forces_path
    )

    write_precision = 6
    fmt = f'%.{write_precision}e'

    out_base_path = case_path + '/constant/simCenter/output/baseForces.txt'

    out_story_path_Fx = case_path + '/constant/simCenter/output/storyForcesFx.txt'  # noqa: N806
    out_story_path_Fy = case_path + '/constant/simCenter/output/storyForcesFy.txt'  # noqa: N806
    out_story_path_Mz = case_path + '/constant/simCenter/output/storyForcesMz.txt'  # noqa: N806

    np.savetxt(out_base_path, base_forces, fmt=fmt)

    np.savetxt(out_story_path_Fx, story_f[:, :, 0], fmt=fmt)
    np.savetxt(out_story_path_Fy, story_f[:, :, 1], fmt=fmt)
    np.savetxt(out_story_path_Mz, story_m[:, :, 2], fmt=fmt)


if __name__ == '__main__':
    input_args = sys.argv

    # Set filenames
    case_path = sys.argv[1]

    write_wind_profiles(case_path)
    write_wind_loads(case_path)
