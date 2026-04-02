import numpy as np
from tqdm import tqdm
import geopandas as gpd
import shapely
import sys
import ujson as json

def get_rups_to_run(scenario_info, num_scenarios):  # noqa: C901, D103
    # If there is a filter
    if scenario_info['Generator'].get('method', None) == 'MonteCarlo':
        rup_filter = scenario_info['Generator'].get('RuptureFilter', None)
        if rup_filter is None or len(rup_filter) == 0:
            rups_to_run = list(range(num_scenarios))
        else:
            rups_requested = []
            for rups in rup_filter.split(','):
                if '-' in rups:
                    asset_low, asset_high = rups.split('-')
                    rups_requested += list(
                        range(int(asset_low), int(asset_high) + 1)
                    )
                else:
                    rups_requested.append(int(rups))
            rups_requested = np.array(rups_requested)
            rups_requested = (
                rups_requested - 1
            )  # The input index starts from 1, not 0
            rups_available = list(range(num_scenarios))
            rups_to_run = rups_requested[
                np.where(np.isin(rups_requested, rups_available))[0]
            ]
    else:
        sys.exit(
            f'The scenario selection method {scenario_info["Generator"].get("method", None)} is not available'
        )
    return rups_to_run

def load_earthquake_rupFile(scenario_info, rupFilePath):  # noqa: N802, N803, D103
    # Getting earthquake rupture forecast data
    source_type = scenario_info['EqRupture']['Type']
    try:
        with open(rupFilePath) as f:  # noqa: PTH123
            user_scenarios = json.load(f)
    except:  # noqa: E722
        sys.exit(f'CreateScenario: source file {rupFilePath} not found.')
    # number of features (i.e., ruptures)
    num_scenarios = len(user_scenarios.get('features', []))
    if num_scenarios < 1:
        sys.exit('CreateScenario: source file is empty.')
    rups_to_run = get_rups_to_run(scenario_info, num_scenarios)
    # get rupture and source ids
    scenario_data = {}
    if source_type == 'ERF':
        # source model
        source_model = scenario_info['EqRupture']['Model']
        for rup_tag in rups_to_run:
            cur_rup = user_scenarios.get('features')[rup_tag]
            cur_id_source = cur_rup.get('properties').get('Source', None)
            cur_id_rupture = cur_rup.get('properties').get('Rupture', None)
            scenario_data.update(
                {
                    rup_tag: {
                        'Type': source_type,
                        'RuptureForecast': source_model,
                        'Name': cur_rup.get('properties').get('Name', ''),
                        'Magnitude': cur_rup.get('properties').get(
                            'Magnitude', None
                        ),
                        'MeanAnnualRate': cur_rup.get('properties').get(
                            'MeanAnnualRate', None
                        ),
                        'SourceIndex': cur_id_source,
                        'RuptureIndex': cur_id_rupture,
                        'SiteSourceDistance': cur_rup.get('properties').get(
                            'Distance', None
                        ),
                        'SiteRuptureDistance': cur_rup.get('properties').get(
                            'DistanceRup', None
                        ),
                    }
                }
            )
    elif source_type == 'PointSource':
        sourceID = 0  # noqa: N806
        rupID = 0  # noqa: N806
        for rup_tag in rups_to_run:
            try:
                cur_rup = user_scenarios.get('features')[rup_tag]
                magnitude = cur_rup.get('properties')['Magnitude']
                location = cur_rup.get('properties')['Location']
                average_rake = cur_rup.get('properties')['AverageRake']
                average_dip = cur_rup.get('properties')['AverageDip']
                scenario_data.update(
                    {
                        0: {
                            'Type': source_type,
                            'Magnitude': magnitude,
                            'Location': location,
                            'AverageRake': average_rake,
                            'AverageDip': average_dip,
                            'SourceIndex': sourceID,
                            'RuptureIndex': rupID,
                        }
                    }
                )
                rupID = rupID + 1  # noqa: N806
            except:  # noqa: PERF203, E722
                print('Please check point-source inputs.')  # noqa: T201
    # return
    return scenario_data

def load_earthquake_rup_scenario(scenario_info, user_scenarios):  # noqa: N802, N803, D103
    # Getting earthquake rupture forecast data
    source_type = scenario_info['EqRupture']['Type']
    # number of features (i.e., ruptures)
    num_scenarios = len(user_scenarios)
    if num_scenarios < 1:
        sys.exit('CreateScenario: source file is empty.')
    rups_to_run = get_rups_to_run(scenario_info, num_scenarios)
    # get rupture and source ids
    scenario_data = {}
    if source_type == 'ERF':
        # source model
        source_model = scenario_info['EqRupture']['Model']
        for rup_tag in rups_to_run:
            cur_rup = user_scenarios.iloc[rup_tag, :]
            cur_id_source = cur_rup['Source']
            cur_id_rupture = cur_rup['Rupture']
            print("DEBUG: cur_id_source, cur_id_rupture", cur_id_source, cur_id_rupture)
            scenario_data.update(
                {
                    rup_tag: {
                        'Type': source_type,
                        'RuptureForecast': source_model,
                        'Name': cur_rup.get('Name', ''),
                        'Magnitude': cur_rup.get(
                            'Magnitude', None
                        ),
                        'MeanAnnualRate': cur_rup.get(
                            'MeanAnnualRate', None
                        ),
                        'SourceIndex': cur_id_source,
                        'RuptureIndex': cur_id_rupture,
                        'SiteSourceDistance': cur_rup.get(
                            'Distance', None
                        ),
                        'SiteRuptureDistance': cur_rup.get(
                            'DistanceRup', None
                        ),
                    }
                }
            )
    return scenario_data