#  # noqa: INP001, D100
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Wael Elhaddad
#

import argparse
import glob
from datetime import datetime

import numpy as np
import pandas as pd


def log_msg(msg):  # noqa: ANN001, ANN201, D103
    print(  # noqa: T201
        '{} {}'.format(datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S:%fZ')[:-4], msg)  # noqa: DTZ003
    )


def main(threads=1):  # noqa: ANN001, ANN201, C901, D103
    headers = dict(  # noqa: C408
        IM=[0, 1, 2, 3],
        BIM=[
            0,
        ],
        EDP=[0, 1, 2, 3],
        DM=[0, 1, 2],
        DV=[0, 1, 2, 3],
    )

    use_dask = threads > 1

    if use_dask:
        log_msg(f'{threads} threads requested. Using DASK.')

        import math  # noqa: PLC0415

        from dask import delayed  # noqa: PLC0415
        from dask.distributed import Client, LocalCluster  # noqa: PLC0415

        @delayed
        def read_csv_files(file_list, header):  # noqa: ANN001, ANN202
            return [pd.read_csv(fn, header=header, index_col=0) for fn in file_list]

        def read_csv_np(file, header):  # noqa: ANN001, ANN202
            res = np.loadtxt(file, delimiter=',', dtype=str)

            first_row = header[-1] + 1
            data = res[first_row:].T[1:].T
            data[data == ''] = np.nan

            tuples = [tuple(h) for h in res[:first_row].T[1:]]
            MI = pd.MultiIndex.from_tuples(tuples, names=res[:first_row].T[0])  # noqa: N806

            df = pd.DataFrame(  # noqa: PD901
                data, columns=MI, index=res[first_row:].T[0], dtype=float
            )

            return df  # noqa: RET504

        @delayed
        def read_csv_files_np(file_list, header):  # noqa: ANN001, ANN202
            return [read_csv_np(fn, header=header) for fn in file_list]

        cluster = LocalCluster()
        client = Client(cluster)

        log_msg('Cluster initialized.')
        log_msg(client)

    for res_type in ['IM', 'BIM', 'EDP', 'DM', 'DV']:
        log_msg(f'Loading {res_type} files...')

        files = glob.glob(f'./results/{res_type}/*/{res_type}_*.csv')  # noqa: PTH207
        # files = files[:1000]

        if len(files) > 0:
            if use_dask:
                file_count = len(files)
                chunk = math.ceil(file_count / threads)
                df_list = []

                print(f'Creating threads for {file_count} files...')  # noqa: T201

                for t_i in range(threads):
                    # print(t_i)

                    if t_i * chunk < file_count - 1:
                        df_list_i = delayed(read_csv_files)(
                            files[t_i * chunk : (t_i + 1) * chunk], headers[res_type]
                        )
                        df_i = delayed(pd.concat)(df_list_i, axis=0, sort=False)

                        df_list.append(df_i)

                    elif t_i * chunk == file_count - 1:
                        df_i = delayed(read_csv_files)(
                            files[t_i * chunk : (t_i + 1) * chunk], headers[res_type]
                        )
                        df_i = df_i[0]

                        df_list.append(df_i)

                df_all = delayed(pd.concat)(df_list, axis=0, sort=False)

                df_all = client.compute(df_all)

                df_all = client.gather(df_all)

            else:
                log_msg('Loading all files')
                df_list = [
                    pd.read_csv(resFileName, header=headers[res_type], index_col=0)
                    for resFileName in files
                ]

                log_msg('Concatenating all files')
                df_all = pd.concat(df_list, axis=0, sort=False)

            df_all.sort_index(axis=0, inplace=True)  # noqa: PD002

            # save the results
            log_msg('Saving results')
            df_all.index = df_all.index.astype(np.int32)
            df_all.to_hdf(
                f'{res_type}.hdf',
                'data',
                mode='w',
                format='fixed',
                complevel=1,
                complib='blosc:blosclz',
            )
            # df_all.to_csv('{}.csv'.format(res_type))

        else:
            print(f'No {res_type} files found')  # noqa: T201

    if use_dask:
        log_msg('Closing cluster...')
        cluster.close()
        client.close()

    # aggregate the realizations files
    log_msg('Aggregating individual realizations...')

    files = glob.glob(  # noqa: PTH207
        './results/{}/*/{}_*.hdf'.format('realizations', 'realizations')
    )

    log_msg(f'Number of files: {len(files)}')

    # get the keys from the first file
    if len(files) > 0:
        first_file = pd.HDFStore(files[0])
        keys = first_file.keys()
        first_file.close()

        for key in keys:
            log_msg(f'Processing realizations for key {key}')
            df_list = [pd.read_hdf(resFileName, key) for resFileName in files]

            log_msg('\t\tConcatenating files')
            df_all = pd.concat(df_list, axis=0, sort=False)

            df_all.index = df_all.index.astype(np.int32)

            df_all.sort_index(axis=0, inplace=True)  # noqa: PD002

            try:
                df_all.astype(np.float32).to_hdf(
                    'realizations.hdf',
                    key,
                    mode='a',
                    format='fixed',
                    complevel=1,
                    complib='blosc:blosclz',
                )
            except:  # noqa: E722
                df_all.to_hdf(
                    'realizations.hdf',
                    key,
                    mode='a',
                    format='fixed',
                    complevel=1,
                    complib='blosc:blosclz',
                )

            log_msg(f'\t\tResults saved for {key}.')

    log_msg('End of script')


if __name__ == '__main__':
    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser('Aggregate the results from rWHALE.')  # noqa: N816

    workflowArgParser.add_argument(
        '-threads',
        '-t',
        type=int,
        default=48,
        help='Number of threads to use to aggregate the files.',
    )

    # Parsing the command line arguments
    line_args = workflowArgParser.parse_args()

    main(line_args.threads)
