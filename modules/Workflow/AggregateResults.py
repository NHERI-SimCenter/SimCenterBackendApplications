import glob
import numpy as np
import pandas as pd
import argparse

from datetime import datetime
from time import strftime

def log_msg(msg):

    print('{} {}'.format(datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S:%fZ')[:-4], msg))


def main(threads = 1):
    headers = dict(
        EDP = [0, 1, 2, 3],
        DM = [0, 1, 2],
        DV = [0, 1, 2, 3])

    use_dask = threads > 1

    if use_dask:

        log_msg('{} threads requested. Using DASK.'.format(threads))

        from dask.distributed import Client, LocalCluster
        from dask import delayed
        import math

        @delayed
        def read_csv_files(file_list, header):
            return [pd.read_csv(fn, header=header, index_col=0) for fn in file_list]

        def read_csv_np(file, header):
            res = np.loadtxt(file, delimiter=',', dtype=str)

            first_row = header[-1]+1
            data = res[first_row:].T[1:].T
            data[data == ''] = np.nan

            tuples = [tuple(h) for h in res[:first_row].T[1:]]
            MI = pd.MultiIndex.from_tuples(tuples, names=res[:first_row].T[0])

            df = pd.DataFrame(data, columns=MI, index=res[first_row:].T[0], dtype=float)

            return df

        @delayed
        def read_csv_files_np(file_list, header):
            return [read_csv_np(fn, header=header) for fn in file_list]


        cluster = LocalCluster()
        client = Client(cluster)

        log_msg('Cluster initialized.')
        log_msg(client)

    for res_type in ['EDP', 'DM', 'DV']:
    #for res_type in ['EDP', 'DV']:

        log_msg('Loading {} files...'.format(res_type))

        files = glob.glob('./results/{}/*/{}_*.csv'.format(res_type, res_type))
        #files = files[:1000]

        if use_dask:

            file_count = len(files)
            chunk = math.ceil(file_count/threads)
            df_list = []

            for t_i in range(threads):
                
                if t_i*chunk < file_count-1:
                    df_list_i = delayed(read_csv_files)(files[t_i*chunk:(t_i+1)*chunk], headers[res_type])
                    df_i = delayed(pd.concat)(df_list_i, axis=0, sort=False)

                elif t_i*chunk == file_count-1:
                    df_i = delayed(read_csv_files)(files[t_i*chunk:(t_i+1)*chunk], headers[res_type])
                    df_i = df_i[0]
                
                df_list.append(df_i)

            df_all = delayed(pd.concat)(df_list, axis=0, sort=False)

            df_all = client.compute(df_all)

            df_all = client.gather(df_all)

        else:
            log_msg('Loading all files')
            df_list = [pd.read_csv(resFileName, header=headers[res_type], index_col=0) for resFileName in files]

            log_msg('Concatenating all files')
            df_all = pd.concat(df_list, axis=0, sort=False)

        df_all.sort_index(axis=0, inplace=True)

        # save the results
        log_msg('Saving results')
        df_all.to_hdf('{}.hd5'.format(res_type), 'data', mode='w', format='fixed', complevel=1, complib='blosc:snappy')
        #df_all.to_csv('{}.csv'.format(res_type))

    if use_dask:

        log_msg('Closing cluster...')
        cluster.close()
        client.close()

    # aggregate the realizations files
    log_msg('Aggregating individual realizations...')

    files = glob.glob('./results/{}/*/{}_*.hd5'.format('realizations','realizations'))

    # get the keys from the first file
    if len(files) > 0:
        keys = pd.HDFStore(files[0]).keys()

        for key in keys:
            log_msg('Processing realizations for key {key}'.format(key=key))
            df_list = [pd.read_hdf(resFileName, key) for resFileName in files]

            log_msg('\t\tConcatenating files')
            df_all = pd.concat(df_list, axis=0, sort=False)

            df_all.sort_index(axis=0, inplace=True)

            df_all.to_hdf('realizations.hd5', key, mode='a', format='fixed', complevel=1, complib='blosc:snappy')

            log_msg('\t\tResults saved for {key}.'.format(key=key))

    log_msg('End of script')
        

if __name__ == "__main__":

    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Aggregate the reuslts from rWHALE.")

    workflowArgParser.add_argument("-threads", "-t", 
        type=int, default=48, 
        help="Number of threads to use to aggregate the files.")

    #Parsing the command line arguments
    line_args = workflowArgParser.parse_args() 


    main(line_args.threads)