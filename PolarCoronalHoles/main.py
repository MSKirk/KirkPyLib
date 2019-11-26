from PolarCoronalHoles import PCH_stats
import os
import pandas as pd
from multiprocessing import Pool
from functools import partial
import time


def prep_params(wav_filter_list):
    params_dict_list = []
    for w in wav_filter_list:
        for northern in [True, False]:
            params_dict_list.append({'wav_filter':w, 'northern':northern})
    return params_dict_list


def agg_results(wav_filter_list, df_pool):
    df_dict = {}
    for i, w in enumerate(wav_filter_list):
        df_n_s = [df_pool[i * 2], df_pool[i * 2 + 1]]
        df_dict[w] = df_n_s
    return df_dict


if __name__ == '__main__':

    # Read the dataframe and do not put in the local scope of the function parallelized.
    pch_df = pd.read_pickle(os.path.expanduser('/Users/mskirk/data/PCH_Project/pch_obj_concat.pkl'))

    def run_stat(dict_params):
        hem_df = PCH_stats.df_chole_stats_hem(pch_df, binsize=10, sigma=1.0, wave_filter=dict_params['wav_filter'], northern=dict_params['northern'], window_size='33D')
        return hem_df

    wav_list = ['EIT171', 'EIT195', 'EIT304', 'EUVI171', 'EUVI195', 'EUVI304', 'AIA171', 'AIA193', 'AIA304', 'AIA211', 'SWAP174']
    params = prep_params(wav_list)

    tstart = time.time()
    # set number of parallel jobs to run at a time and run the pool of workers
    nprocesses = 22
    with Pool(nprocesses) as p:
        df_pool = p.map(run_stat, params)

    dfs_dict = agg_results(wav_list, df_pool)

    elapsed_time = time.time() - tstart
    print('Compute time: {:1.0f} sec ({:1.1f} min)'.format(elapsed_time, elapsed_time / 60))
