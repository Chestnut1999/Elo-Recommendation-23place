import numpy as np
import pandas as pd
import os
import sys
HOME = os.path.expanduser("~")
sys.path.append(f'{HOME}/kaggle/data_analysis/library')
import utils



# ========================================================================
# First Month別にFitさせるにあたり、データセットの分布を各First Monthに揃える
# ========================================================================

def make_fam_dist(base_fam, limit_diff_num, is_drop=False):
    #  base_fam = '2017-12'
    #  limit_diff_num = 5

    # ========================================================================
    # Args
    key = 'card_id'
    target = 'target'
    is_viz = False
    base_year = int(base_fam[:4])
    base_month = int(base_fam[-2:])
    max_fam = '2017-12'
    min_fam = '2011-11'
    result_id = []
    # ========================================================================

    # ========================================================================
    # Data Load
    base = utils.read_df_pkl('../input/base_first*')
    base[target] = base[target].map(lambda x: np.round(x, 1))
    # ========================================================================
    df_list = []

    val_cnt = base[target].value_counts()
    val_cnt.name = 'all'
    df_list.append(val_cnt.copy())

    base_1712 = base[base['first_active_month'] == base_fam]
    val_cnt = base_1712[target].value_counts()

    # もう使わないかも?
    is_max = False
    if is_max:
        val_cnt_max = val_cnt.max()
        val_cnt /= val_cnt_max
    val_cnt.name = base_fam
    df_list.append(val_cnt)
    df = pd.concat(df_list, axis=1)


    def arange_ratio(df, multi, is_viz=False):
        df[base_fam] *= multi
        df['diff'] = df['all'] - df[base_fam]
        diff_len = len(df[df['diff'] < 0])
        if is_viz:
            display(df[df['diff'] < 0])
        if diff_len > limit_diff_num:
            return -1
        return 0


    df = pd.concat(df_list, axis=1)

    # もう使わないかも?
    target_max = np.max(df.dropna().index.tolist())
    target_min = np.min(df.dropna().index.tolist())

    cnt_0_fam = df.loc[0.0, :][base_fam]
    cnt_0_all = df.loc[0.0, :]['all']
    multi = int(cnt_0_all / cnt_0_fam)+1

    while True:
        tmp = df.copy().dropna()
        is_minus = arange_ratio(tmp, multi)
        if is_minus:
            multi -= 1
            continue
        break

    print(f"multi: {multi}")
    df[base_fam] *= multi
    if is_drop:
        df_loy = df.dropna()
        loy_list = list(df_loy.index)
    else:
        loy_list = list(np.arange(target_min, target_max, 0.1))

    # ========================================================================
    # Sampling
    # ========================================================================
    before = 0
    for i in loy_list:
        loy = np.round(i, 1)
        df_id = base[base[target] == loy]
        if len(df_id) == 0:
            continue
        sample = df.loc[loy, base_fam]
        if sample == sample:
            before = sample
        else:
            sample = before
        sample = np.int(sample)
        remain = sample
        sampling_id = []

        if remain==0:
            continue

        if is_viz:
            print('''
    #========================================================================
    # Sampling Start!!
    ''')

        for i in range(100):

            is_add = True
            if i == 0:
                yyyymm = base_fam
                tmp_id = df_id[df_id['first_active_month'] == yyyymm]
            else:
                year = base_year
                month = base_month + i

                if month > 12:
                    num_year = month//12
                    year = year + num_year
                    month = month - 12 * num_year
                elif month < 1:
                    num_year = month//12
                    if num_year == 0:
                        year = year - 1
                        month = month + 12
                    else:
                        num_year *= -1
                        year = year - num_year
                        month = month + 12*num_year
                if month < 10:
                    month = f'0{month}'

                yyyymm = f"{year}-{month}"

                if yyyymm < min_fam or yyyymm > max_fam:
                    is_add = False
                else:
                    tmp_id = df_id[df_id['first_active_month'] == yyyymm]
                    if i > 0 and yyyymm == base_fam:
                        is_add = False
    
            # ========================================================================
            # Sampling
            if is_add:
                if is_viz:
                    print(f'future yyyymm: {yyyymm}')
                id_list = list(tmp_id[key].values)
                if len(id_list) <= remain:
                    sampling_id += id_list
                else:
                    sampling_id += list(np.random.choice(id_list,
                                                         remain, replace=False))
    
                if is_viz:
                    print(f"sampling_id: {len(sampling_id)} / {sample}")
            # ========================================================================
    
            remain = sample - len(sampling_id)
            if remain <= 0:
                break
    
            is_add = True
            if i > 0:
                year = base_year
                month = base_month - i
    
                if month > 12:
                    num_year = month//12
                    year = year + num_year
                    month = month - 12 * num_year
                elif month < 1:
                    num_year = month//12
                    if num_year == 0:
                        year = year - 1
                        month = month + 12
                    else:
                        num_year *= -1
                        year = year - num_year
                        month = month + 12*num_year
                if month < 10:
                    month = f'0{month}'
    
                yyyymm = f"{year}-{month}"
    
                if yyyymm < min_fam or yyyymm > max_fam:
                    is_add = False
                else:
                    tmp_id = df_id[df_id['first_active_month'] == yyyymm]
    
                # ========================================================================
                # Sampling
                if is_add:
                    if is_viz:
                        print(f'past yyyymm: {yyyymm}')
                    id_list = list(tmp_id[key].values)
                    if len(id_list) <= remain:
                        sampling_id += id_list
                    else:
                        sampling_id += list(np.random.choice(id_list,
                                                             remain, replace=False))
    
                    if is_viz:
                        print(f"sampling_id: {len(sampling_id)} / {sample}")
                # ========================================================================
    
                remain = sample - len(sampling_id)
                if remain <= 0:
                    break

        result_id += sampling_id
        if is_viz:
            print(f"loy:{loy} | {len(sampling_id)}/{sample} | All: {len(result_id)}")
            print('''
    # Sampling Complete!!
    #========================================================================
    ''')
    print(f"All: {len(result_id)} | Unique: {len(np.unique(result_id))}")
    print(base[base[key].isin(result_id)]
            ['first_active_month'].value_counts().head())
    print(base[base[key].isin(result_id)]['target'].value_counts().head())

    return result_id
