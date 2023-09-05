import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd
import akshare

# from generate_adj_mx import generate_adj_pems03
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
#from basicts.data.transform import standard_transform
pd.set_option('mode.chained_assignment', None)

    
def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """
    
    #region 参数列表
    data_file_path = args.data_file_path
    add_day_of_week = args.dow
    add_day_of_month = args.dom
    add_day_of_year = args.doy
    add_day_of_quarter = args.doq
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    train_end_date = args.train_end_date
    valid_end_date = args.valid_end_date
    output_dir = args.output_dir
    threshold_up = args.label_threshold_up
    threshold_down = args.label_threshold_down
    #endregion
    
    #region 获取交易日历、读取原始数据、特征工程、定义全局变量
    trade_date_df = akshare.tool_trade_date_hist_sina()
    trade_date_df = trade_date_df[(trade_date_df['trade_date']>=pd.to_datetime('2000-01-04')) & (trade_date_df['trade_date']<=pd.to_datetime('2023-09-01'))]

    # 读取原始数据
    all_data = pd.read_csv(data_file_path)
    code_list = np.sort(all_data['代码'].unique())
    ind = akshare.stock_board_industry_name_em()
    # 创建一个字典，其中 ind['板块代码'] 是键，而 ind['板块名称'] 是值。
    industry_dict = dict(zip(ind['板块代码'], ind['板块名称']))
    all_data = all_data[['日期', '开盘', '收盘', '最高', '最低', '成交量', '换手率', '行业板块', '代码']]
    
    # 特征工程
    if add_day_of_week or add_day_of_month or add_day_of_year or add_day_of_quarter:
        # 请参照STID进行编码，它不是one-hot，而是只用一列保存，然后min-max归一化，为了方便归一化保存为float。
        all_data['日期'] = pd.to_datetime(all_data['日期'])
        if add_day_of_week:
            all_data['day_of_week'] = all_data['日期'].dt.dayofweek.astype(float)
        if add_day_of_month:# Monday=0, Sunday=6
            all_data['day_of_month'] = all_data['日期'].dt.day.astype(float)
        if add_day_of_year:
            all_data['day_of_year'] = all_data['日期'].dt.dayofyear.astype(float)
        if add_day_of_quarter:
            # 计算季度的开始月份
            all_data['quarter_start_month'] = 3 * (all_data['日期'].dt.quarter - 1) + 1
            # 计算季度的开始日期
            all_data['quarter_start_date'] = pd.to_datetime(all_data['日期'].dt.year.astype(str) + '-' +
                                                    all_data['quarter_start_month'].astype(str) + '-1')
            # 计算季度中的第几天
            all_data['day_of_quarter'] = (all_data['日期'] - all_data['quarter_start_date']).dt.days + 1
            all_data['day_of_quarter'] = all_data['day_of_quarter'].astype(float)
            # 删除临时列
            all_data.drop(['quarter_start_month', 'quarter_start_date'], axis=1, inplace=True)
        all_data['日期'] = all_data['日期'].apply(lambda x: x.strftime('%Y-%m-%d'))
        #all_data.drop(['日期2'], axis=1, inplace=True)
    
    all_data = pd.get_dummies(all_data, columns=['行业板块'], dtype=int)

    # 全局变量
    TRAIN, VALID, TEST, ALL = [], [], [], []
    #endregion
    
    #region 确认列号、列大小和列名
    new_columns_count = len([col for col in all_data.columns if '行业板块_' in col])
    print(f"生成了 {new_columns_count} 个行业板块列")
    print(f"day_of_week_size: {all_data['day_of_week'].min()} ~  {all_data['day_of_week'].max()}")
    print(f"day_of_month_size: {all_data['day_of_month'].min()} ~  {all_data['day_of_month'].max()}")
    print(f"day_of_quarter_size: {all_data['day_of_quarter'].min()} ~  {all_data['day_of_quarter'].max()} ")
    print(f"day_of_year_size: {all_data['day_of_year'].min()} ~  {all_data['day_of_year'].max()} ")
    print(f"所有列：")
    print(all_data.columns)
        #endregion
    
    # 遍历所有行业指数
    for x, code in enumerate(code_list):
        print(f'对第{x}/{len(code_list)}个{industry_dict[code]}行业指数进行数据预处理...')
        df = all_data.loc[all_data['代码'] == code]
        # 部分行业指数是后续加入的，数据量不够，无法正常按照日期划分数据。
        if len(df) < 5000: continue

        #region 异常值处理: Remove NaNs, infs and quantile outliers here. 主要问题是有些滑窗行可能会用到一些label不到的行 所以暂时不能移除
        # 移除 NaN 和 inf 值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['开盘', '收盘', '最高', '最低', '成交量', '换手率'], inplace=True)

        # 定义要进行分位数处理的列
        columns_to_process = ['开盘', '收盘', '最高', '最低', '成交量', '换手率']

        # 基于分位数移除异常值
        low = .01
        high = .99

        for name in columns_to_process:
            low_val = df[name].quantile(low)
            high_val = df[name].quantile(high)
            df = df[(df[name] >= low_val) & (df[name] <= high_val)]

        df = df.reset_index(drop=True)  # Reset index after dropping rows
        #print(f'after 异常值处理: {df.head(1)}')
        #endregion
        
        #region 划分数据集: Split the data into train, VALID, and test. Assuming the result are train_df, val_df, test_df
        train_df = df[df['日期'] <= train_end_date]
        val_df = df[(df['日期'] > train_end_date) & (df['日期'] <= valid_end_date)]
        test_df = df[df['日期'] > valid_end_date]
        #print(f'after 数据集划分: {len(train_df)}, {len(val_df)}, {len(test_df)}')
        #endregion
        
        #region 打标签: Your code to generate label based on F_L and P_L here. Assuming label is in a column named 'label'
        # 定义一个用于打标签的函数
        def label_data(df, history_seq_len, future_seq_len, threshold_up, threshold_down):
            # 计算过去 L 日的平均收盘价
            df['P_L'] = df['收盘'].rolling(window=history_seq_len, min_periods=history_seq_len).mean()
            
            # 计算未来 L 日的平均收盘价
            # 注意：这里没有翻转数据，因为我们对每个独立的数据集（训练、验证、测试）进行操作
            df['F_L'] = df['收盘'].rolling(window=future_seq_len, min_periods=future_seq_len).mean()
            df['F_L'] = df['F_L'].shift(-future_seq_len+1)

            # 计算 (F_L - P_L) / P_L
            df['ratio'] = (df['F_L'] - df['P_L']) / df['P_L']
            
            # 基于 (F_L - P_L) / P_L 计算标签
            df['label'] = np.where(df['ratio'] >= threshold_up, 1, np.where(df['ratio'] <= threshold_down, 0, np.nan))
            
            # 删除多余的计算列
            df.drop(['P_L', 'F_L', 'ratio', '代码'], axis=1, inplace=True)
            
            # 过滤出标签非空的行
            df = df[df['label'].notna()]

            return df

        # 为每个数据集打标签
        train_df = label_data(train_df, history_seq_len, future_seq_len, threshold_up, threshold_down)
        val_df = label_data(val_df, history_seq_len, future_seq_len, threshold_up, threshold_down)
        test_df = label_data(test_df, history_seq_len, future_seq_len, threshold_up, threshold_down)
        #print(f'after 打标签: {test_df.head(1)}')

        #endregion
        
        #region 归一化：Min-max scaling based on train data
        # 列名列表，对这些列进行Min-Max归一化
        scale_columns = ['开盘', '收盘', '最高', '最低', '成交量', '换手率', 'day_of_week', 'day_of_month', 'day_of_year', 'day_of_quarter']

        # 计算训练数据的最小值和最大值
        min_vals = train_df[scale_columns].min()
        max_vals = train_df[scale_columns].max()

        # 使用训练数据的最小值和最大值进行归一化
        train_df[scale_columns] = (train_df[scale_columns] - min_vals) / (max_vals - min_vals)
        val_df[scale_columns] = (val_df[scale_columns] - min_vals) / (max_vals - min_vals)
        test_df[scale_columns] = (test_df[scale_columns] - min_vals) / (max_vals - min_vals)
        #endregion
        
        #region 滑动窗口: Your code to create time-series windows based on trade_date_df. Assuming the results are train_windows, val_windows, test_windows
        def generate_time_series_windows(df, trade_date_df, history_seq_len):
            '''
            不允许出现停牌等情况造成的交易日时序数据缺失，高度保证数据质量
            假设 trade_date_df 是交易日历的 DataFrame，包含一列 'trade_date'
            假设 df 是已经划分好的训练集、验证集或测试集 DataFrame，包含一列 '日期'
            '''
            # 先将字符串转换为 datetime64[ns] 类型
            df['日期'] = pd.to_datetime(df['日期'])
            # 然后将 datetime64[ns] 转换为 datetime.date
            df['日期'] = df['日期'].apply(lambda x: x.date())

            #print(df['日期'], df['日期'].values.tolist(), df['日期'].values.tolist()[0])
            #print(type(trade_date_df['trade_date'].values.tolist()[0]), type(df['日期'].values.tolist()[0]))
            #assert(type(trade_date_df['trade_date'].values.tolist()[0])==type(df['日期'].values.tolist()[0])) # 日期格式不一致
            
            windows = []  # 用于存储时间窗口的列表
            trade_dates = trade_date_df['trade_date'].values.tolist()  # 从交易日历中获取所有交易日
            
            trade_date_set = set(df['日期'])  # 将 DataFrame 中所有存在的日期存储到一个集合中以便快速查找
            
            i = 0  # 初始化索引变量
            # 遍历交易日列表
            valid_date = 0
            while i < len(trade_dates) - history_seq_len:
                cur_date_seq = trade_dates[i:i+history_seq_len]  # 取 history_seq_len 天作为当前的时间窗口
                
                # 检查这 history_seq_len 天是否都存在于 DataFrame 中
                if set(cur_date_seq).issubset(trade_date_set):
                    #print(f"cur_date_seq is a subset of trade_date_set: {cur_date_seq}")  # Debugging line
                    # 提取这些日期对应的数据
                    window_data = df[df['日期'].isin(cur_date_seq)].sort_values(by='日期')
                    
                    # 如果这个时间窗口内的数据行数等于 history_seq_len，则认为这是一个有效窗口
                    if len(window_data) == history_seq_len:
                        # 提取除了日期外的其他特征，并添加到时间窗口列表中
                        window_data = window_data.drop(columns=['日期']).to_numpy()
                        windows.append(window_data)
                        
                        i += 1  # 移动到下一个时间窗口
                        valid_date += 1
                    else:
                        i += 1
                else:
                    #print(f"cur_date_seq is NOT a subset of trade_date_set: {cur_date_seq}")  # Debugging line
                    i += 1
            #valid_percent = valid_date / (len(trade_dates) - history_seq_len)
            #print(f'有效日期占比：{valid_date}/{len(trade_dates) - history_seq_len} = {valid_percent}')
            np_windows = np.array(windows)  # 返回时间窗口数组
            #assert np_windows.ndim == 3, f'{np_windows.shape}, {industry_dict[code]}'
            return np_windows
        train_windows = generate_time_series_windows(train_df, trade_date_df, history_seq_len)
        val_windows = generate_time_series_windows(val_df, trade_date_df, history_seq_len)
        test_windows = generate_time_series_windows(test_df, trade_date_df, history_seq_len)
        #print(f'{len(train_df)}, {len(val_df)}, {len(test_df)} after 滑动窗口: {train_windows.shape[0]}, {val_windows.shape[0]}, {test_windows.shape}')
        #endregion

        #region 存储所有有效窗口数据，当某个数据集为空时，直接跳过该行业。
        if train_windows.ndim != 3 or val_windows.ndim != 3 or test_windows.ndim != 3: continue
        TRAIN.append(train_windows)
        VALID.append(val_windows)
        TEST.append(test_windows)
        ALL.append(train_windows)
        ALL.append(val_windows)
        ALL.append(test_windows)
        #endregion

    #region 保存所有行业指数的窗口数据:Concatenate all the arrays
    TRAIN = np.concatenate(TRAIN, axis=0)
    VALID = np.concatenate(VALID, axis=0)
    TEST = np.concatenate(TEST, axis=0)
    ALL = np.concatenate(ALL, axis=0)
    #endregion

    #region 生成索引： Generate indices
    train_num = len(TRAIN)
    valid_num = len(VALID)
    test_num = len(TEST)

    index_list = np.arange(train_num + valid_num + test_num)
    train_index = index_list[:train_num]
    valid_index = index_list[train_num:train_num + valid_num]
    test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    
    print(len(TRAIN), len(VALID), len(TEST), len(TRAIN)/len(TEST),':',len(VALID)/len(TEST),':',1)
    #endregion
    
    #region 序列化数据
    data = {}
    data["processed_data"] = ALL
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(data, f)
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(index, f)
    #endregion
    
    #region 统计0、1标签
    my_array = data['processed_data'][index['train'],-1:,-1]
    count_zeros = np.count_nonzero(my_array == 0.0)
    count_ones = np.count_nonzero(my_array == 1.0)
    ratio = count_ones/ (count_zeros + count_ones)
    print(f"1/0 Ratio of train dataset: {ratio}/")
    
    my_array = data['processed_data'][index['valid'],-1:,-1]
    count_zeros = np.count_nonzero(my_array == 0.0)
    count_ones = np.count_nonzero(my_array == 1.0)
    ratio = count_ones/ (count_zeros + count_ones)
    print(f"1/0 Ratio of valid dataset: {ratio}/")
    
    my_array = data['processed_data'][index['test'],-1:,-1]
    count_zeros = np.count_nonzero(my_array == 0.0)
    count_ones = np.count_nonzero(my_array == 1.0)
    ratio = count_ones/ (count_zeros + count_ones)
    print(f"1/0 Ratio of test dataset: {ratio}/")
    
    #endregion

if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 5
    FUTURE_SEQ_LEN = 5
    
    # label_threshold
    LABEL_THRESHOLD_UP = 0.005
    LABEL_THRESHOLD_DOWN = -0.005

    #TRAIN_RATIO = 0.6
    #VALID_RATIO = 0.2
    '''
    2000-01-01 ~ '2014-04-02' ~ '2018-12-11' ~ '2023-09-01' = 4.2981523664894965 : 1.1931789420399899 : 1
    2000-01-01 ~ '2013-01-01' ~ '2017-08-01' ~ '2023-09-01' = 60.4:18.5:21.5
    '''
    TRAIN_END_DATE = '2012-12-31'
    VALID_END_DATE = '2017-12-31'
    
    #TARGET_CHANNEL = [0]                   # target channel(s)
    #STEPS_PER_DAY = 288

    DATASET_NAME = "STOCK-IND"
    DATASET_FOLDER_NAME = "阿行业指数"
    DATASET_ALL_FILE_NAME = "all_data"
    
    #TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    DOM = True
    DOQ = True
    DOY = True

    #NORM_EACH_CHANNEL = False

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{1}/{2}.csv".format(DATASET_NAME, DATASET_FOLDER_NAME, DATASET_ALL_FILE_NAME)
    #GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw stock readings.")
    #parser.add_argument("--graph_file_path", type=str,
    #                    default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--label_threshold_up", type=float,
                        default=LABEL_THRESHOLD_UP, help="Label Threshold.")
    parser.add_argument("--label_threshold_down", type=float,
                        default=LABEL_THRESHOLD_DOWN, help="Label Threshold.")
    #parser.add_argument("--steps_per_day", type=int,
    #                    default=STEPS_PER_DAY, help="Sequence Length.")
    #parser.add_argument("--tod", type=bool, default=TOD,
    #                    help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_month.")
    parser.add_argument("--doq", type=bool, default=DOQ,
                        help="Add feature day_of_quarter.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_year.")
    #parser.add_argument("--target_channel", type=list,
    #                    default=TARGET_CHANNEL, help="Selected channels.")
    #parser.add_argument("--train_ratio", type=float,
    #                    default=TRAIN_RATIO, help="Train ratio")
    #parser.add_argument("--valid_ratio", type=float,
    #                    default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--train_end_date", type=str,
                        default=TRAIN_END_DATE, help="Train Split date.")
    parser.add_argument("--valid_end_date", type=str,
                        default=VALID_END_DATE, help="Validate Split date.")
    #parser.add_argument("--norm_each_channel", type=float,
    #                    default=NORM_EACH_CHANNEL, help="Validate ratio.")
    args_metr = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if os.path.exists(args_metr.output_dir):
        reply = str(input(
            f"{args_metr.output_dir} exists. Do you want to overwrite it? (y/n)")).lower().strip()
        if reply[0] != "y":
            sys.exit(0)
    else:
        os.makedirs(args_metr.output_dir)
    generate_data(args_metr)
