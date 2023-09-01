"""
概念板块做不了，因为更新太快，而历史数据跨度太大，没法获取历史概念板块
"""

import os, sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
import akshare as ak
from datetime import datetime
import pandas as pd

start_date = "19000101"
end_date = "20231231"

ind = ak.stock_board_industry_name_em()
ind_list = ind['板块名称'].tolist() # 行业板块名称
ind_code_list = ind['板块代码'].tolist()

con = ak.stock_board_concept_name_em()
con_list = con['板块名称'].tolist() # 概念板块名称
con_code_list = con['板块代码'].tolist()

'''
# 维护一个{行业板块：[成分股]}的list 和 一个{概念板块：[成分股]}的list,以便于查找某只股票属于哪个行业，并在概念板块中引入该特征
ind_dict = {}
for i in ind_list:
    stock_board_industry_cons_em_df = ak.stock_board_industry_cons_em(symbol=i)
    stock_list = stock_board_industry_cons_em_df['代码'].tolist()
    ind_dict[i] = stock_list
print('行业板块字典维护完毕')
print(ind_dict)

con_dict = {}
for i in con_list:
    stock_board_concept_cons_em_df = ak.stock_board_concept_cons_em(symbol=i)
    stock_list = stock_board_concept_cons_em_df['代码'].tolist()
    con_dict[i] = stock_list
print('概念板块字典维护完毕')
print(con_dict)


# 查询行业/概念字典
def get_key(dict, value):
    return [k for k, v in dict.items() if value in v]
'''


# 获取该行业的指数、成分股日频数据
for idx, i in enumerate(ind_list):
    print(i)
    folder_path=f'../../../datasets/raw_data/Stock-Industry-Board/{i}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # 获取行业指数日频数据
    m_df = ak.stock_board_industry_hist_em(symbol=i, start_date=start_date, end_date=end_date, period="日k", adjust="hfq")
    m_df['是否行业指数'] = True
    m_df['行业板块'] = i
    m_df['是否次新股'] = False
    m_df['代码']=ind_code_list[idx]
    m_df.to_csv(f'{folder_path}/{i}_index.csv')
    
    # 获取行业成分股日频数据，并写入行业信息
    stock_board_industry_cons_em_df = ak.stock_board_industry_cons_em(symbol=i)
    stock_list = stock_board_industry_cons_em_df['代码'].tolist()
    for j in stock_list:
        try:
            """
            添加行业、概念、次新股相关特征
            """
            
            df = ak.stock_zh_a_hist(symbol=j, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
            
            df['代码']=j
            
            df['是否行业指数'] = False

            if '行业板块' in df.columns: df['行业板块'] = df['行业板块'].apply(lambda x: x + [i])
            else: df['行业板块'] = df['日期'].apply(lambda x: [i])
            
            if df['日期'].min().strftime("%Y-%m-%d") >= "2022-09-01": df['是否次新股'] = True
            else: df['是否次新股'] = False
            
            #df['概念板块'] = df['日期'].apply(lambda x: get_key(con_dict, j))
            
            """
            添加day_of_week, day_of_month, day_of_quarter, day_of_year作为特征
            """
            
            # 计算各种日期信息
            df['day_of_week'] = df['日期'].apply(lambda x: x.weekday() + 1)  # 一周内的第几天，周一为1，周日为7
            df['day_of_month'] = df['日期'].apply(lambda x: x.day)  # 一月内的第几天

            # 计算一季度内的第几天
            df['quarter_start'] = df['日期'].apply(lambda x: pd.Timestamp(x).to_period("Q").start_time.date())
            df['day_of_quarter'] = (df['日期'] - df['quarter_start']).apply(lambda x: x.days + 1)

            # 计算一年内的第几天
            df['day_of_year'] = df['日期'].apply(lambda x: x.timetuple().tm_yday)

            # 删除临时列（如果需要）
            df.drop(columns=['quarter_start'], inplace=True)

            df.to_csv(f'{folder_path}/{j}.csv')
        except Exception as e:
            print(f"An error occurred: {e}")
            continue


'''
# 获取该概念的指数、成分股日频数据
for idx, i in enumerate(con_list):
    print(i)
    folder_path=f'../../../datasets/raw_data/Stock-Concept-Board/{i}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # 获取行业指数日频数据
    m_df = ak.stock_board_concept_hist_em(symbol=i, start_date=start_date, end_date=end_date, period="daily", adjust="hfq")
    m_df['是否概念指数'] = True
    m_df['概念板块'] = i
    m_df['是否次新股'] = False
    m_df['代码'] = con_code_list[idx]
    m_df.to_csv(f'{folder_path}/{i}_index.csv')
    
    # 获取行业成分股日频数据，并写入行业信息
    concept_df = ak.stock_board_concept_cons_em(symbol=i)
    stock_list = concept_df['代码'].tolist()
    for j in stock_list:
        try:
            """
            添加行业、概念、次新股相关特征
            """
            
            df = ak.stock_zh_a_hist(symbol=j, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
            
            df['代码']=j
            
            df['是否概念指数'] = False

            if '概念板块' in df.columns: df['概念板块'] = df['概念板块'].apply(lambda x: x + [i])
            else: df['概念板块'] = df['日期'].apply(lambda x: [i])
            
            if df['日期'].min().strftime("%Y-%m-%d") >= "2022-09-01": df['是否次新股'] = True
            else: df['是否次新股'] = False
            
            df['行业板块'] = df['日期'].apply(lambda x: get_key(ind_dict, j))
            
            """
            添加day_of_week, day_of_month, day_of_quarter, day_of_year作为特征
            """
            
            # 计算各种日期信息
            df['day_of_week'] = df['日期'].apply(lambda x: x.weekday() + 1)  # 一周内的第几天，周一为1，周日为7
            df['day_of_month'] = df['日期'].apply(lambda x: x.day)  # 一月内的第几天

            # 计算一季度内的第几天
            df['quarter_start'] = df['日期'].apply(lambda x: pd.Timestamp(x).to_period("Q").start_time.date())
            df['day_of_quarter'] = (df['日期'] - df['quarter_start']).apply(lambda x: x.days + 1)

            # 计算一年内的第几天
            df['day_of_year'] = df['日期'].apply(lambda x: x.timetuple().tm_yday)

            # 删除临时列（如果需要）
            df.drop(columns=['quarter_start'], inplace=True)
            
            df.to_csv(f'{folder_path}/{j}.csv')
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
        
'''        
        