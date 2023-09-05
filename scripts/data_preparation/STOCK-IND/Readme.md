### 数据预处理过程
1. 全局初始化TRAIN, VAL, TEST, ALL为4个numpy文件。
2. all_data.csv是我将数据预处理的文件，它包含了不同行业指数的日频交易数据。
3. 我将通过all_data.columns告诉你我的csv列格式：

```python
Index(['日期', '开盘', '收盘', '最高', '最低', '涨跌幅', '涨跌额', '成交量', '成交额', '振幅', '换手率',
'是否行业指数', '行业板块', '是否次新股', '代码'],
dtype='object')
```

1. 通过code_list=np.sort(all_data['代码'].unique())，我们可以获得csv中不同行业指数的代码
2. 通过遍历code_list，我们通过all_data.loc[all_data[‘代码’]==单个行业指数代码]每次读取一个行业指数的日频交易数据df，它是dataframe格式。
3. 对于单个行业指数的日频交易数据df，我们只取[’日期’ ,'开盘', '收盘', '最高', '最低', '成交量',  '换手率', '行业板块']作为特征。基于行业板块的数量len(code_list)，我们对特征'行业板块'进行one-hot编码。得到df_2。
4. 对于单个行业指数的日频交易数据df，假设当前日期为X，划分样本的时间窗口大小为L，则对于时间跨度为(X-L, X]的样本，我们取未来L日收盘价F_L与过去L日收盘价P_L进行比较计算样本的标签label。如果(F_L-P_L)/P_L ≥ 0.55% 记为涨(label=1)，如果(F_L-P_L)/P_L ≤ -0.5% 记为跌(label=0)，从而得到label列的值，记为df_3。
5. 去除df_3数据中的nan值或inf值，去除quantile为top1%和bottom1%的异常值，得到df_4。
6. 对于df_4，我们需要根据日期区间[:train_end], [train_end, valid_end], [valid_end:]对数据划分出训练集train、验证集val、测试集test。
7. 对于单个行业指数的训练集train、验证集val、测试集test，我们需要对特征['开盘', '收盘', '最高', '最低', '成交量',  '换手率']进行min-max归一化。为了避免数据泄露，需要记录下训练集的最小值min和最大值max，用训练集的min、max对验证集、测试集进行min-max归一化。
8. 获取A股的交易日历
    
    ```python
    # 获取交易日历
    import akshare
    asaktool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()
    print(tool_trade_date_hist_sina_df)
    
    '''
    # 数据格式
    trade_date
    0     1990-12-19
    1     1990-12-20
    2     1990-12-21
    3     1990-12-24
    4     1990-12-25
              ...
    7823  2022-12-26
    7824  2022-12-27
    7825  2022-12-28
    7826  2022-12-29
    7827  2022-12-30
    '''
    ```
    
9. 对于单个行业指数的训练集train、验证集val、测试集test，假设训练集/验证集/测试集的长度为P，我们需要根据遍历日期区间[history_seq_len, P-history_seq_len]得到shape为[样本数, 时间窗口大小, 特征数]的数据。在遍历日期区间时，需要保障日期列符合A股的交易日历的顺序，比如：有一个样本的窗口大小为3，其日期为“1990-12-19，1990-12-20，1990-12-24”，由于其不符合交易日历的顺序“1990-12-19，1990-12-20，1990-12-21“，所以该样本交易数据存在缺失，被舍弃。
10. 将单个行业指数的训练集train拼接append到全局变量TRAIN中、验证集val拼接append到全局变量VAL中、测试集test拼接append到全局变量TEST中。
11. 遍历完所有行业指数后，得到所有行业指数的训练集TRAIN，验证集VAL，测试集TEST，根据每个数据的history_seq_len，计算得到每个数据的索引，然后构造出train_index、valid_index和test_index，使得我们能够通过index读取数据集中的数据。最后，将train_index, valid_index和test_index合并为index_list，前len(TRAIN)的index是训练集的index，len(TRAIN)至len(TRAIN)+ len(VAL)的数据是验证集的数据，len(TRAIN)+ len(VAL)至len(TRAIN)+ len(VAL) + len(TEST)
    
    ```python
        train_index = index_list[:train_num]
        valid_index = index_list[train_num: train_num + valid_num]
        test_index = index_list[train_num +
                                valid_num: train_num + valid_num + test_num]
    ```
    
12. 将训练集TRAIN，验证集VAL，测试集TEST依次append到全局变量ALL中。
13. 最后，将ALL、index以词典形式dump到不同的pkl文件里，格式分别为{’processed_data’: ALL}，{’train’: train_index, ‘valid’: valid_index, ‘test’: test_index}，如下代码：
    
    ```python
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(index, f)
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
            pickle.dump(data, f)
    ```