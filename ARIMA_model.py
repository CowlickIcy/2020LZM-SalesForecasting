# %%
# 1导入包
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams  # 设定画布大小
from statsmodels.tsa.stattools import adfuller  # 判断时序数据稳定性的第二种方法
from statsmodels.tsa.seasonal import seasonal_decompose  # 分解(decomposing) 可以用来把时序数据中的趋势和周期性数据都分离出来
from statsmodels.tsa.stattools import acf, pacf  # 自相关   偏相关
from statsmodels.tsa.arima_model import ARIMA  # ARIMA模型    (p,d,q)
# p--代表预测模型中采用的时序数据本身的滞后数(lags) ,也叫做AR/Auto-Regressive项
# d--代表时序数据需要进行几阶差分化，才是稳定的，也叫Integrated项。
# q--代表预测模型中采用的预测误差的滞后数(lags)，也叫做MA/Moving Average项
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 2.设定画布的大小
rcParams['figure.figsize'] = 15, 6

# SuperParam

halflife = 0.5
alpha = 1 - np.exp(np.log(0.5) / halflife)

# %%
# 3.读取数据
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('salesdata.csv', parse_dates=['DATE'], index_col='DATE', date_parser=dateparse)


# 根据DeID划分4类
data_DeID2 = data.loc[(data['DeID'] == 2)]
data_DeID3 = data.loc[(data['DeID'] == 3)]
data_DeID13 = data.loc[(data['DeID'] == 13)]
data_DeID14 = data.loc[(data['DeID'] == 14)]


# 再分配index
data_DeID2.insert(loc=0, column='index_num', value=np.arange(len(data_DeID2['ID'])))
#data_DeID3['index_number'] = np.arange(len(data_DeID3['ID']))
#data_DeID13['index_number'] = np.arange(len(data_DeID13['ID']))
#data_DeID14['index_number'] = np.arange(len(data_DeID14['ID']))


# 根据ID获取起止index
data_DeID2_duplicated = data_DeID2.drop_duplicates(subset='ID')
data_DeID2_index = data_DeID2_duplicated['index_number'].tolist()

data_DeID3_duplicated = data_DeID3.drop_duplicates(subset='ID')
data_DeID3_index = data_DeID3_duplicated['index_number'].tolist()

data_DeID13_duplicated = data_DeID13.drop_duplicates(subset='ID')
data_DeID13_index = data_DeID13_duplicated['index_number'].tolist()

data_DeID14_duplicated = data_DeID14.drop_duplicates(subset='ID')
data_DeID14_index = data_DeID14_duplicated['index_number'].tolist()


# %%

# 4.检查时序数据的稳定性
# 均值，方差，自方差与时间无关，数据是稳定的
# python判断时序数据稳定性的两种方法
# Rolling statistic-- 即每个时间段内的平均的数据均值和标准差情况。
# Dickey-Fuller Test(迪基-福勒检验) -- 这个比较复杂，大致意思就是在一定置信水平下，对于时序数据假设 Null hypothesis: 非稳定。
# if 通过检验值(statistic)< 临界值(critical value)，则拒绝null hypothesis，即数据是稳定的；反之则是非稳定的。

def test_stationarity(timeseries):
    # 这里以一年为一个窗口，每一个时间t的值由它前面12个月（包括自己）的均值代替，标准差同理。
    rolmean = timeseries.rolling(12).mean()  # 均值
    rolstd = timeseries.rolling(12).std()  # 标准差

    '''

    画示例图

    #plot rolling statistics:
    fig = plt.figure()#画子图
    fig.add_subplot()  #fig.add_subplot(111)    参数1：子图总行数    参数2：子图总列数   参数3：子图位置
    orig = plt.plot(timeseries, color = 'blue',label='Original')    #原始数据
    mean = plt.plot(rolmean , color = 'red',label = 'rolling mean')    #均值
    std = plt.plot(rolstd, color = 'black', label= 'Rolling standard deviation')    #标准差

    plt.legend(loc = 'best')   #图例
    plt.title('Rolling Mean & Standard Deviation')   #标题
    plt.show(block=False)

    '''

    # Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    # dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value

    print(dfoutput)


# %%

def ARIMA_result(dataload):
    ts = dataload['AMOUNT']
    ts = ts.sort_values(by=['DATE'])
    test_stationarity(ts)

    # 数据归一化
    ts_norm = ts.apply(lambda x: (x - np.min(ts)) / (np.max(ts) - np.min(ts)))
    ts_norm['intercept'] = 1.0

    # 使用平滑方法，检测和去除趋势

    # ts_log = np.log(ts)

    moving_avg = ts.rolling(12).mean()

    # 然后作差：
    ts_moving_avg_diff = ts - moving_avg  # 取对数的减去平滑以后的
    ts_moving_avg_diff.dropna(inplace=True)
    test_stationarity(ts_moving_avg_diff)  # 红色为差值的均值，黑色为差值的标准差

    # 引入指数加权移动平均
    expweighted_avg = ts.ewm(span=12).mean()
    ts_ewma_diff = ts - expweighted_avg
    test_stationarity(ts_ewma_diff)

    # Differencing--差分
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)  # 删掉有缺失的数据
    test_stationarity(ts_diff)

    ts_diff = ts.diff(1)
    ts_diff.dropna(inplace=True)
    test_stationarity(ts_diff)

    # diff()函数是求导数和差分的
    ts_diff1 = ts.diff(1)  # 一阶差分
    ts_diff2 = ts.diff(2)  # 二阶差分
    ts_diff1.plot()
    ts_diff2.plot()

    # 基本已经没有变化。所以使用一阶差分。

    # step2： 得到参数估计值p，d，q之后，生成模型ARIMA（p，d，q）p=2   d=1   q=2

    model = ARIMA(ts, order=(2, 1, 2))
    results_ARIMA = model.fit(disp=-1)

    # ARIMA拟合的其实是一阶差分ts_log_diff，predictions_ARIMA_diff[i]是第i个月与i-1个月的ts_log的差值。
    # 由于差分化有一阶滞后，所以第一个月的数据是空的，
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    # print (predictions_ARIMA_diff.head)
    # 累加现有的diff，得到每个值与第一个月的差分（同log底的情况下）。
    # 即predictions_ARIMA_diff_cumsum[i] 是第i个月与第1个月的ts_log的差值。
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    # 先ts_log_diff => ts_log=>ts_log => ts
    # 先以ts_log的第一个值作为基数，复制给所有值，然后每个时刻的值累加与第一个月对应的差值(这样就解决了，第一个月diff数据为空的问题了)
    # 然后得到了predictions_ARIMA_log => predictions_ARIMA

    predictions_ARIMA = pd.Series(ts.ix[0], index=ts.index)
    predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum, fill_value=0)

    return predictions_ARIMA
