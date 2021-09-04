# %% 1.导入包
import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams  # 设定画布大小
from statsmodels.tsa.arima_model import ARIMA  # ARIMA模型    (p,d,q)
from statsmodels.tsa.stattools import adfuller  # 判断时序数据稳定性的第二种方法

import warnings

warnings.filterwarnings('ignore')
# p--代表预测模型中采用的时序数据本身的滞后数(lags) ,也叫做AR/Auto-Regressive项
# d--代表时序数据需要进行几阶差分化，才是稳定的，也叫Integrated项。
# q--代表预测模型中采用的预测误差的滞后数(lags)，也叫做MA/Moving Average项

# 2.设定画布的大小
rcParams['figure.figsize'] = 15, 6

# %% 3.读取数据
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('grouped.csv', parse_dates=['DATE'], index_col='DATE', date_parser=dateparse)
data = data.rename(columns={'Unnamed: 0': 'index_number'})


# %% 4.get_index
def get_index(data):
    data_duplicated = data.drop_duplicates(subset='ID')
    data_index = data_duplicated['index_number'].tolist()

    return data_index


index_list = get_index(data)


# %% 4.检查时序数据的稳定性

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


# %% model

def ARIMA_fit(dataload):
    ts = dataload.sort_index()
    # test_stationarity(ts)
    ts_log = np.log(ts)

    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)

    model = ARIMA(ts_log, order=(2, 1, 0))
    results_ARIMA = model.fit(disp=-1)

    # plt.plot(ts_log_diff)
    # plt.plot(results_ARIMA.fittedvalues, color='red')  #拟合值
    # plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

    # ARIMA拟合的其实是一阶差分ts_log_diff，predictions_ARIMA_diff[i]是第i个月与i-1个月的ts_log的差值。
    # 由于差分化有一阶滞后，所以第一个月的数据是空的，

    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    # print (predictions_ARIMA_diff.head)

    # 累加现有的diff，得到每个值与第一个月的差分（同log底的情况下）。
    # 即predictions_ARIMA_diff_cumsum[i] 是第i个月与第1个月的ts_log的差值。

    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    # print(predictions_ARIMA_diff_cumsum.head())

    predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

    # predictions_ARIMA_log.head()

    # plt.plot(ts_log)
    # plt.plot(predictions_ARIMA_log)

    # 预测最后一个月

    predictions_ARIMA_log = results_ARIMA.forecast()[0]

    predictions_ARIMA = np.exp(predictions_ARIMA_log)

    # plt.plot(ts)
    # plt.plot(predictions_ARIMA)
    # plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))

    return predictions_ARIMA


prediction_list = []
deviation_ratio_list = []

import time

start = time.time()
for i in range(len(index_list)):
    ts = data.iloc[index_list[i]:index_list[i + 1] - 1]['AMOUNT']
    prediction_list.append(ARIMA_fit(ts))

    deviation_ratio_list.append(((ARIMA_fit(ts)[0]) - data.iat[index_list[i + 1] - 1, 2])
                                / np.max([ARIMA_fit(ts)[0], data.iat[index_list[i + 1] - 1, 2]]))

    end = time.time()

    print('Coding Time: %s seconds' % (end - start))
