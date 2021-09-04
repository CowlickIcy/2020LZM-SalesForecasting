import numpy as np
import pandas as pd
import fbprophet as fpp
import warnings
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


# load data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
data = pd.read_csv('grouped.csv',parse_dates=['DATE'], index_col='DATE',date_parser=dateparse)
data = data.rename(columns={'Unnamed: 0':'index_number'})
data['AMOUNT'] = np.log(data['AMOUNT'])

# 得到不同ID的index分界点
def get_index(data):
    data_duplicated = data.drop_duplicates(subset='ID')
    data_index = data_duplicated['index_number'].tolist()

    return data_index

index_list = get_index(data)


# 将不同ID商品按照时间存入字典
data_amount = data['AMOUNT']
data_allid = {}
data_allid_cut = {}
index_list_cut = [0]
for i in range(len(index_list) - 1):
    end_index = np.int(np.floor(index_list_cut[i] + (index_list[i + 1] - index_list[i]) * 0.75))
    index_list_cut.append(end_index)
    data_allid[i] = data_amount[index_list[i]:index_list[i + 1]]
    data_allid_cut[i] = data_allid[i][0:index_list_cut[i + 1] - index_list_cut[i]]


# 重命名data
def data_alter(data_allid_i):
    data_allid_i = data_allid_i.sort_index()
    data_a = data_allid_i.reset_index()
    data_a.columns = ['ds', 'y']

    return data_a


def prophet_model(dataloader):
    model = fpp.Prophet(
 #                       growth='logistic'
                        )
    forecast = model.fit(dataloader)

    future = model.make_future_dataframe(periods=12)
    forecast = model.predict(future)

    return forecast


result = prophet_model(data_alter(data_allid[0]))