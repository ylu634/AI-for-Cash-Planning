import numpy as np
import pandas as pd
import re
from collections import defaultdict
import datetime
from dateutil.relativedelta import *
from collections import Counter
import math

def wrangle_data(path):
    f = pd.ExcelFile(path)
    data = f.parse('rate')
    constrains = f.parse('constrains')
    balance = f.parse('daily_bal')
    balance_a = f.parse('daily_bal_compa')
    balance_b = f.parse('daily_bal_compb')
    balance_c = f.parse('daily_bal_compc')
    fx = f.parse('fx')
    #tgl_td_due = f.parse('tgl_td_due')
    #---------------------------------------------------------------------
    #make sure the dates are in ascending order
    for ba in [balance,balance_a,balance_b,balance_c]:
        ba = ba.sort_values(['Date'],ascending = True).reset_index().drop(['index'], axis = 1)
    # ensure data['Date'] is the same as balance['Date]
    # 如果有dates 不在balance['Date']裡，會改成往後最近的balance['Date']
    data['Date'] = [balance['Date'][np.argmax([i <= out_date for out_date in balance['Date']])] for i in data['Date']]

    week_index = list(balance.loc[(balance['Date'].apply(lambda x: x.weekday()+1 == 5)) |\
                              (balance['Date'] == max(balance['Date']))].index)
    # #有大額付款的日期取代最接近的日期 (付款日在每個interval period的最後一天)
    # for m_payment_date in balance.loc[balance['monthly_payment_date'] == 1].index:
    #     week_index[np.argmin([abs(i-m_payment_date) for i in week_index])] = m_payment_date

    # # 不要有前後連續的數字 ---> divide conquer 的matrix shape 會不對
    # to_del = []
    # for i in range(len(week_index))[1:]:
    #     if week_index[i]-week_index[i-1] == 1:
    #         if week_index[i] not in balance.loc[balance['monthly_payment_date'] == 1].index:
    #             to_del.append(week_index[i])
    #         elif week_index[i-1] not in balance.loc[balance['monthly_payment_date'] == 1].index:
    #             to_del.append(week_index[i])
    # week_index = [i for i in week_index if i not in to_del]

    balance_daily = balance.copy()
    balance_a_daily = balance_a.copy()
    balance_b_daily = balance_b.copy()
    balance_c_daily = balance_c.copy()

    balance = balance.loc[week_index,:].reset_index().drop(['index'], axis = 1)
    balance_a = balance_a.loc[week_index,:].reset_index().drop(['index'], axis = 1)
    balance_b = balance_b.loc[week_index,:].reset_index().drop(['index'], axis = 1)
    balance_c = balance_c.loc[week_index,:].reset_index().drop(['index'], axis = 1)
    #把cash flow 加總起來
    #Balance 不會有差
    for data_2,data_1 in zip([balance_daily,balance_a_daily,balance_b_daily,balance_c_daily],\
                    [balance,balance_a,balance_b,balance_c]):
        for col in [i for i in data_2.columns if any(['in' in i, 'out' in i])]:

            cumsum = [sum(data_2[col][0:i+1]) for i in week_index] 

            data_1[col] = np.hstack(([cumsum[0]],[cumsum[i+1]-cumsum[i] for i in range(len(cumsum)-1)]))

    in_bal_days = []
    for i in range(len(balance_daily['Date'])-1):
        c = 7
        #c = 1
        # while balance_daily['Date'][i+1] != balance_daily['Date'][i]+relativedelta(days = c):
        #     c += 1
        in_bal_days.append(c)
    #in_bal_days.append(1)
    #by-week, should be one week time
    in_bal_days.append(7)

    #bank_limit = {x:y for x, y in zip(constrains['Bank'],constrains['Limit'])}
    bank_limit = defaultdict(int)
    for b in set(constrains['Bank']):
        df_temp = constrains.loc[constrains['Bank'] == b]
        bank_limit[b] = {datetime.datetime.strftime(y,'%Y%m%d'):x for x, y in zip(df_temp['Limit'],df_temp['Date'])}
    ###############
    #rate movement
    ##############
    df_rate = pd.DataFrame(columns = ['Date','Rate'],data = [[datetime.datetime(2021,1,1),-0.05],[datetime.datetime(2021,2,1),-0.1],\
                                                         [datetime.datetime(2021,6,1),-0.15]])
    df_rate = balance_daily[['Date']].merge(df_rate, how = 'left').ffill().fillna(0)
    df_rate['Rate'] = np.repeat(0, len(df_rate.index))
    #can apply time series forecast
    rate_path = {'USD':df_rate.copy(),\
                'TWD':df_rate.copy(),\
                'EUR':df_rate.copy(),\
                'JPY':df_rate.copy()}

    return data,fx, balance_daily,balance,\
        balance_a_daily ,balance_a,\
            balance_b_daily,balance_b,\
                balance_c_daily,balance_c,\
                    in_bal_days, rate_path, week_index, bank_limit