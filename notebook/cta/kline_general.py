from cta.kline import Kline
import pandas as pd
import numpy as np
from datetime import datetime, time
from collections import namedtuple

class Kline_General(Kline):
    def __init__(self, data, kfreq, **kwargs):
        super().__init__(data.sort_values(by='start', ascending=False, ignore_index=True), kfreq, **kwargs)

    def construct_all_klines(self, ckfreq, df = None):
        if df is None:
            df = self.copy()
        df = df.sort_values(by='start', ascending=False, ignore_index=True)

        period_set = namedtuple('period_set', ['p1', 'p2', 'p3'])
        period = namedtuple('period', ['begin_time', 'end_time'])

        ps = period_set(p1=period(begin_time=time(9, 0), end_time=time(11, 30)), # 早盘
                        p2=period(begin_time=time(13, 0), end_time=time(15, 0)), # 午盘
                        p3=period(begin_time=time(21, 0), end_time=time(2, 30))) # 晚盘

        def is_time_between(check_time, period):
            begin_time = period.begin_time
            end_time = period.end_time
            if begin_time < end_time:
                return check_time >= begin_time and check_time <= end_time
            else:  # crosses midnight
                return check_time >= begin_time or check_time <= end_time

        def period_identifier(check_time, period_set=ps):
            check_time = check_time.time()
            for i in period_set._fields:
                if is_time_between(check_time, period_set._asdict()[i]):
                    return i

        df['date'] = df.start.dt.strftime('%Y%m%d')
        df['period'] = df.start.apply(period_identifier)
        df['by_freq'] = np.arange(len(df))
        df.by_freq = df.groupby(['date','period']).by_freq.transform(lambda x: (x.max() - x) // ckfreq)
        grouped = df.groupby(['date', 'period', 'by_freq'])

        high_ = grouped.high.max().to_list()
        low_ = grouped.low.min().to_list()
        open_ = grouped.open.last().to_list()
        close_ = grouped.close.first().to_list()
        start_ = grouped.start.last().to_list()
        end_ = grouped.end.first().to_list()

        dfn = pd.DataFrame(data={'start': start_, 'end': end_,
                                 'low': low_, 'high': high_,
                                 'open': open_, 'close': close_, })
        dfn = dfn.sort_values(by='start', ascending=False, ignore_index=True)
        return Kline_General(data=dfn, kfreq=ckfreq)

    def construct_backward_klines(self, ckfreq=None, end=None, numkrows=None):
        df = self.copy()
        if(end != None):
        	df = df[df['end']<=end]
        df = df.sort_values(by='start', ascending=False, ignore_index=True)

        dfn = self.construct_all_klines(ckfreq, df)
        dfn = dfn.iloc[:numkrows]
        return Kline_General(data=dfn, kfreq=ckfreq)


