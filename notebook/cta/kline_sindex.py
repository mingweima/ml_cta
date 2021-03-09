from cta.kline import Kline
import pandas as pd
import numpy as np


class Kline_Sindex(Kline):
    def __init__(self, data, kfreq, **kwargs):
        # com_data = com_data[com_data['end'].apply( lambda x : x.hour != 9 ) | com_data['end'].apply( lambda x : x.minute != 30 )]
        # com_data = com_data[com_data['end'].apply( lambda x : x.hour != 13 ) | com_data['end'].apply( lambda x : x.minute != 0 )]
        super().__init__(
            data.sort_values(
                by='start',
                ascending=False,
                ignore_index=True),
            kfreq,
            **kwargs)

    def time_seek(self, date=None, forward=True):
        df = self.copy()
        df = df.sort_values(by='start', ascending=False)
        # remove 9: 30 (period of call auction)
        df = df[df['end'].apply(lambda x: x.hour != 9)
                | df['end'].apply(lambda x: x.minute != 30)]
        if forward:
            try:
                if date is None:
                    return df['end'].iloc[-1]
                else:
                    return df['end'][df['end'] >= date].iloc[-1]
            except BaseException:
                raise RuntimeError('Out of Range Error')
        else:
            try:
                if date is None:
                    return df['end'].iloc[0]
                else:
                    return df['end'][df['end'] <= date].iloc[0]
            except BaseException:
                raise RuntimeError('Out of Range Error')

    def construct_klines(self, ckfreq, start=None, end=None):
        """
        :param ckfreq: int, mins
        :param start: pd.Timestamp
        :param end: pd.Timestamp
        :return: Kline object with kfreq=kfreq, start=start, end=end
        """
        df = self.copy()
        # df = df.sort_values(by='start', ascending=False, ignore_index=True)
        # remove 9: 30 (period of call auction)
        df = df[df['end'].apply(lambda x: x.hour != 9)
                | df['end'].apply(lambda x: x.minute != 30)]
        df = df[df['end'].apply(lambda x: x.hour != 13)
                | df['end'].apply(lambda x: x.minute != 0)]
        start = self.time_seek(start)
        end = self.time_seek(end, False)

        if start.hour >= 13:
            trade_begin = pd.Timestamp(year=start.year,
                                       month=start.month,
                                       day=start.day,
                                       hour=13, minute=1)
            delta = ((start - trade_begin).seconds / 60) // ckfreq
            start0 = trade_begin + \
                pd.Timedelta(pd.offsets.Minute((delta + 1) * ckfreq))
        else:
            trade_begin = pd.Timestamp(year=start.year,
                                       month=start.month,
                                       day=start.day,
                                       hour=9, minute=31)
            delta = ((start - trade_begin).seconds / 60) // ckfreq
            start0 = trade_begin + \
                pd.Timedelta(pd.offsets.Minute((delta + 1) * ckfreq))

        group_h = df.groupby((df['end'] >= start) & (df['end'] < start0))
        close_ = [group_h.first().close[True]]
        high_ = [group_h.max().high[True]]
        low_ = [group_h.min().low[True]]
        open_ = [group_h.last().open[True]]
        end_ = [group_h.first().end[True]]
        start_ = [group_h.last().start[True]]

        df = df[(df['end'] >= start0) & (df['end'] <= end)]
        partition = df['end'].apply(lambda x: pd.Timestamp.strftime(
            x, '%Y%m%d') + str(0.5 * (x.hour >= 13))).to_numpy()
        group = df.groupby(partition)
        for i in np.unique(partition):
            d = len(group.get_group(i))
            s = list(np.arange(d) // ckfreq)
            s.reverse()
            group_i = group.get_group(i).groupby(np.array(s))
            close_ = group_i.first().close.to_list() + close_
            high_ = group_i.max().high.to_list() + high_
            low_ = group_i.min().low.to_list() + low_
            open_ = group_i.last().open.to_list() + open_
            end_ = group_i.first().end.to_list() + end_
            start_ = group_i.last().start.to_list() + start_
        dfn = pd.DataFrame(data={'start': start_, 'end': end_,
                                 'low': low_, 'high': high_,
                                 'open': open_, 'close': close_, })
        dfn = dfn.sort_values(by='start', ascending=False, ignore_index=True)
        return Kline_Sindex(data=dfn, kfreq=ckfreq)

    def construct_backward_klines(self, ckfreq=None, end=None, numkrows=None):
        """
        :param ckfreq: int, mins
        :param end: pd.Timestamp
        :param numkrows: int
        :return: Kline object with kfreq=kfreq, end=start, and timespan prescribed by numkrows pr timedelta
        """
        def Trade_Begin(date, morning=True):
            if morning:
                return pd.Timestamp(
                    year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=9,
                    minute=31)
            else:
                return pd.Timestamp(
                    year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=13,
                    minute=1)

        df = self.copy()
        # df = df.sort_values(by='start', ascending=False, ignore_index=True)
        df = df[df['end'].apply(lambda x: x.hour != 9)
                | df['end'].apply(lambda x: x.minute != 30)]
        calendar = df['end'].apply(
            lambda x: pd.Timestamp.strftime(
                x, '%Y-%m-%d')).unique()
        end = self.time_seek(end, False)
        loc = list(pd.to_datetime(calendar) <= end).index(True)

        if end.hour >= 13:
            morning = False
        else:
            morning = True
        trade_begin = Trade_Begin(end, morning)

        num_1 = ((end - trade_begin).seconds / 60) // ckfreq + 1
        # num of kfreq within the present morning/afternoon
        if numkrows <= num_1:
            start = trade_begin + \
                pd.Timedelta(pd.offsets.Minute((num_1 - numkrows) * ckfreq))
        else:
            num_2 = 119 // ckfreq + 1
            # num of kfreq within one single morning/afternoon
            num_3 = (numkrows - num_1 - 1) // num_2 + 1
            # num of morning/afternoon retrieved

            if num_3 % 2 == 0:
                trade_begin_1 = Trade_Begin(pd.to_datetime(
                    calendar)[loc + int(num_3 // 2)], morning)
                num_4 = numkrows - num_1 - (num_3 - 1) * num_2
                start = trade_begin_1 + \
                    pd.Timedelta(pd.offsets.Minute((num_2 - num_4) * ckfreq))
            else:
                trade_begin_1 = Trade_Begin(pd.to_datetime(
                    calendar)[loc + int((morning + num_3) // 2)], not morning)
                num_4 = numkrows - num_1 - (num_3 - 1) * num_2
                start = trade_begin_1 + \
                    pd.Timedelta(pd.offsets.Minute((num_2 - num_4) * ckfreq))

        return self.construct_klines(start=start, end=end, ckfreq=ckfreq)
