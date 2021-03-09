from cta.helpers import *
import sys
import os
from collections import namedtuple

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


class Krow:
    """Object that mimicks a single k-line"""

    def __init__(self, high, low, open, close, start, kfreq):
        # high low open close: float
        # start_time pd.Timestamp
        # duration: int, min
        self._high = high
        self._low = low
        self._open = open
        self._close = close
        self._start = start
        self._kfreq = pd.Timedelta(pd.offsets.Minute(kfreq))
        self._end = start + self._kfreq

    def __str__(self):
        return f"K row: Open {self._open}, Close, {self._close}, High {self._high}, Low {self._low} \n" \
            f"Start {self._start}, End {self._end}, Duration {self._kfreq}"

    def to_df(self):
        return pd.DataFrame(
            data={
                'start': [
                    self._start], 'end': [
                    self._end], 'high': [
                    self._high], 'low': [
                        self._low], 'open': [
                            self._open], 'close': [self._close]})

    @classmethod
    def from_dfrow(cls, dfrow, kfreq):
        high = dfrow.loc['high']
        low = dfrow.loc['low']
        open = dfrow.loc['open']
        close = dfrow.loc['close']
        start = dfrow.loc['start']
        return cls(high, low, open, close, start, kfreq)


class Kline(pd.DataFrame):
    def __init__(self, data, kfreq, **kwargs):
        super().__init__(
            data=data.sort_values(
                by='start',
                ascending=False,
                ignore_index=True),
            **kwargs)
        self._kfreq = pd.Timedelta(pd.offsets.Minute(kfreq))

    @property
    def kfreq(self):
        return self._kfreq

    @classmethod
    def from_krows(cls, krow_list, **kwargs):
        """
         construct kline from list of k-rows
         return Kline object
        """
        kfreq = int(abs(krow_list[0]._kfreq.total_seconds() // 60))
        new_list = []
        for k in krow_list:
            new_list += [k.to_df()]
        df = pd.concat(new_list)
        return cls(
            data=df.sort_values(
                by='start',
                ascending=False,
                ignore_index=True),
            kfreq=kfreq,
            **kwargs)


    @staticmethod
    def add_kline(kline1, kline2):
        if kline1._kfreq != kline2._kfreq:
            raise ValueError('Cannot add k line with different k freq!')
        else:
            return Kline(data=pd.concat([kline1,
                                         kline2]).sort_values(by='start',
                                                              ascending=False,
                                                              ignore_index=True),
                         kfreq=int(kline1._kfreq.seconds // 60))

    @staticmethod
    def add_krow(kline, krow):
        if kline._kfreq != krow._kfreq:
            raise ValueError(
                'Cannot add k line to k row with different k freq!')
        else:
            return Kline(data=pd.concat([krow.to_df(), kline]).sort_values(
                by='start', ascending=False, ignore_index=True), kfreq=int(kline._kfreq.seconds // 60))
            # return Kline(com_data=pd.concat([krow.to_df(), kline]),
            # kfreq=int(kline._kfreq.seconds // 60))

    def plot_k(
            self,
            fractal_name='fractal',
            top_names=(
                't_j',
                't_l',
                't_r',
                't_a'),
            bottom_names=(
                'b_j',
                'b_l',
                'b_r',
                'b_a'),
        top_color='gold',
        bottom_color='lightskyblue',
        plot_ma=False,
        ma=3,
        pnl=None,
            pnl_font_size=10):
        temp = self.sort_values(by='start', ascending=True, ignore_index=True)
        trace = go.Candlestick(
            x=temp['start'],
            open=temp['open'], high=temp['high'],
            low=temp['low'], close=temp['close'],
            increasing_line_color='red', decreasing_line_color='green',
            name='all com_data'
        )

        if fractal_name in temp.columns:
            temp_t = temp.loc[temp[fractal_name].isin(top_names)]
            temp_b = temp.loc[temp[fractal_name].isin(bottom_names)]
            trace_t = go.Candlestick(
                x=temp_t['start'],
                open=temp_t['open'],
                high=temp_t['high'],
                low=temp_t['low'],
                close=temp_t['close'],
                increasing_line_color=top_color,
                decreasing_line_color=top_color,
                name='top')
            trace_b = go.Candlestick(
                x=temp_b['start'],
                open=temp_b['open'],
                high=temp_b['high'],
                low=temp_b['low'],
                close=temp_b['close'],
                increasing_line_color=bottom_color,
                decreasing_line_color=bottom_color,
                name='bottom')
            if plot_ma:
                fig = go.Figure(data=[trace,
                                      trace_t,
                                      trace_b,
                                      go.Scatter(x=temp.start, y=temp[f'high_ma{ma}'], mode='markers',
                                                 marker=dict(size=5 * np.ones(temp[f'high_ma{ma}'].shape),
                                                             color=temp[f'trend{ma}'].map(
                                                                 {0: 'gray', 1: 'fuchsia', -1: 'lime'}),
                                                             opacity=0.65)),
                                      go.Scatter(x=temp.start, y=temp[f'low_ma{ma}'], mode='markers',
                                                 marker=dict(size=5 * np.ones(temp[f'high_ma{ma}'].shape),
                                                             color=temp[f'trend{ma}'].map(
                                                                 {0: 'gray', 1: 'fuchsia', -1: 'lime'}),
                                                             opacity=0.65)),
                                      ])
            else:
                fig = go.Figure(data=[trace, trace_t, trace_b])
        else:
            if plot_ma:
                fig = go.Figure(data=[trace,
                                      go.Scatter(x=temp.start, y=temp[f'high_ma{ma}'], mode='markers',
                                                 marker=dict(size=5 * np.ones(temp[f'high_ma{ma}'].shape),
                                                             color=temp[f'trend{ma}'].map({0: 'gray', 1: 'fuchsia', -1: 'lime'}),
                                                             opacity=0.65)),
                                      go.Scatter(x=temp.start, y=temp[f'low_ma{ma}'], mode='markers',
                                                 marker=dict(size=5 * np.ones(temp[f'high_ma{ma}'].shape),
                                                             color=temp[f'trend{ma}'].map({0: 'gray', 1: 'fuchsia', -1: 'lime'}),
                                                             opacity=0.65)),
                                      ])
            else:
                fig = go.Figure(data=[trace])

        try:
            fig['com_data'][0]['showlegend'] = True
        except Exception:
            pass
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[15.1, 9], pattern="hour"),
                dict(bounds=[11.6, 12.9], pattern='hour')
            ]
        )
        fig.update_layout(plot_bgcolor='ghostwhite')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='ivory')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='ivory')

        if pnl is not None:
            if isinstance(pnl, pd.DataFrame):
                pnl = pnl.copy(deep=True)
            elif isinstance(pnl, str):
                pnl = pd.read_csv(pnl)
            else:
                raise ValueError('invalid value for pnl.')
            pnl = pnl[(pd.to_datetime(pnl['start_time']) > pd.to_datetime(self['start'][len(
                self) - 1])) & (pd.to_datetime(pnl['realized_time']) < pd.to_datetime(self['end'][0]))].copy()
            pnl['enter_price'] = pnl['exit_price'] + pnl['pnl']
            pnl['enter_price'].where(
                pnl['type'].to_numpy(
                    dtype='U1') == 'S',
                pnl['exit_price'] -
                pnl['pnl'],
                inplace=True)
            pnl['exit_price_adj'] = pnl['enter_price'] - pnl['pnl_adj']
            pnl['exit_price_adj'].where(
                pnl['type'].to_numpy(
                    dtype='U1') == 'S',
                pnl['enter_price'] +
                pnl['pnl_adj'],
                inplace=True)
            pnl['plot_color'] = 'crimson'
            pnl['plot_color'].where(
                pnl['pnl_adj'] > 0, 'darkgreen', inplace=True)
            n_pnl = len(pnl)
            v_left = [
                dict(
                    x0=pnl['start_time'].iloc[i_pnl],
                    x1=pnl['start_time'].iloc[i_pnl],
                    y0=0,
                    y1=1,
                    xref='x',
                    yref='paper',
                    line={
                        'width': 1,
                        'color': pnl['plot_color'].iloc[i_pnl]}) for i_pnl in range(n_pnl)]
            v_right = [
                dict(
                    x0=pnl['realized_time'].iloc[i_pnl],
                    x1=pnl['realized_time'].iloc[i_pnl],
                    y0=0,
                    y1=1,
                    xref='x',
                    yref='paper',
                    line={
                        'width': 1,
                        'color': pnl['plot_color'].iloc[i_pnl]}) for i_pnl in range(n_pnl)]
            h_enter = [
                dict(
                    x0=pnl['start_time'].iloc[i_pnl],
                    x1=pnl['realized_time'].iloc[i_pnl],
                    y0=pnl['enter_price'].iloc[i_pnl],
                    y1=pnl['enter_price'].iloc[i_pnl],
                    xref='x',
                    yref='y',
                    line={
                        'width': 0.65,
                        'color': 'silver',
                        'dash': 'dot'}) for i_pnl in range(n_pnl)]
            h_exit = [
                dict(
                    x0=pnl['start_time'].iloc[i_pnl],
                    x1=pnl['realized_time'].iloc[i_pnl],
                    y0=pnl['exit_price_adj'].iloc[i_pnl],
                    y1=pnl['exit_price_adj'].iloc[i_pnl],
                    xref='x',
                    yref='y',
                    line={
                        'width': 0.65,
                        'color': 'silver',
                        'dash': 'dot'}) for i_pnl in range(n_pnl)]
            fig.update_layout(
                shapes=v_left + v_right + h_enter + h_exit,
                annotations=[
                    dict(
                        x=pnl['realized_time'].iloc[i_pnl],
                        y=0.05,
                        xref='x',
                        yref='paper',
                        showarrow=False,
                        xanchor='left',
                        text=' {:.1f} ({})'.format(
                            pnl['pnl_adj'].iloc[i_pnl],
                            pnl['type'].iloc[i_pnl]),
                        font={
                            'size': pnl_font_size,
                            'color': pnl['plot_color'].iloc[i_pnl]}) for i_pnl in range(n_pnl)])
            """for i_pnl in range(n_pnl):
                fig.add_vrect(x0=pnl['start_time'].iloc[i_pnl], x1=pnl['realized_time'].iloc[i_pnl], fillcolor=pnl['plot_color'].iloc[i_pnl], opacity=0.3, line_width=0)"""

        fig.show()


    def ma_trend_filter(self, ma=3, in_range_ratio=0.75):
        df = self.copy()
        # df = df.sort_values(by='start', ascending=True)
        # TODO: HMA is bugged
        df[f'high_ma{ma}'] = df.iloc[::-1].high._hma(period=ma).mean()
        df[f'low_ma{ma}'] = df.iloc[::-1].low._hma(period=ma).mean()
        df[f'low_ma_diff{ma}'] = -(df[f'low_ma{ma}'].shift(-1) - df[f'low_ma{ma}']) * 100 / df.close
        df[f'high_ma_diff{ma}'] = -(df[f'high_ma{ma}'].shift(-1) - df[f'high_ma{ma}']) * 100 / df.close
        df['low_ma_abs_diff_ma'] = df.iloc[::-1][f'low_ma_diff{ma}'].abs()._hma(period=15).mean()
        df[f'in_range{ma}'] = df[f'low_ma_diff{ma}'].abs()._hma(period=1).mean() < df[
            'low_ma_abs_diff_ma'] * in_range_ratio
        df[f'in_range{ma}'] = df[f'in_range{ma}'] & ~(~df[f'in_range{ma}'].shift(1).astype(bool) & ~df[f'in_range{ma}'].shift(-1).astype(bool))
        df[f'trend{ma}'] = 0
        df.loc[(df[f'low_ma_diff{ma}'] > 0) & ~ df[f'in_range{ma}'], f'trend{ma}'] = 1
        df.loc[(df[f'low_ma_diff{ma}'] < 0) & ~ df[f'in_range{ma}'], f'trend{ma}'] = -1
        df = df.drop(
            columns=[f'low_ma_diff{ma}', f'high_ma_diff{ma}', 'low_ma_abs_diff_ma'])
        return Kline(data=df, kfreq=int(self._kfreq.seconds // 60 % 60))

    def construct_klines(self, ckfreq, start=None, end=None):
        raise NotImplementedError

    _pan_length = 120

    def compute_slzt(
            self,
            kfreq_1=5,
            kfreq_2=30,
            return_percentile=(
                25,
                50,
                75),
            ma=10,
            map_fun=None):
        assert not self._kfreq.seconds % 60
        kfreq = int(self._kfreq.seconds // 60)
        small_first = kfreq_1 < kfreq_2
        kfreq_1, kfreq_2 = np.sort(np.array((kfreq_1, kfreq_2), dtype=np.int))
        assert kfreq <= kfreq_1 < kfreq_2 and not kfreq_1 % kfreq and not kfreq_2 % kfreq_1
        assert kfreq_1 <= self._pan_length and kfreq_2 <= self._pan_length
        kline_1 = self if kfreq_1 == self._kfreq else self.construct_klines(
            kfreq_1)
        kline_2 = self if kfreq_2 == self._kfreq else self.construct_klines(
            kfreq_2)
        f_1 = int(kfreq_1 // kfreq)
        f_21 = int(kfreq_2 // kfreq_1)
        diff_1 = (kline_1['high'] - kline_1['low']).ewm(halflife=ma).mean()
        diff_2 = (
            kline_2['high'] -
            kline_2['low']).repeat(f_21).reset_index()[0].ewm(
            halflife=ma).mean()
        if map_fun is None:
            def map_fun(x): return x
        else:
            assert callable(map_fun)
        foo = (diff_1 / diff_2).repeat(f_1).reset_index()[0]
        if not small_first:
            foo = 1 / foo
        self['slzt'] = map_fun(foo).fillna(0)
        if return_percentile is not None:
            return np.percentile(self['slzt'], (25, 50, 75))
        return self
