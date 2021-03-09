import pandas as pd
import numpy as np
import math

pd.set_option('display.max_columns', None)

import plotly.graph_objects as go
from datetime import datetime


def _hma(df, period=5):
    window_t_ewm = df['close'].ewm(span=period).mean()
    window_half_ewm = df['close'].ewm(span=period / 2).mean()
    diff = window_half_ewm * 2 - window_t_ewm
    df['hma'] = diff.ewm(span=int(math.sqrt(period))).mean()

# Data tools

class IndicatorMixin:
    """Util mixin indicator class"""

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        """Check if fillna flag is True.
        Args:
            series(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.
        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with "Nans" values"""
    df = df.copy()
    number_cols = df.select_dtypes("number").columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df


def _sma(series, periods: int, fillna: bool = False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def _ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def _get_min_max(series1: pd.Series, series2: pd.Series, function: str = "min"):
    """Find min or max value between two lists for each index"""
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)

class ADXIndicator(IndicatorMixin):
    """Average Directional Movement Index (ADX)
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).
    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.
    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        if self._window == 0:
            raise ValueError("window may not be 0")

        close_shift = self._close.shift(1)
        pdm = _get_min_max(self._high, close_shift, "max")
        pdn = _get_min_max(self._low, close_shift, "min")
        diff_directional_movement = pdm - pdn

        self._trs_initial = np.zeros(self._window - 1)
        self._trs = np.zeros(len(self._close) - (self._window - 1))
        self._trs[0] = diff_directional_movement.dropna()[0 : self._window].sum()
        diff_directional_movement = diff_directional_movement.reset_index(drop=True)

        for i in range(1, len(self._trs) - 1):
            self._trs[i] = (
                self._trs[i - 1]
                - (self._trs[i - 1] / float(self._window))
                + diff_directional_movement[self._window + i]
            )

        diff_up = self._high - self._high.shift(1)
        diff_down = self._low.shift(1) - self._low
        pos = abs(((diff_up > diff_down) & (diff_up > 0)) * diff_up)
        neg = abs(((diff_down > diff_up) & (diff_down > 0)) * diff_down)

        self._dip = np.zeros(len(self._close) - (self._window - 1))
        self._dip[0] = pos.dropna()[0 : self._window].sum()

        pos = pos.reset_index(drop=True)

        for i in range(1, len(self._dip) - 1):
            self._dip[i] = (
                self._dip[i - 1]
                - (self._dip[i - 1] / float(self._window))
                + pos[self._window + i]
            )

        self._din = np.zeros(len(self._close) - (self._window - 1))
        self._din[0] = neg.dropna()[0 : self._window].sum()

        neg = neg.reset_index(drop=True)

        for i in range(1, len(self._din) - 1):
            self._din[i] = (
                self._din[i - 1]
                - (self._din[i - 1] / float(self._window))
                + neg[self._window + i]
            )

    def adx(self) -> pd.Series:
        """Average Directional Index (ADX)
        Returns:
            pandas.Series: New feature generated.tr
        """
        dip = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            dip[i] = 100 * (self._dip[i] / self._trs[i])

        din = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            din[i] = 100 * (self._din[i] / self._trs[i])

        directional_index = 100 * np.abs((dip - din) / (dip + din))

        adx_series = np.zeros(len(self._trs))
        adx_series[self._window] = directional_index[0 : self._window].mean()

        for i in range(self._window + 1, len(adx_series)):
            adx_series[i] = (
                (adx_series[i - 1] * (self._window - 1)) + directional_index[i - 1]
            ) / float(self._window)

        adx_series = np.concatenate((self._trs_initial, adx_series), axis=0)
        adx_series = pd.Series(data=adx_series, index=self._close.index)

        adx_series = self._check_fillna(adx_series, value=20)
        return pd.Series(adx_series, name="adx")

    def adx_pos(self) -> pd.Series:
        """Plus Directional Indicator (+DI)
        Returns:
            pandas.Series: New feature generated.
        """
        dip = np.zeros(len(self._close))
        for i in range(1, len(self._trs) - 1):
            dip[i + self._window] = 100 * (self._dip[i] / self._trs[i])

        adx_pos_series = self._check_fillna(
            pd.Series(dip, index=self._close.index), value=20
        )
        return pd.Series(adx_pos_series, name="adx_pos")

    def adx_neg(self) -> pd.Series:
        """Minus Directional Indicator (-DI)
        Returns:
            pandas.Series: New feature generated.
        """
        din = np.zeros(len(self._close))
        for i in range(1, len(self._trs) - 1):
            din[i + self._window] = 100 * (self._din[i] / self._trs[i])

        adx_neg_series = self._check_fillna(
            pd.Series(din, index=self._close.index), value=20
        )
        return pd.Series(adx_neg_series, name="adx_neg")

