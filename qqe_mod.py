#!/usr/bin/env python3
"""
    Python porting of QQE MOD by Mihkel00
    https://ru.tradingview.com/script/TpUW4muw-QQE-MOD/
    Developed by @edyatl <edyatl@yandex.ru> March 2023
    https://github.com/edyatl

"""
# Standard imports
import pandas as pd
import numpy as np
import talib as tl
import os
from os import environ as env
from dotenv import load_dotenv
from binance import Client

# Load API keys from env
project_dotenv = os.path.join(os.path.abspath(""), ".env")
if os.path.exists(project_dotenv):
    load_dotenv(project_dotenv)

api_key, api_secret = env.get("ENV_API_KEY"), env.get("ENV_SECRET_KEY")

# Make API Client instance
client = Client(api_key, api_secret)

short_col_names = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "qav",
    "num_trades",
    "taker_base_vol",
    "taker_quote_vol",
    "ignore",
]

# Load Dataset
# Get last 500 records of ATOMUSDT 15m Timeframe
klines = client.get_klines(symbol="ATOMUSDT", interval=Client.KLINE_INTERVAL_15MINUTE)
data = pd.DataFrame(klines, columns=short_col_names)

# Convert Open and Close time fields to DateTime
data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
data["close_time"] = pd.to_datetime(data["close_time"], unit="ms")

#--------------------------INPUTS--------------------------------
# Common constant in calculations
CONST50 = 50

# First RSI input block
RSI_Period: int = 6
SF: int = 5
QQE: float = 3.0
ThreshHold: int = 3  # !not used input var

src: pd.Series = data["close"]

# Second RSI input block
RSI_Period2: int = 6
SF2: int = 5
QQE2: float = 1.61
ThreshHold2: int = 3

src2: pd.Series = data["close"]

# Bollinger input block
length: int = CONST50
mult: float = 0.35

#--------------------------FUNCIONS------------------------------
def cross(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    Returns a boolean Series indicating where two pandas Series have crossed.
    """
    # Ensure the inputs are pandas Series
    x = pd.Series(x)
    y = pd.Series(y)

    # Compare the values at corresponding indices
    cross_above = (x.shift(1) < y.shift(1)) & (x >= y)
    cross_below = (x.shift(1) > y.shift(1)) & (x <= y)

    # Combine the above and below crosses into a single boolean Series
    crosses = cross_above | cross_below

    return crosses

def qqe_hist(src: pd.Series, RSI_Period: int, SF: int, QQE: float) -> tuple:
    Wilders_Period: int = RSI_Period * 2 - 1
    Rsi = tl.RSI(src, RSI_Period)
    RsiMa = tl.EMA(Rsi, SF)
    AtrRsi = np.abs(np.roll(RsiMa.copy(), 1) - RsiMa)
    MaAtrRsi = tl.EMA(AtrRsi, Wilders_Period)
    dar = tl.EMA(MaAtrRsi, Wilders_Period) * QQE

    longband: pd.Series = np.zeros_like(src, dtype=float)
    shortband: pd.Series = np.zeros_like(src, dtype=float)
    trend: pd.Series = np.zeros_like(src, dtype=int)
    FastAtrRsiTL: pd.Series = np.zeros_like(src, dtype=float)

    DeltaFastAtrRsi = dar
    RSIndex = RsiMa
    newshortband = RSIndex + DeltaFastAtrRsi
    newlongband = RSIndex - DeltaFastAtrRsi

    for i in range(1, len(src)):
        if RSIndex[i - 1] > longband[i - 1] and RSIndex[i] > longband[i - 1]:
            longband[i] = max(longband[i - 1], newlongband[i])
        else:
            longband[i] = newlongband[i]

        if RSIndex[i - 1] < shortband[i - 1] and RSIndex[i] < shortband[i - 1]:
            shortband[i] = min(shortband[i - 1], newshortband[i])
        else:
            shortband[i] = newshortband[i]

    cross_1 = cross(np.roll(longband.copy(), 1), RSIndex)
    cross_2 = cross(RSIndex, np.roll(shortband.copy(), 1))

    for i in range(1, len(src)):
        trend[i] = (
            1
            if cross_2[i]
            else -1
            if cross_1[i]
            else 1
            if np.isnan(trend[i - 1])
            else trend[i - 1]
        )

    FastAtrRsiTL = np.where(trend == 1, longband, shortband)

    FastAtrRsiTL = np.nan_to_num(FastAtrRsiTL)

    return FastAtrRsiTL, RsiMa

def bollinger_uplower(
    FastAtrRsiTL: np.ndarray, length: int, mult: float, CONST50: int
) -> tuple:
    basis = tl.SMA(FastAtrRsiTL - CONST50, timeperiod=length)
    dev = mult * tl.STDDEV(FastAtrRsiTL - CONST50, length)

    upper = basis + dev
    lower = basis - dev
    return upper, lower

def zero_cross(src: pd.Series, RSIndex: np.ndarray, CONST50: int) -> tuple:
    QQEzlong: np.ndarray = np.zeros_like(src, dtype=int)
    QQEzshort: np.ndarray = np.zeros_like(src, dtype=int)

    for i in range(1, len(src)):
        QQEzlong[i] = QQEzlong[i - 1]
        QQEzshort[i] = QQEzshort[i - 1]

        QQEzlong[i] = QQEzlong[i] + 1 if RSIndex[i] >= CONST50 else 0
        QQEzshort[i] = QQEzshort[i] + 1 if RSIndex[i] < CONST50 else 0
    return QQEzlong, QQEzshort

def qqe_up_down(RsiMa, RsiMa2, upper, lower, ThreshHold2, CONST50) -> tuple:
    Greenbar1: np.ndarray = RsiMa2 - CONST50 > ThreshHold2
    Greenbar2: np.ndarray = RsiMa - CONST50 > upper

    Redbar1: np.ndarray = RsiMa2 - CONST50 < 0 - ThreshHold2
    Redbar2: np.ndarray = RsiMa - CONST50 < lower

    qqe_up_cond = Greenbar1 & Greenbar2
    qqe_down_cond = Redbar1 & Redbar2

    qqe_up = np.full_like(RsiMa2, fill_value=np.nan)
    qqe_down = np.full_like(RsiMa2, fill_value=np.nan)

    qqe_up = np.where(qqe_up_cond, RsiMa2 - CONST50, np.nan)
    qqe_down = np.where(qqe_down_cond, RsiMa2 - CONST50, np.nan)

    return qqe_up, qqe_down


def main():
    FastAtrRsiTL, RsiMa = qqe_hist(src, RSI_Period, SF, QQE)
    FastAtrRsi2TL, RsiMa2 = qqe_hist(src2, RSI_Period2, SF2, QQE2)

    upper, lower = bollinger_uplower(FastAtrRsiTL, length, mult, CONST50)

    qqe_up, qqe_down = qqe_up_down(RsiMa, RsiMa2, upper, lower, ThreshHold2, CONST50)

    res = pd.DataFrame(
        {

            "qqe_line": FastAtrRsi2TL,
            "histo2": RsiMa2,
            "qqe_up": qqe_up,
            "qqe_down": qqe_down,
        }
    )

    res.to_csv('qqe_mod-ATOMUSDT-15m.csv', index = None, header=True)


if __name__ == "__main__":
    main()
