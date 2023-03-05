## Python porting of QQE MOD by Mihkel00

<https://ru.tradingview.com/script/TpUW4muw-QQE-MOD/>

>Developed by [@edyatl](https://github.com/edyatl) March 2023 <edyatl@yandex.ru>

### Using [Python wrapper](https://github.com/TA-Lib/ta-lib-python) for [TA-LIB](http://ta-lib.org/) based on Cython instead of SWIG.

### Original Indicator code

```python
//@version=4
//By Glaz, Modified
//
study("QQE MOD")
RSI_Period = input(6, title='RSI Length')
SF = input(5, title='RSI Smoothing')
QQE = input(3, title='Fast QQE Factor')
ThreshHold = input(3, title="Thresh-hold")
//

src = input(close, title="RSI Source")
//

//
Wilders_Period = RSI_Period * 2 - 1


Rsi = rsi(src, RSI_Period)
RsiMa = ema(Rsi, SF)
AtrRsi = abs(RsiMa[1] - RsiMa)
MaAtrRsi = ema(AtrRsi, Wilders_Period)
dar = ema(MaAtrRsi, Wilders_Period) * QQE

longband = 0.0
shortband = 0.0
trend = 0

DeltaFastAtrRsi = dar
RSIndex = RsiMa
newshortband = RSIndex + DeltaFastAtrRsi
newlongband = RSIndex - DeltaFastAtrRsi
longband := RSIndex[1] > longband[1] and RSIndex > longband[1] ? 
   max(longband[1], newlongband) : newlongband
shortband := RSIndex[1] < shortband[1] and RSIndex < shortband[1] ? 
   min(shortband[1], newshortband) : newshortband
cross_1 = cross(longband[1], RSIndex)
trend := cross(RSIndex, shortband[1]) ? 1 : cross_1 ? -1 : nz(trend[1], 1)
FastAtrRsiTL = trend == 1 ? longband : shortband
////////////////////


length = input(50, minval=1, title="Bollinger Length")
mult = input(0.35, minval=0.001, maxval=5, step=0.1, title="BB Multiplier")
basis = sma(FastAtrRsiTL - 50, length)
dev = mult * stdev(FastAtrRsiTL - 50, length)
upper = basis + dev
lower = basis - dev
color_bar = RsiMa - 50 > upper ? #00c3ff : RsiMa - 50 < lower ? #ff0062 : color.gray


//
// Zero cross
QQEzlong = 0
QQEzlong := nz(QQEzlong[1])
QQEzshort = 0
QQEzshort := nz(QQEzshort[1])
QQEzlong := RSIndex >= 50 ? QQEzlong + 1 : 0
QQEzshort := RSIndex < 50 ? QQEzshort + 1 : 0
//  

Zero = hline(0, color=color.white, linestyle=hline.style_dotted, linewidth=1)

////////////////////////////////////////////////////////////////

RSI_Period2 = input(6, title='RSI Length')
SF2 = input(5, title='RSI Smoothing')
QQE2 = input(1.61, title='Fast QQE2 Factor')
ThreshHold2 = input(3, title="Thresh-hold")

src2 = input(close, title="RSI Source")
//

//
Wilders_Period2 = RSI_Period2 * 2 - 1


Rsi2 = rsi(src2, RSI_Period2)
RsiMa2 = ema(Rsi2, SF2)
AtrRsi2 = abs(RsiMa2[1] - RsiMa2)
MaAtrRsi2 = ema(AtrRsi2, Wilders_Period2)
dar2 = ema(MaAtrRsi2, Wilders_Period2) * QQE2
longband2 = 0.0
shortband2 = 0.0
trend2 = 0

DeltaFastAtrRsi2 = dar2
RSIndex2 = RsiMa2
newshortband2 = RSIndex2 + DeltaFastAtrRsi2
newlongband2 = RSIndex2 - DeltaFastAtrRsi2
longband2 := RSIndex2[1] > longband2[1] and RSIndex2 > longband2[1] ? 
   max(longband2[1], newlongband2) : newlongband2
shortband2 := RSIndex2[1] < shortband2[1] and RSIndex2 < shortband2[1] ? 
   min(shortband2[1], newshortband2) : newshortband2
cross_2 = cross(longband2[1], RSIndex2)
trend2 := cross(RSIndex2, shortband2[1]) ? 1 : cross_2 ? -1 : nz(trend2[1], 1)
FastAtrRsi2TL = trend2 == 1 ? longband2 : shortband2


//
// Zero cross
QQE2zlong = 0
QQE2zlong := nz(QQE2zlong[1])
QQE2zshort = 0
QQE2zshort := nz(QQE2zshort[1])
QQE2zlong := RSIndex2 >= 50 ? QQE2zlong + 1 : 0
QQE2zshort := RSIndex2 < 50 ? QQE2zshort + 1 : 0
//  

hcolor2 = RsiMa2 - 50 > ThreshHold2 ? color.silver :
   RsiMa2 - 50 < 0 - ThreshHold2 ? color.silver : na
plot(FastAtrRsi2TL - 50, title='QQE Line', color=color.white, transp=0, linewidth=2)
plot(RsiMa2 - 50, color=hcolor2, transp=50, title='Histo2', style=plot.style_columns)

Greenbar1 = RsiMa2 - 50 > ThreshHold2
Greenbar2 = RsiMa - 50 > upper

Redbar1 = RsiMa2 - 50 < 0 - ThreshHold2
Redbar2 = RsiMa - 50 < lower
plot(Greenbar1 and Greenbar2 == 1 ? RsiMa2 - 50 : na, title="QQE Up", style=plot.style_columns, color=#00c3ff, transp=0)
plot(Redbar1 and Redbar2 == 1 ? RsiMa2 - 50 : na, title="QQE Down", style=plot.style_columns, color=#ff0062, transp=0)
```
### Original Indicator Overview
This is a Pine Script code for a modified QQE (Quantitative Qualitative Estimation) indicator in TradingView. The code calculates two QQE values with different input parameters and plots their respective fast adaptive trendlines. One that is shown on the chart as columns, and the other "hidden" in the background which also has a 50 MA bollinger band acting as a zero line.
When both of them agree - you get a blue or a red bar. 

The code also plots Bollinger Bands around the trendlines and displays zero crosses. The indicator is primarily based on the RSI (Relative Strength Index) with different smoothing and threshold values. The trend is identified based on the crossover of the price with the fast adaptive trendlines.



