
import numpy as np
import pandas as pd

class FinancialFunctions:
    def __init__(self):
        pass

    @staticmethod
    def bbands(close_prices, window, no_of_stdev):
        # rolling_mean = close_prices.rolling(window=window).mean()
        # rolling_std = close_prices.rolling(window=window).std()
        rolling_mean = close_prices.ewm(span=window).mean()
        rolling_std = close_prices.ewm(span=window).std()

        upper_band = rolling_mean + (rolling_std * no_of_stdev)
        lower_band = rolling_mean - (rolling_std * no_of_stdev)

        return rolling_mean, upper_band, lower_band

    @staticmethod
    def relative_strength_index(df, n):
        """Calculate Relative Strength Index(RSI) for given data.
        https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
        
        :param df: pandas.DataFrame
        :param n: 
        :return: pandas.DataFrame
        """
        i = 0
        UpI = [0]
        DoI = [0]
        while i + 1 <= df.index[-1]:
            UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
            DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
            if UpMove > DoMove and UpMove > 0:
                UpD = UpMove
            else:
                UpD = 0
            UpI.append(UpD)
            if DoMove > UpMove and DoMove > 0:
                DoD = DoMove
            else:
                DoD = 0
            DoI.append(DoD)
            i = i + 1
        UpI = pd.Series(UpI)
        DoI = pd.Series(DoI)
        PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
        NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
        RSI = pd.Series(round(PosDI * 100. / (PosDI + NegDI)), name='RSI_' + str(n))
        # df = df.join(RSI)
        return RSI

