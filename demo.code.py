# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:47:42 2024

@author: liang
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
import pytz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

 
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
from alpaca_trade_api.rest import REST, TimeFrame

from lumibot.backtesting import YahooDataBacktesting
from lumibot.backtesting import BacktestingBroker
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader

from pytickersymbols import PyTickerSymbols

import time
from datetime import datetime
import pytz
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
import schedule
from keras.models import load_model
import joblib

random.seed(1693)
np.random.seed(1693)
tf.random.set_seed(1693)



BASE_URL = "https://paper-api.alpaca.markets/v2"
KEY_ID = 'PK3FOTS6P0FWF2CF9235'
SECRET_KEY = '1d1jwGolg1ixJmZhOwGcm3Q1koKuxsL2rVuqIoNg'


api = REST(KEY_ID, SECRET_KEY, base_url= "https://paper-api.alpaca.markets/v2")


#################################################################                             #################################
##########################################################################  [Demonstration]  ##################################
#################################################                                                ##############################

trading_client = TradingClient(KEY_ID, SECRET_KEY)

class demonstration():
    def __init__(self, trading_client):
        self.trading_client = trading_client

    def trade(self):
        action = random.choice(['buy', 'sell', 'hold'])  # Randomly choose between 'buy' and 'sell'
        qty = 0.001  # Set the quantity for both actions
        if action in ['buy', 'sell']:
            marketorderdata = MarketOrderRequest(
                symbol='BTC/USD',  # Ticker to trade
                qty=qty,  # Quantity of shares, fractional shares are allowed
                side=action,  # 'buy' or 'sell' based on the random choice
                time_in_force='gtc'  # 'gtc' = good till cancelled
            )
            self.trading_client.submit_order(marketorderdata)
            print(f"Executed {action} order for {qty} share of BTC/USD")
        elif action == 'hold':
            print("Holding position, no action taken.")

def scheduler(strategy_instance):
    # Clear any existing jobs
    schedule.clear()

    # Schedule the job every 6 second
    schedule.every(.1).minutes.do(strategy_instance.trade)

    # Run the scheduler in a loop
    while True:
        schedule.run_pending()
        time.sleep(1)
        
my_strategy = demonstration(trading_client)

# Start the scheduler
scheduler(my_strategy)