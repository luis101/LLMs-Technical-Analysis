


import pandas as pd
import numpy as np
from datetime import datetime
import os
import gc

from tradingview_ta import TA_Handler, Interval, Exchange

import mplfinance as mpf
import yfinance as yf
import plotly.graph_objects as go

import torch
from transformers import pipeline, BitsAndBytesConfig, LlavaNextProcessor, AutoProcessor

output = "C:/Users/lukas/Downloads/"


# Import SP500 tickers

sp500 = pd.read_csv(output+"sp500-master/S&P 500 Historical components & Changes(08-17-2024).csv", delimiter=",")
sp500.dtypes

ticker = set([s for s in sp500.iloc[1,1].split(",")])

for i in range(2,len(sp500)):
    t = set([s for s in sp500.iloc[i,1].split(",")])
    for ele in t:
        ticker.add(ele)

print(len(ticker))
#print(ticker)

sp500[['year','month','day']] = sp500['date'].str.split('-', expand=True)

# Define the stock symbol and date range
stock_symbol = "AAPL"  
start_date = "2024-08-01"
end_date = "2024-10-31"

# Load historical data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
sdf = stock_data.stack(1)
sdf = sdf.reset_index(level=1)

#save = dict(fname=output+'tsave30.jpg')
#mpf.plot(sdf, type='candle', figsize=(8,6), tight_layout=True, style='tradingview', savefig=save)
plot = mpf.plot(sdf, type='candle', figsize=(8,6), tight_layout=True, style='tradingview')


# Run LLM model

#Quantization for efficient loading

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

#Load the LLaVA model:

model_id = "llava-hf/llava-1.5-7b-hf"
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})



# General plotting

# Define the date range
 
start_date = "1996-01-01"
end_date = "2024-10-31"

stocks = pd.DataFrame()
stocklist = [] 

# Define the stock symbol and loop over symbols

for stock_symbol in ticker:
    
    # Load historical data
    
    print("Ticker: "+stock_symbol)
    
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    sdf = stock_data.stack(1)
    sdf = sdf.reset_index(level=1)
    
    #stocklist.append(sdf)  
    #stocks = pd.concat([stocks,sdf])
    
    sdf['year'] = sdf.index.year
    sdf['month'] = sdf.index.month
    sdf['day'] = sdf.index.day
    
    sdf['numst'] = sdf.groupby(['year','month'])['Ticker'].transform('count')
    sdf = sdf[(sdf['numst']>=17)]
    
    path = output+'charts/'+str(stock_symbol)
    if not os.path.isdir(path):
       os.makedirs(path)
       
    # Create a chart for each 90 day period:
       
    for y in sorted(sdf['year'].unique()):
        for m in sorted(sdf['month'].unique()):       
            
            if (y==2024) & (m>=11):
                continue
            
            print(str(y)+"-"+str(m))
            
            if (m>=3):
                sdfc = sdf[(sdf['year']==y)&(sdf['month']<=m)&(sdf['month']>=m-2)]
            elif (m<3):
                sdfc = sdf[((sdf['year']==y)&(sdf['month']<=m))|((sdf['year']==y-1)&(sdf['month']>=m+10))]
            
            #sdfc = sdf[(sdf['year']==y)&(sdf['month']==m)]
            
            if len(sdfc) > 50:
                pass
            else:
                continue
            
            save = dict(fname=path+'/chart-'+str(y)+"-"+str(m)+'.jpg')
            mpf.plot(sdfc, type='candle', figsize=(8,6), tight_layout=True, style='tradingview', 
                     savefig=save, warn_too_much_data=10000)
            
    #del sdf
    #gc.collect()

   
   






