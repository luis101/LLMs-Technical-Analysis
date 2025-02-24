

"""
Chart analysis by LLM modeLs

Using GPT-4o

"""


# Importing packages

import pandas as pd
import numpy as np

from datetime import datetime

import os
import gc
import re 
import base64

#from PIL import Image
#from tradingview_ta import TA_Handler, Interval, Exchange
#import talib as tl
import openai

import mplfinance as mpf
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#import torch
#from transformers import pipeline, BitsAndBytesConfig, LlavaNextProcessor, AutoProcessor


#Defining folder

output = "C:/Users/lukas/OneDrive/Dokumente/Research/LLMs Technical Analysis/"
path = "C:/Users/lukas/Anlagen/charts/"

#Importing firm universe

sp500 = pd.read_csv(output+"sp500-master/S&P 500 Historical components & Changes(08-17-2024).csv", delimiter=",")
sp500.dtypes

ticker = set([s for s in sp500.iloc[1,1].split(",")])

for i in range(2,len(sp500)):
    t = set([s for s in sp500.iloc[i,1].split(",")])
    for ele in t:
        ticker.add(ele)

print(len(ticker))


### One firm case

# General plotting

# Define the stock symbol and date range
stock_symbol = "AAPL"  
start_date = "1996-01-01"
end_date = "2024-11-30"

# Load historical data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
sdf = stock_data.stack(1)
sdf = sdf.reset_index(level=1)

sdf['year'] = sdf.index.year
sdf['month'] = sdf.index.month
sdf['day'] = sdf.index.day
    
sdf['numst'] = sdf.groupby(['year','month'])['Ticker'].transform('count')
sdf = sdf[(sdf['numst']>=15)]
   
# Generate folder 
fpath = path+str(stock_symbol)
if not os.path.isdir(fpath):
   os.makedirs(fpath)   
       
# Create a chart for each 90 day period:
   
for y in sorted(sdf['year'].unique()):
    for m in sorted(sdf['month'].unique()):       
        
        if (y==2024) & (m>=12):
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
        
# Obtain monthly dataframe

sdf = sdf.reset_index()
sdf['month_id'] = pd.to_datetime(sdf['Date']).dt.strftime('%Y-%m')

sdfm = sdf.groupby(['month_id']).last().reset_index()

sdfm['month_id'] = pd.to_datetime(sdfm['month_id']).dt.to_period("M")
sdfm.set_index('month_id', inplace=True)

# Resample the DataFrame with the complete date range
sdfm = sdfm.resample('M').asfreq()

#sdfm.set_index('Date', inplace=True)
# Create a complete date range (monthly frequency) from the first to the last date
#date_range = pd.date_range(start=sdfm.index.min(), end=sdfm.index.max(), freq='M')
# Reindex the DataFrame with the complete date range
#df_full = sdfm.reindex(date_range)

#Set signals
sdfm['Signal'] = np.nan
sdfm['SignalLS'] = np.nan

## Analyse chart by OpenAI model:
    
#Set client and image:
    
client = openai.OpenAI(
    api_key='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'
)

# Function to encode the images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Loop over months and determine model completions for each 90-day chart

for d in sdfm.index:
    
    # Path to your image
    image_path = path+stock_symbol+'/chart-'+str(d.year)+"-"+str(d.month)+'.jpg'
    
    # Getting the base64 string
    try:
        base64_image = encode_image(image_path)
    except:
        continue
    
    print(str(d.year)+"-"+str(d.month))
    
    # Define prompt and obtain model completion

    response = client.chat.completions.create(
      #model="gpt-4o",
      model="gpt-4o-mini",
      messages=[{
          "role": "user",
          "content": [{
              "type": "text",
              "text": "Based on the chart would you recommend to buy, hold, or sell?​ \
                  For each of these options answer with either yes, no, or maybe. You are forced to give an answer.​",
              #"text": "Is there a shoulder head shoulder pattern in the chart?​" - AAPL 1996-11,
            },
            {
              "type": "image_url",
              "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}"
              },
            },],
        }],
    )

    print(response.choices[0].message.content)
    text = response.choices[0].message.content 
    
    # Regex to find the word after **Buy:**, **Hold:**, and **Sell:**
    bhs = {}
    for sig in ['Buy', 'Hold', 'Sell']:
        for pattern in [r'\*\*' + sig + r':\*\*\s(\w+)', 
                        r'\*\*' + sig + r'\*\*:\s(\w+)', 
                        r'' + sig + r':\s(\w+)',
                        r'' + sig + r':\s\*\*(\w+)\*\*']:
            try:
                trdir = re.search(pattern, text).group(1)
                bhs[sig] = trdir
            except AttributeError:
                continue
            except IndexError:
                continue
            
    # Define trading strategy

    """
    Long only investor:
        
    if sell == "Yes": t+1 == 0
    if buy == "Yes": t+1 == 1

    if t == 0 & hold == "Yes" : t+1 == 0
    if t == 1 & hold == "Yes" : t+1 == 1

    Long/short investor:
        
    if sell == "Yes": t+1 == -1
    if buy == "Yes": t+1 == 1

    if t == 0 & hold == "Yes" : t+1 == 0
    if t == 1 & hold == "Yes" : t+1 == 1
    if t == -1 & hold == "Yes" : t+1 == -1
    if t == -1 & hold == "Yes" & Sell == "No" : t+1 == 0
    """
    
    lagsignall = sdfm.loc[sdfm.index==d]['Signal'].iloc[0]
    lagsignalls = sdfm.loc[sdfm.index==d]['SignalLS'].iloc[0]

    if bhs['Sell'] == "Yes": 
        sdfm.loc[sdfm.index==d+1, 'Signal'] = 0
    elif bhs['Buy'] == "Yes": 
        sdfm.loc[sdfm.index==d+1, 'Signal'] = 1
    elif (bhs['Hold'] == "Yes")&(lagsignall==0): 
        sdfm.loc[sdfm.index==d+1, 'Signal'] = 0
    elif (bhs['Hold'] == "Yes")&(lagsignall==1): 
        sdfm.loc[sdfm.index==d+1, 'Signal'] = 1    
    elif np.isnan(lagsignall): 
        sdfm.loc[sdfm.index==d+1, 'Signal'] = 0
    
    if bhs['Sell'] == "Yes": 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = -1
    elif bhs['Buy'] == "Yes": 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = 1
    elif (bhs['Hold'] == "Yes")&(lagsignalls==0): 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = 0
    elif (bhs['Hold'] == "Yes")&(lagsignalls==1): 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = 1  
    elif (bhs['Hold'] == "Yes")&(bhs['Buy'] == "No"): 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = 0
    elif (bhs['Hold'] == "Yes")&(bhs['Buy'] == "Maybe")&(lagsignalls==1): 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = 1  
    elif (bhs['Hold'] == "Yes")&(bhs['Sell'] == "No"): 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = 0       
    elif (bhs['Hold'] == "Yes")&(bhs['Sell'] == "Maybe")&(lagsignalls==-1): 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = -1
    elif np.isnan(lagsignalls): 
        sdfm.loc[sdfm.index==d+1, 'SignalLS'] = 0
        
        
#Compute cumulative returns

sdfm.loc[np.isnan(sdfm['SignalLS']), 'SignalLS'] = 0   

sdfm['ret'] = (sdfm['Adj Close'] - sdfm['Adj Close'].shift(1))/sdfm['Adj Close'].shift(1)      
sdfm['retl'] = sdfm['ret'].copy()  
sdfm.loc[sdfm['Signal']==0, 'retl'] = 0   
sdfm['retls'] = sdfm['ret'].copy()  
sdfm.loc[sdfm['SignalLS']==0, 'retls'] = 0   
sdfm.loc[sdfm['SignalLS']==-1, 'retls'] = -sdfm['ret']      

sdfm['Gret'] = (1+sdfm['ret']).cumprod() - 1   
sdfm['Sret'] = (1+sdfm['retl']).cumprod() - 1
sdfm['LSret'] = (1+sdfm['retls']).cumprod() - 1

#Plot

sdfm['Period_str'] = sdfm.index.astype(str)

# Plotting the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(sdfm['Period_str'], sdfm['Gret'], linestyle='-')

# Adding labels and title
plt.title('Cumulative Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)

# Show the plot
tick_positions = range(0, len(sdfm), 12)  # Show every second period
plt.xticks(ticks=tick_positions, labels=sdfm['Period_str'][tick_positions], rotation=45) # Rotate x-axis labels for better readability
plt.legend()
plt.tight_layout()
plt.show()












