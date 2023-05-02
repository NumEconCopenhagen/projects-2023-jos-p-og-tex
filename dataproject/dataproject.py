# Import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import requests # library for making HTTP requests
import datetime as dt # library for handling date and time objects
from PIL import Image
from scipy.stats import pearsonr
import yfinance as yf
from yahoofinancials import YahooFinancials


def import_data_dmi(api_key, DMI_URL):
    """Requesting and importing data from DMI""" 
    
    # Issues a HTTP GET request
    r = requests.get(DMI_URL, params={'api-key': api_key})

    # Extract JSON data
    json = r.json()

    # Convert JSON object to a Pandas DataFrame
    df = pd.json_normalize(json['features'])  

    df['time'] = pd.to_datetime(df['properties.observed'])

    # Generate a list of unique parameter ids
    parameter_ids = df['properties.parameterId'].unique() 
    
    return parameter_ids


def selecting_data_dmi(parameterId):
    """Selecting relevant data from DMI""" 

    # We use an API-key given to us from DMI's database to retrieve data
    api_key = 'bd463c7d-f6f8-431d-a5a7-c466766a8363'
    DMI_URL = 'https://dmigw.govcloud.dk/v2/metObs/collections/observation/items'

    # Specify the desired start and end time
    start_time = pd.Timestamp(2022, 1, 1)
    end_time = pd.Timestamp(2023, 1, 1)

    # Specify our selection of station IDs
    all_stationsDK = [
        '05005', '05009', '05015', '05031', '05035', '05042', '05065', 
        '05070', '05075', '05081', '05085', '05089', '05095', '05105', 
        '05109', '05135', '05140', '05150', '05160', '05165', '05169', 
        '05185', '05199', '05202', '05205', '05220', '05225', '05269', 
        '05272', '05276', '05277', '05290', '05296', '05300', '05305', 
        '05320', '05329', '05343', '05345', '05350', '05355', '05365', 
        '05375', '05381', '05395', '05400', '05406', '05408', '05435', 
        '05440', '05450', '05455', '05469', '05499', '05505', '05510', 
        '05529', '05537', '05545', '05575', '05735', '05880', '05889', 
        '05935', '05945', '05970', '05986', '05994'
    ]

    # Specify one or more parameter IDs or all_parameters
    #parameterId = ['precip_past1h']

    # Derive datetime specifier string
    datetime_str = start_time.tz_localize('UTC').isoformat() + '/' + end_time.tz_localize('UTC').isoformat()

    dfs = []
    for station in all_stationsDK:
        for parameter in parameterId:
            # Specify query parameters
            params = {
                'api-key' : api_key,
                'datetime' : datetime_str,
                'stationId' : station,
                'parameterId' : parameter,
                'limit' : '300000',  # max limit
            }

            # Submit GET request with url and parameters
            r = requests.get(DMI_URL, params=params)

            # Extract JSON object
            json = r.json()
            
            # Convert JSON object to a MultiIndex DataFrame and add to list
            dfi = pd.json_normalize(json['features'])
            if dfi.empty is False:
                dfi['Time'] = pd.to_datetime(dfi['properties.observed'])
                dfi[['station', 'parameter']] = station, parameter
                dfi = dfi.set_index(['station', 'Time'])
                dfi = dfi['properties.value'].unstack(['station'])
                dfs.append(dfi)

    df = pd.concat(dfs, axis='columns').sort_index()

    # Alter double-indexed column
    df.reset_index(inplace=True)

    # Drop last row, as this is wrong date
    df.drop(df.tail(1).index, inplace=True) 

    return df


def import_data_yahoo(from_time, to_time):
    """Requesting and importing data from Yahoo Finance""" 

    OMXC25 = yf.download('^OMXC25', start=from_time, end=to_time, progress=False)
    OMXC25.reset_index(inplace=True) 
    OMXC25['Date'] =  pd.to_datetime(OMXC25['Date'])

    # Cleaning the data from irrelevant variables 
    OMXC25.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

    return OMXC25


def explore_data(x_value,y_value,title,xlabel,ylabel,min_y):
    """Creating figures to explore the data""" 

    # a. create the figure
    fig = plt.figure()

    # b. plot
    ax = fig.add_subplot(1,1,1)

    ax.bar(x_value,y_value)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin=min_y);

    return ax


def correlation(df):

    # extracting the columns to explore correlation for
    precip = df['Precip']
    change_in_stock = df['Change_in_stock']

    #calculation of correlation coefficient and p-value between precipitation and the change in OMXC25
    correlation_coefficient, p_value = pearsonr(precip, change_in_stock)
    print(f"Correlation coefficient: {correlation_coefficient:.2f}")
    print(f"p-value: {p_value:.2f}")

    return None

