'''
Created on 14. nov. 2017

@author: michal
'''

def load_quotes(ticker):
    try:
        q = pd.read_csv("%s%s.mst"%('data/', ticker), sep=',', header=0, names=['Ticker', 'Date', 'o', 'h', 'l', 'c', 'v']) #Load the dataframe with headers
        q.index = pd.to_datetime(q['Date'], format='%Y%m%d')
        q['ch'] = q['c'] / q['c'].shift(1) - 1
        q.fillna(method='ffill', inplace=True)
        q.fillna(method="bfill", inplace=True)
        return q
    except OSError as e:
        print("Not found %s"%(ticker))

def plot_candlestick(tckr):
    from matplotlib.finance import candlestick2_ohlc
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker    
    fig, ax = plt.subplots()
    candlestick2_ohlc(ax,tckr['o'],tckr['h'],tckr['l'],tckr['c'],width=0.6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
    fig.autofmt_xdate()
    fig.tight_layout()
    return plt

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn import cluster
    
    # input
    ticker = 'CDPROJEKT'
    days = 250
    # end input

    tckr = load_quotes(ticker)
    tckr = tckr.iloc[-days:]

    #
    # clustering just by number of days
    kmeans = cluster.KMeans(n_clusters=5).fit(tckr[['c']])

    mean_vol = tckr['v'].sum()/days
    vol_bin = mean_vol/10.
    
    # create new DataFrame
    # for each price at a date, create that many instances as many bins 
    tckr['n_bins']=tckr['v']/vol_bin
    tckr['n_bins'] = tckr['n_bins'].astype(int)
    
    binned_close = np.array([])
    for i in range(days):
        # repeat closing price n_bins times
        binned_close = np.append( binned_close, values = np.repeat(tckr['c'].iloc[i],tckr['n_bins'].iloc[i]), axis=0 )

    #
    # clustering by volume
    kmeans_vol = cluster.KMeans(n_clusters=5).fit(binned_close.reshape(-1,1))

    #
    # plot
    my_plot = plot_candlestick(tckr)

    # red cluster centers by days
    for cc in kmeans.cluster_centers_:
        horiz_line_data = np.array([cc for i in range(2)])
        my_plot.plot([0,tckr.index.size-1], horiz_line_data, 'r--')

    # green cluster centers by volume
    for cc in kmeans_vol.cluster_centers_:
        horiz_line_data = np.array([cc for i in range(2)])
        my_plot.plot([0,tckr.index.size-1], horiz_line_data, 'g--')

    my_plot.show()

