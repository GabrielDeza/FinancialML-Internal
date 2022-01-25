import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def greedy_gain(period_dates, pred, target,df_open,df_close,freq_type ='day'):
    if freq_type =='day':
        freq_str = '%Y-%m-%d'
    if freq_type =='hour':
        freq_str = '%Y-%m-%d %H:%M:%S'
    returns = []
    gain = 0
    for i in range(period_dates.shape[0]):
        c = df_close[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        o = df_open[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        if pred[i] > 0:
            #Open > Close so you want to buy a stock at the close price right before market opens and then sell it instantly at the open price to make a quick buck
            gain += c - o
            #the percentage return is  100*((Open - Close)/Close)
            returns.append(100*((c-o)/o))
    init_investement = df_open[pd.Timestamp(period_dates[0]).strftime(freq_str)]
    net_gain_percent = (100*gain)/init_investement
    percent_days_traded = 100 * (len(returns) / len(period_dates))
    return net_gain_percent,init_investement, returns,percent_days_traded

def threshold_gain(period_dates, pred, target,df_open,df_close,freq_type ='day'):
    if freq_type =='day':
        freq_str = '%Y-%m-%d'
    if freq_type =='hour':
        freq_str = '%Y-%m-%d %H:%M:%S'
    k =5
    returns = []
    gain = 0
    for i in range(k,len(period_dates)):
        #get the past k true "diff" values (ie: Open - Close)
        past_vals = target[i-k:i-1]
        std_dev = np.std(past_vals)
        avg = np.mean(past_vals)
        c = df_close[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        o = df_open[pd.Timestamp(period_dates[i]).strftime(freq_str)]
        if pred[i] >= avg+std_dev and pred[i] > 0:
            #if I believe that Open>Close and that this is a really good opportunity compared to the past 5 days
            gain += c - o
            returns.append(100*((c-o)/o))
    init_investment = df_open[pd.Timestamp(period_dates[0]).strftime(freq_str)]
    net_gain_percent = (100*gain)/init_investment
    percent_days_traded = 100* (len(returns)/len(period_dates))
    return net_gain_percent, returns,percent_days_traded



def plot_return_distributions(returns,PDT,show_figs =True,save_figs= False,save_path = '.'):
    fig,ax = plt.subplots(nrows=2, ncols=3,figsize=(14, 9))
    for j,dataset in enumerate(['Training', 'Testing','Adversarial']):
        for i,strategy in enumerate(['Greedy', 'Threshold']):
            x = np.array(returns[dataset][strategy])
            if len(x)!=0:
                bins = 30
                ax[i,j].hist(x, density=True, bins=bins, label="Returns")
                ax[i,j].legend(loc="upper left")
            ax[i,j].set_ylabel('Percentage')
            ax[i,j].set_xlabel('Returns')
            ax[i,j].set_title(f"{strategy} Strategy on {dataset} \n Days Traded = {round(PDT[dataset][strategy],2)}%",fontsize=7)
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)
    if save_figs:
        plt.savefig(save_path, dpi=100)
    if show_figs:
        plt.show()
    else:
        plt.clf()


'''
#mn, mx = ax[i,j].get_xlim()
#ax[i,j].set_xlim(mn, mx)
#ax[i,j].set_ylim(-0.05,1.0)
#kde_xs = np.linspace(mn, mx, 300)
#kde = st.gaussian_kde(x)
#ax[i,j].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
'''