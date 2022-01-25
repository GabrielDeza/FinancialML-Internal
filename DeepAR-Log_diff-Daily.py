import pandas as pd
import numpy as np
import mxnet as mx
import datetime as dt
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from random import choice
from scipy.stats import zscore
from string import ascii_uppercase
from gluonts.dataset.util import to_pandas
from gluonts.mx.distribution import StudentTOutput
from gluonts.mx.distribution import GaussianOutput
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
import scipy.stats as st
import pickle
import argparse
import datetime
from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset)
import seaborn as sns
from helper_financial_metrics import *
sns.set(style="darkgrid")


########################################################################
##############   PARAMETERS & HYPERPARAMETERS  #########################
########################################################################
parser = argparse.ArgumentParser(description='Gradient Attack')
parser.add_argument('--company', default='ADSK', type=str, help='Company')
parser.add_argument('--train_length', default=1, type=int, help='Length of training set')
parser.add_argument('--validation_length', default=250, type=int, help='Length of validation set')
parser.add_argument('--test_length', default=380, type=int, help='Length of testing set')
parser.add_argument('--prediction_length', default=5, type=int, help='Prediction Length')
parser.add_argument('--epochs', default=5, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--nbpe', default=200, type=int, help='Number of batches per epoch')
parser.add_argument('--student_t', default=True, type=bool, help='Student-T distribution or guassian')
parser.add_argument('--twitter', default=True, type=bool, help='Using twitter as dynamic features or not')
parser.add_argument('--save_figs', default=False, type = bool, help = 'decides if the figures and txt files should be saved or not')
parser.add_argument('--show_figs', default=True, type=bool, help='decides if the figures should be shown to the screen')
parser.add_argument('--seed', default=6, type=int, help='seed')

args = parser.parse_args()
plottype ="log_diff"
company = args.company
train_length = args.train_length
validation_length = args.validation_length
test_length = args.test_length
prediction_length = args.prediction_length
step_size = prediction_length
epochs = args.epochs
batch_size = args.batch_size
num_batches_per_epoch = args.nbpe
student_t = args.student_t
twitter = args.twitter


seed = args.seed
mx.random.seed(seed)
np.random.seed(seed)
end_filename = f"twitter={twitter}_type={plottype}_Student_T={student_t}_prediction_length={prediction_length}" \
               f"_train_length={train_length}_test_length={test_length}_BS={batch_size}_NBpE={num_batches_per_epoch}_epochs={epochs}.png"


########################################################################
########################   LOADING DATA  ###############################
########################################################################
#df is the main df, df_target is the target we want to predict, feat_dynamic = covars, plotting_feat_dynamic = covars + prediction length of time.

filename = f"/Users/gabriel/Desktop/Fixed_Stock_Project/CHTR_1d.csv"
columns = ["Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score","positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count", "neutral_count","General_score"]

df = pd.read_csv(filename, index_col=1)
df = df.drop(['Unnamed: 0'],axis = 1)
df['diff'] = df['Open']-df['Close']
df['log_diff'] = np.log((df['Open']/df['Close']))
df_target = df[plottype]
df_close = df['Close']
df_open = df['Open']

########################################################################################
######################## Z-score normalization of the covariates #######################
########################################################################################
df[columns] = df[columns].apply(zscore)

start_date  =[df_target.index[0] for _ in range(1)]
target_values = np.zeros((1, train_length))
target_values[0,:] = df_target[:train_length].to_numpy()

#dynamic features for training
feat_dynamic = np.zeros((len(columns), train_length))
for i,col in enumerate(columns):
    feat_dynamic[i,:] = df[col][:train_length].to_numpy()
feat_dynamic = [feat_dynamic]

training_data = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: FDR}
            for (target, start,FDR) in zip(target_values, start_date, feat_dynamic)], freq ="1B")

########################################################################################
###### Plotting version of Training Set (Length of "train_length + pred_length")########
########################################################################################

plotting_feat_dynamic = np.zeros((len(columns),train_length+prediction_length))
for i,col in enumerate(columns):
    plotting_feat_dynamic[i,:] = df[col][:train_length+prediction_length].to_numpy()
plotting_feat_dynamic = [plotting_feat_dynamic]

plotting_training_data = ListDataset([{
    FieldName.TARGET: target,
    FieldName.START: start,
    FieldName.FEAT_DYNAMIC_REAL: FDR}
        for (target, start, FDR) in zip(target_values, start_date, plotting_feat_dynamic)], freq ="1B")

########################################################################################
################# Validation Set (Length of "validation_length") #######################
########################################################################################
validation_target_values = np.zeros((1, validation_length))
validation_target_values[0,:] = df_target[:validation_length].to_numpy()

feat_dynamic_val = np.zeros((len(columns), validation_length))
for i,col in enumerate(columns):
    feat_dynamic_val[i,:] = df[col][:validation_length].to_numpy()
feat_dynamic_val = [feat_dynamic_val]

validation_data = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: FDR}
            for (target, start,FDR) in zip(validation_target_values, start_date, feat_dynamic_val)], freq ="1B")

########################################################################################
#################### Testing Set (Length of "testing_length") ##########################
########################################################################################

testing_target_values = np.zeros((1, test_length))
testing_target_values[0,:] = df_target[:test_length].to_numpy()

plotting_test_feat_dynamic =np.zeros((len(columns),test_length+prediction_length))
for i,col in enumerate(columns):
    plotting_test_feat_dynamic[i,:] = df[col][:test_length+prediction_length].to_numpy()
plotting_test_feat_dynamic = [plotting_test_feat_dynamic]

testing_data = ListDataset([{
        FieldName.TARGET: target,
        FieldName.START: start,
        FieldName.FEAT_DYNAMIC_REAL: FDR}
    for (target, start, FDR) in zip(testing_target_values, start_date,plotting_test_feat_dynamic)], freq ="1B")

########################################################################################
################################ Initializing Model ####################################
########################################################################################

estimator = DeepAREstimator(
        freq="1B",
        scaling=True,
        prediction_length=prediction_length,
        batch_size=batch_size,
        distr_output= StudentTOutput() if student_t == True else GaussianOutput(),
        use_feat_dynamic_real= True if twitter == True else False,
        trainer=Trainer(
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            learning_rate=1e-3,),)

########################################################################################
################################# Training Model #######################################
########################################################################################

grad_str = ''.join(choice(ascii_uppercase) for i in range(200))
grad_str = f"./gradient_dir/" +grad_str
predictor = estimator.train(training_data=training_data,validation_data = validation_data, num_workers = 0, grad_str = grad_str)

########################################################################################
###################### Training Rolling Dataset for Plotting ###########################
########################################################################################

dataset_rolled = generate_rolling_dataset(
        dataset=plotting_training_data,
        start_time=pd.Timestamp(df_target.index[0], freq="1B"),
        end_time=pd.Timestamp(df_target.index[train_length+prediction_length], freq = "1B"),
        strategy=StepStrategy(prediction_length=prediction_length, step_size = step_size))

########################################################################################
############################ Plotting Training Forecasts ###############################
########################################################################################

prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()

for i,train_dict in enumerate(dataset_rolled):
    #step 1 create the start
    start_ = [train_dict['start']]
    #step 2, create the target
    interim_target = train_dict['target'].reshape((1,-1))
    #step 3, create the feat dynamic real
    fdr =  [plotting_feat_dynamic[0][:,:train_length + prediction_length - (i*(step_size))]]
    #print(f"shape makes sense ? -> target  = {interim_target.shape} fdr = {fdr.shape}, orig fdr = {plotting_feat_dynamic.shape}")
    train_data = ListDataset([{
            FieldName.TARGET: target,
            FieldName.START: start,
            FieldName.FEAT_DYNAMIC_REAL: FDR}
                for (target, start, FDR) in zip(interim_target, start_, fdr)], freq ="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data, predictor.predict(train_data)):
        to_pandas(test_entry).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
        if i !=0:
            forecast.plot(color='m', prediction_intervals=prediction_intervals, show_mean=True, zorder=10)

plt.title(f"Resulting Forecasts on the Training Set")
plt.ylabel("Change in Open and Close Price")
legend = ["observations", "median prediction", "Mean of distribution"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Date")
plt.legend(legend, loc="upper left")

start_filename =f"./plots/{company}/training/"
filename = start_filename + end_filename
if args.save_figs:
    plt.savefig(filename, dpi =400)
if args.show_figs:
    plt.show()
else:
    plt.clf()

mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
        dataset_rolled, predictor=predictor, num_samples=100)
mx.random.seed(seed)
np.random.seed(seed)
training_agg_metrics, _ = Evaluator(num_workers = 0)(ts_it, forecast_it)

mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
        dataset_rolled, predictor=predictor, num_samples=100)
mx.random.seed(seed)
np.random.seed(seed)
forecasts = list(forecast_it) #this is the pd series
tss = list(ts_it) #this is the dataframe

training_target_vals = np.array([])
training_pred_vals = np.array([])
training_dates = np.array([], dtype = np.datetime64)
for i,(target, forecast) in enumerate(zip(tss, forecasts)):
    #first get the dates. they are added in opposite order so thats why accuracy was wrong before.
    training_dates = np.concatenate((target.index[-step_size:],training_dates))
    #Now get the target values
    training_target_vals = np.concatenate((target.values[-step_size:].reshape((-1,)),training_target_vals))
    #Lastly, get the predictions
    training_pred_vals = np.concatenate((np.mean(forecast.samples, axis =0).reshape((-1,)), training_pred_vals))
training_target_vals = training_target_vals[prediction_length:]
training_pred_vals = training_pred_vals[prediction_length:]
training_dates = training_dates[prediction_length:]

########################################################################
######################  MAKING TEST SET   ##############################
########################################################################

dataset_rolled = generate_rolling_dataset(
        dataset=testing_data,
        start_time=pd.Timestamp(df_target.index[validation_length], freq="1B"),
        end_time=pd.Timestamp(df_target.index[test_length+prediction_length], freq = "1B"),
        strategy=StepStrategy(prediction_length=prediction_length, step_size = step_size))

########################################################################
#####################  PLOTTING TEST DATA  #############################
########################################################################

prediction_intervals=[50.0, 90.0]
for i,(train_dict) in enumerate(dataset_rolled):
    start_ = [train_dict['start']]
    interim_target = train_dict['target'].reshape((1,-1))
    fdr =  [plotting_test_feat_dynamic[0][:,:test_length + prediction_length - (i*(step_size))]]

    train_data = ListDataset([{FieldName.TARGET: target,FieldName.START: start,FieldName.FEAT_DYNAMIC_REAL: FDR} for (target, start, FDR)
                              in zip(interim_target, start_, fdr)], freq ="1B")
    mx.random.seed(seed)
    np.random.seed(seed)
    for test_entry, forecast in zip(train_data, predictor.predict(train_data)):
        to_pandas(test_entry)[validation_length:].plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
        if i!=0:
            forecast.plot(color='g', prediction_intervals=prediction_intervals, show_mean=True)


plt.title(f"Resulting Forecasts on a Rolling Test Set")
plt.ylabel("Change in Open and Close Price")
legend = ["Target", "Regular Median prediction", "Regular Mean of distribution"] + [f"Regular {k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Date")
plt.legend(legend, loc="upper left")

start_filename =f"./plots/{company}/testing/"
filename = start_filename + end_filename
if args.save_figs:
    plt.savefig(filename, dpi =400)
if args.show_figs:
    plt.show()
else:
    plt.clf()

########################################################################
#####################  metrics of TEST DATA  ###########################
########################################################################

mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
    dataset_rolled, predictor=predictor, num_samples=100)
forecasts = list(forecast_it)  # this is the pd series
tss = list(ts_it)  # this is the dataframe
mx.random.seed(seed)
np.random.seed(seed)
forecast_it, ts_it = make_evaluation_predictions(
    dataset_rolled, predictor=predictor, num_samples=100)
testing_agg_metrics, _ = Evaluator(num_workers=0)(ts_it, forecast_it)

testing_target_vals = np.array([])
testing_pred_vals = np.array([])
testing_dates = np.array([], dtype = np.datetime64)
for i,(target, forecast) in enumerate(zip(tss, forecasts)):
    #first get the dates. they are added in opposite order so thats why accuracy was wrong before.
    testing_dates = np.concatenate((target.index[-step_size:],testing_dates))
    #Now get the target values
    testing_target_vals = np.concatenate((target.values[-step_size:].reshape((-1,)),testing_target_vals))
    #Lastly, get the predictions
    testing_pred_vals = np.concatenate((np.mean(forecast.samples, axis =0).reshape((-1,)), testing_pred_vals))

testing_target_vals = testing_target_vals[5:]
testing_pred_vals = testing_pred_vals[5:]
testing_dates = testing_dates[5:]

t =training_agg_metrics
v =testing_agg_metrics


########################################################################
#############  Binary accuracy shennagins  #############################
########################################################################

new_dates = [pd.Timestamp(training_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,training_dates.shape[0])]
close_vals_training = df_close[new_dates].to_numpy().reshape((-1,training_target_vals.shape[0])).reshape((-1,))
training_target_diff = (np.exp(np.array(training_target_vals))* close_vals_training) - close_vals_training
training_pred_diff = (np.exp(np.array(training_pred_vals))* close_vals_training) - close_vals_training

new_dates = [pd.Timestamp(testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,testing_dates.shape[0])]
close_vals_testing = df_close[new_dates].to_numpy().reshape((-1,testing_target_vals.shape[0])).reshape((-1,))
testing_target_diff = (np.exp(np.array(testing_target_vals))* close_vals_testing) - close_vals_testing
testing_pred_diff = (np.exp(np.array(testing_pred_vals))* close_vals_testing) - close_vals_testing

train_bias_acc_up = (np.sum(training_target_diff >= 0, axis=0)) / training_target_diff.shape[0]
train_bias_acc_down =  (np.sum(training_target_diff < 0, axis=0)) / training_target_diff.shape[0]
train_acc = np.sum(np.sign(training_target_diff) == np.sign(training_pred_diff)) / training_target_diff.shape[0]

test_bias_acc_up = (np.sum(testing_target_diff >= 0, axis=0)) / testing_target_diff.shape[0]
test_bias_acc_down = (np.sum(testing_target_diff < 0, axis=0)) / testing_target_diff.shape[0]
test_acc = np.sum(np.sign(testing_target_diff) == np.sign(testing_pred_diff)) / testing_target_diff.shape[0]


####################################################################
##################  Getting financial metrics ######################
####################################################################

#passive gain
passive_gain_train = 100*(df_open[pd.Timestamp(training_dates[-1]).strftime('%Y-%m-%d')] - df_open[pd.Timestamp(training_dates[0]).strftime('%Y-%m-%d')])/df_open[pd.Timestamp(training_dates[0]).strftime('%Y-%m-%d')]
passive_gain_test = 100*(df_open[pd.Timestamp(testing_dates[-1]).strftime('%Y-%m-%d')] - df_open[pd.Timestamp(testing_dates[0]).strftime('%Y-%m-%d')])/df_open[pd.Timestamp(testing_dates[0]).strftime('%Y-%m-%d')]

#greedy gain
greedy_gain_train,train_invest, training_greedy_returns,training_greedy_PDT = greedy_gain(training_dates, training_pred_diff,training_target_diff, df_open,df_close)
greedy_gain_test, test_invest, testing_greedy_returns,testing_greedy_PDT = greedy_gain(testing_dates, testing_pred_diff, testing_target_diff,df_open,df_close)


#threshold gain
threshold_gain_train, training_threshold_returns,training_threshold_PDT = threshold_gain(training_dates, training_pred_diff,training_target_diff,df_open,df_close)
threshold_gain_test,testing_threshold_returns,testing_threshold_PDT = threshold_gain(testing_dates, testing_pred_diff,testing_target_diff,df_open,df_close)

print(f"Training: RMSE: {round(t['RMSE'],3)} MAPE: {round(t['MAPE'],3)} CRPS: {round(t['mean_wQuantileLoss'],3)}"
      f" Only Up: {round(train_bias_acc_up,3)} Only Down: {round(train_bias_acc_down,3)} Accuracy: {round(train_acc,3)}"
      f" Initial Investment: {round(train_invest,2)}$ Passive Return: {round(passive_gain_train,1)}% Greedy Return: {round(greedy_gain_train,1)}%"
      f" Threshold Return: {round(threshold_gain_train,1)}%")
print(f"Training: RMSE: {round(v['RMSE'],3)} MAPE: {round(v['MAPE'],3)} CRPS: {round(v['mean_wQuantileLoss'],3)}"
      f" Only Up: {round(test_bias_acc_up,3)} Only Down: {round(test_bias_acc_down,3)} Accuracy: {round(test_acc,3)}"
      f" Initial Return: {round(test_invest,2)}$ Passive Return: {round(passive_gain_test,1)}% Greedy Return: {round(greedy_gain_test,1)}%"
      f" Threshold Return: {round(threshold_gain_test,1)}%")

ret_dict ={}
training_metrics = {}
testing_metrics = {}

for feat in ['RMSE','MAPE','mean_wQuantileLoss']:
    training_metrics[feat] = t[feat]
    testing_metrics[feat] = v[feat]

training_metrics['Biased Acc'] = (train_bias_acc_up,train_bias_acc_down)
training_metrics['Pred Acc'] = train_acc
training_metrics['Passive Gain'] = passive_gain_train
training_metrics['Greedy Gain'] = greedy_gain_train
training_metrics['Threshold Gain'] = threshold_gain_train
training_metrics['initial investment'] = train_invest

testing_metrics['Biased Acc'] = (test_bias_acc_up, test_bias_acc_down)
testing_metrics['Pred Acc'] = test_acc
testing_metrics['Passive Gain'] = passive_gain_test
testing_metrics['Greedy Gain'] = greedy_gain_test
testing_metrics['Threshold Gain'] = threshold_gain_test
testing_metrics['initial investment'] = test_invest

ret_dict['training'] = training_metrics
ret_dict['testing'] = testing_metrics

returns_dict = {'Training':{'Greedy':training_greedy_returns,'Threshold':training_threshold_returns},
                'Testing': {'Greedy':testing_greedy_returns,'Threshold':testing_threshold_returns}}

PDT_dict  = {'Training':{'Greedy':training_greedy_PDT,'Threshold':training_threshold_PDT},
                'Testing': {'Greedy':testing_greedy_PDT,'Threshold':testing_threshold_PDT}}

plot_return_distributions(returns_dict,PDT_dict)

import os
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
#dir_path = f"./metrics_pickle/twitter={twitter}_type={plottype}_Student_T={student_t}_prediction_length={prediction_length}" \
#               f"_train_length={train_length}_test_length={test_length}_BS={batch_size}_NBpE={num_batches_per_epoch}_epochs={epochs}"
#mkdir_p(dir_path)
#pickle_filename = dir_path + f"/values.p"
#pickle.dump(ret_dict, open(pickle_filename, 'wb'))

########################################################################
################## Plotting the open data  #############################
########################################################################
vt = testing_target_vals
vp = testing_pred_vals
#dates_axis = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in df.index[train_length:test_length -train_length]]
new_dates = [pd.Timestamp(testing_dates[i].astype(dtype='datetime64[D]')).strftime('%Y-%m-%d') for i in range(0,testing_dates.shape[0])]
new_dates_plot = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in testing_dates]
close_vals = df_close[new_dates].to_numpy().reshape((-1,vt.shape[0])).reshape((-1,))
open_vp = np.exp(vp) * close_vals
open_vt  =np.exp(vt) * close_vals
fig, ax = plt.subplots(figsize = (12,8))
plt.plot_date(new_dates_plot, open_vt, fmt ='-b')
plt.plot_date(new_dates_plot, open_vp, fmt = '-r')


plt.legend(['Target', 'Regular Prediction'])
plt.title('Open prices for test set')
plt.xlabel('Date')
plt.ylabel('Open Price')


ax.xaxis.set_major_locator(mdates.DayLocator(interval=21))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate()
start_filename =f"./plots/{company}/open_price/"
end_filename = f"twitter={twitter}_type={plottype}_Student_T={student_t}_prediction_length={prediction_length}" \
               f"_train_length={train_length}_test_length={test_length}_BS={batch_size}_NBpE={num_batches_per_epoch}_epochs={epochs}.png"
filename = start_filename + end_filename
if args.save_figs:
    plt.savefig(filename, dpi =400)
if args.show_figs:
    plt.show()
else:
    plt.clf()

