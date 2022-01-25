#1. Imports
import pandas as pd
import numpy as np
import mxnet as mx
import pickle
import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator, MultivariateEvaluator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.model.gpvar import GPVAREstimator
from gluonts.mx.trainer import Trainer
import argparse
from gluonts.dataset.rolling_dataset import StepStrategy, generate_rolling_dataset
import seaborn as sns
from helper_financial_metrics import *
sns.set(style="darkgrid")

#2. Arguments + Hyper Parameters
##############   PARAMETERS & HYPERPARAMETERS  #########################
########################################################################
parser = argparse.ArgumentParser(description='Regular DeepVAR forecasts')
parser.add_argument('--company', default='ADSK', type=str, help='Company')
parser.add_argument('--train_length', default=800, type=int, help='Length of training set')
parser.add_argument('--validation_length', default=850, type=int, help='Length of validation set')
parser.add_argument('--test_length', default=950, type=int, help='Length of testing set')
parser.add_argument('--prediction_length', default=5, type=int, help='Prediction Length')
parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=30, type=int, help='Batch size')
parser.add_argument('--nbpe', default=200, type=int, help='Number of batches per epoch')
parser.add_argument('--save_figs', default=False, type = bool, help = 'decides if the figures and txt files should be saved or not')
parser.add_argument('--show_figs', default=False, type=bool, help='decides if the figures should be shown to the screen')
parser.add_argument('--seed', default=6, type=int, help='seed')

args = parser.parse_args()
company = args.company
train_length = args.train_length
validation_length = args.validation_length
test_length = args.test_length
prediction_length = args.prediction_length
step_size = prediction_length
epochs = args.epochs
batch_size = args.batch_size
num_batches_per_epoch = args.nbpe

#3. Setting the seed for randomness. Important for when you actually get forecasts or metrics for plotting or the gradient attack.
seed = args.seed
mx.random.seed(seed)
np.random.seed(seed)

#4. Reading in the dataset
columns = ["Open","Close","Retweets", "Replies","Likes", "Volume","positive_score", "negative_score", "neutral_score",
           "positive_percentage","negative_percentage","neutral_percentage","positive_count","negative_count",
           "neutral_count","General_score"]
filename = f"/Users/gabriel/Desktop/Fixed_Stock_Project/CHTR_1d.csv"

df = pd.read_csv(filename, index_col=1)
df = df.drop(['Unnamed: 0'],axis = 1)
df_close = df['Close']
df_open = df['Open']
N = len(columns)
T = df.shape[0]

#5. Convert from a pandas dataframe to a numpy matrix. Had to do it iteratively because the one line solution was scaling my data for some reason.
target = np.zeros((N, T))
for i,col in enumerate(columns):
    target[i,:] = df[col][:].to_numpy()

#6. Creating the Training dataset for training the model. Has several steps

#6 a) making the univariate version of the dataset (training and validation):
time_series_dicts = []
for time_series in target:
    time_series_dicts.append({"target": time_series[:train_length], "start": df.index[0]}) #N items, each have dimensions (train_length,)
train_dataset = ListDataset(time_series_dicts, freq="1B")

val_time_series_dicts = []
for time_series in target:
    val_time_series_dicts.append({"target": time_series[:validation_length], "start": df.index[0]}) #N items, each have dimensions (train_length,)
validation_dataset = ListDataset(val_time_series_dicts, freq="1B")

#6 b) Intialize some multivariate groupers:
grouper_train = MultivariateGrouper(max_target_dim=N)
grouper_validation = MultivariateGrouper(max_target_dim=N)
grouper_interim_train = MultivariateGrouper(max_target_dim=N, num_test_dates=1) #the "num_test_dates=1" is so important it's ridiclious... 5 hours down the drain :(

#6 c) #Create the multivariate dataset
train_ds = grouper_train(train_dataset) #1 item, dimension is (N,train_length)
validation_ds = grouper_validation(validation_dataset)

#7 Define the model of interest

estimator = DeepVAREstimator(target_dim=N,
                             prediction_length=prediction_length,
                             freq="1B",
                             trainer=Trainer(epochs=epochs,num_batches_per_epoch=num_batches_per_epoch,learning_rate=1e-3,),)

#8 Train the model with the multivariate dataset
predictor = estimator.train(training_data = train_ds,validation_data = validation_ds, num_workers = 0)

#9 Create a rolling version of the training dataset. Notice we pass in the univariate version because generate_rolling_dataset does not currently support multivariate version
training_dataset_rolled = generate_rolling_dataset(dataset = train_dataset,
                                                   start_time=pd.Timestamp(df.index[0], freq="1B"),
                                                   end_time=pd.Timestamp(df.index[train_length],freq ="1B"),
                                                   strategy=StepStrategy(prediction_length=prediction_length, step_size = prediction_length))

#10 Since in 9 we pass in the univariate version, we have to fix that
training_sets = [[] for i in range(train_length//prediction_length)] #ie: how many rolling sets we have
for i,train_dict in enumerate(training_dataset_rolled):
    training_sets[i%(train_length//prediction_length)].append({"target": train_dict['target'], "start": pd.Timestamp(df.index[0], freq="1B")})

#11 We can now plot the rolling version of the training dataset:

#a) Plot the Open Price
prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()
for j,ts in enumerate(training_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
            to_pandas(ts[0]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
            if j != 0: #skip the last forecast because it's not in the training set
                forecast.copy_dim(0).plot(color='m', prediction_intervals=prediction_intervals)
plt.grid(which="both")
plt.title(f"Resulting Forecasts on the Training Set")
plt.ylabel("Open Price")
legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Date")
plt.legend(legend, loc="upper left")
plt.show()

#b) Plot the Close Price
prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()
for j,ts in enumerate(training_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
            to_pandas(ts[1]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
            if j != 0: #skip the last forecast because it's not in the training set
                forecast.copy_dim(1).plot(color='m', prediction_intervals=prediction_intervals)
plt.grid(which="both")
plt.title(f"Resulting Forecasts on the Training Set")
plt.ylabel("Close Price")
legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Date")
plt.legend(legend, loc="upper left")
plt.show()

#12 Now we shall get the metrics of the rolling dataset. Similiarily, cause 9 is with the univariate dataset,
# we will take the metric for each individual rolling set and then we will average each of those metrics over the number of rolling periods.

#Btw, ind_metrics is a (N,11 + 2* number of quantiles) dataframe.
quantiles = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
rolling_training_stats = {'Open':{'RMSE':0,'CRPS':0,'MAPE':0},
                          'Close':{'RMSE':0,'CRPS':0,'MAPE':0}}
for j,ts in enumerate(training_sets): #iterating over the rolling sets

        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        agg_metrics, interim_ind_metrics = backtest_metrics(test_dataset=interim_ds,predictor=predictor,evaluator=MultivariateEvaluator(quantiles=quantiles))
        for num,price_type in enumerate(['Open','Close']):
            rolling_training_stats[price_type]['RMSE'] += np.sqrt(interim_ind_metrics['MSE'].values[num])
            rolling_training_stats[price_type]['MAPE'] += interim_ind_metrics['MAPE'].values[num]
            crps = []
            for q in quantiles:
                crps.append(interim_ind_metrics[f'QuantileLoss[{q}]'].values[0]/interim_ind_metrics['abs_target_sum'].values[num])
            rolling_training_stats[price_type]['CRPS'] += sum(crps)/len(crps)
for price_type in ['Open','Close']:
    rolling_training_stats[price_type]['RMSE'] = rolling_training_stats[price_type]['RMSE']/len(training_sets)
    rolling_training_stats[price_type]['CRPS'] = rolling_training_stats[price_type]['CRPS']/len(training_sets)
    rolling_training_stats[price_type]['MAPE'] = rolling_training_stats[price_type]['MAPE']/len(training_sets)
#print(f'>>> final MSE ={final_mse}')

#13 Now we will get the actual point forecasts from monte carlo sampling as well as the corresponding target log returns.
# This will be used for computation of binary accuracy and future plotting

training_actual_open = np.array([])
training_pred_open = np.array([])
training_actual_close = np.array([])
training_pred_close = np.array([])
training_dates = np.array([], dtype = np.datetime64)
for j,ts in enumerate(training_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        forecast_it, ts_it = make_evaluation_predictions(interim_ds, predictor=predictor, num_samples=100)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecasts = list(forecast_it)
        tss = list(ts_it)
        for i, (ground_truth, forecast) in enumerate(zip(tss, forecasts)):
            training_dates = np.concatenate((ground_truth.index[-prediction_length:], training_dates))
            training_actual_open = np.concatenate((ground_truth.values[-prediction_length:, 0].reshape((-1,)), training_actual_open))
            training_pred_open = np.concatenate((np.mean(forecast.samples, axis=0)[:, 0].reshape((-1,)), training_pred_open))

            training_actual_close = np.concatenate((ground_truth.values[-prediction_length:, 1].reshape((-1,)), training_actual_close))
            training_pred_close = np.concatenate((np.mean(forecast.samples, axis=0)[:, 1].reshape((-1,)), training_pred_close))

#for the predictions, remove the first 'prediction_length' (more like context length but they are equal)
correct_training_actual_open = training_actual_open[prediction_length:]
correct_training_pred_open = training_pred_open[prediction_length:]
correct_training_actual_close = training_actual_close[prediction_length:]
correct_training_pred_close = training_pred_close[prediction_length:]
correct_training_dates = training_dates[prediction_length:]


#14 we will now make the testing set

#14 a) Make the univariate dataset
time_series_dict = []
for time_series in target:
    time_series_dicts.append({"target": time_series[validation_length:test_length], "start": df.index[validation_length]}) #N items, each are of dimension (test_length - train_length,)
test_dataset = ListDataset(time_series_dicts, freq="1B")

#14 b) initialize multivariate groupers:

grouper_test = MultivariateGrouper(max_target_dim=N)
grouper_interim_test = MultivariateGrouper(max_target_dim=N, num_test_dates=1)

# 14c) create the multivariate version of the dataset
test_ds = grouper_test(test_dataset) #1 item, dimension is (N,test_length -validation_length)

# 14 d) create the rolling dataset but from the univariate version
testing_dataset_rolled = generate_rolling_dataset(dataset = test_dataset,
                                          start_time = pd.Timestamp(df.index[validation_length], freq="1B"),
                                          end_time = pd.Timestamp(df.index[test_length], freq="1B"),
                                            strategy=StepStrategy(prediction_length=prediction_length, step_size = prediction_length))

#15 Similiar to step 10, we will hand make the multivariate version of the rolling testing set

testing_sets = [[] for i in range((test_length-validation_length)//prediction_length)] #ie: how many rolling sets we have
for i,train_dict in enumerate(testing_dataset_rolled):
    testing_sets[i%((test_length-validation_length)//prediction_length)].append({"target": train_dict['target'], "start": pd.Timestamp(df.index[validation_length], freq="1B")})

#16 Let's plot the testing set

#16 a) plot the open price
prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
            to_pandas(ts[0]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
            if j != 0:
                forecast.copy_dim(0).plot(color='g', prediction_intervals=prediction_intervals)
plt.grid(which="both")
plt.title(f"Resulting Forecasts on the Testing Set")
plt.ylabel("Open Price")
legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Date")
plt.legend(legend, loc="upper left")
plt.show()


#16 a) plot the open price
prediction_intervals=[50.0, 90.0]
fig, ax = plt.subplots()
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_train(interim_dict) #is (N,truncated length)
        for i, (test_entry, forecast) in enumerate(zip(interim_ds, predictor.predict(interim_ds))):
            to_pandas(ts[1]).plot(linewidth=1, color='b', zorder=0, figsize=(13, 7))
            if j != 0:
                forecast.copy_dim(1).plot(color='g', prediction_intervals=prediction_intervals)
plt.grid(which="both")
plt.title(f"Resulting Forecasts on the Testing Set")
plt.ylabel("Close Price")
legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
plt.xlabel("Date")
plt.legend(legend, loc="upper left")
plt.show()

#20 Let's now calculate the metrics over the rolling test set. For similiar reasons as the training set, we do the following:
quantiles=(0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9)
rolling_testing_stats = {'Open':{'RMSE':0,'CRPS':0,'MAPE':0},
                          'Close':{'RMSE':0,'CRPS':0,'MAPE':0}}
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_test(interim_dict) # (N,truncated length)
        agg_metrics,interim_ind_metrics = backtest_metrics(test_dataset=interim_ds,predictor=predictor,evaluator=MultivariateEvaluator(quantiles=quantiles))
        for num, price_type in enumerate(['Open', 'Close']):
            rolling_testing_stats[price_type]['RMSE'] += np.sqrt(interim_ind_metrics['MSE'].values[0])
            rolling_testing_stats[price_type]['MAPE'] += interim_ind_metrics['MAPE'].values[0]
            crps = []
            for q in quantiles:
                crps.append(interim_ind_metrics[f'QuantileLoss[{q}]'].values[0] / interim_ind_metrics['abs_target_sum'].values[0])
            rolling_testing_stats[price_type]['CRPS'] += sum(crps) / len(crps)
for price_type in ['Open', 'Close']:
    rolling_testing_stats[price_type]['RMSE'] = rolling_testing_stats[price_type]['RMSE']/len(training_sets)
    rolling_testing_stats[price_type]['CRPS'] = rolling_testing_stats[price_type]['CRPS']/len(training_sets)
    rolling_testing_stats[price_type]['MAPE'] = rolling_testing_stats[price_type]['MAPE']/len(training_sets)

#21 Calculate actual targets and predictions:

testing_actual_open = np.array([])
testing_pred_open = np.array([])
testing_actual_close = np.array([])
testing_pred_close = np.array([])

testing_dates = np.array([], dtype = np.datetime64)
for j,ts in enumerate(testing_sets): #iterating over the rolling sets
        interim_dict = ListDataset(ts, freq="1B") #N times (1,truncated length)
        interim_ds = grouper_interim_test(interim_dict) # (N,truncated length)
        forecast_it, ts_it = make_evaluation_predictions(interim_ds, predictor=predictor, num_samples=100)
        mx.random.seed(seed)
        np.random.seed(seed)
        forecasts = list(forecast_it)
        tss = list(ts_it)
        for i, (target, forecast) in enumerate(zip(tss, forecasts)):
            testing_dates = np.concatenate((target.index[-5:], testing_dates))

            testing_actual_open = np.concatenate((target.values[-5:, 0].reshape((-1,)), testing_actual_open))
            testing_pred_open = np.concatenate((np.mean(forecast.samples, axis=0)[:, 0].reshape((-1,)), testing_pred_open))

            testing_actual_close = np.concatenate((target.values[-5:, 0].reshape((-1,)), testing_actual_close))
            testing_pred_close = np.concatenate((np.mean(forecast.samples, axis=0)[:, 0].reshape((-1,)), testing_pred_close))



correct_testing_actual_open = testing_actual_open[5:]
correct_testing_pred_open = testing_pred_open[5:]
correct_testing_actual_close = testing_actual_close[5:]
correct_testing_pred_close = testing_pred_close[5:]
correct_testing_dates = testing_dates[5:]


#22 We are now at the stage that we want to calculate additional metrics.

#22) a) Binary Accuracy:
#Depending if its the log difference or the difference, we have to undo the log


training_target_diff = correct_training_actual_close - correct_training_actual_open
training_pred_diff = correct_training_pred_close - correct_training_pred_open

testing_target_diff = correct_testing_actual_close - correct_testing_actual_open
testing_pred_diff = correct_testing_pred_close - correct_testing_pred_open

train_bias_acc_up = (np.sum(training_target_diff >= 0, axis=0)) / training_target_diff.shape[0]
train_bias_acc_down =  (np.sum(training_target_diff < 0, axis=0)) / training_target_diff.shape[0]
train_acc = np.sum(np.sign(training_target_diff) == np.sign(training_pred_diff)) / training_target_diff.shape[0]

test_bias_acc_up = (np.sum(testing_target_diff >= 0, axis=0)) / testing_target_diff.shape[0]
test_bias_acc_down = (np.sum(testing_target_diff < 0, axis=0)) / testing_target_diff.shape[0]
test_acc = np.sum(np.sign(testing_target_diff) == np.sign(testing_pred_diff)) / testing_target_diff.shape[0]

# 22 b) Financial Metrics

#passive gain
passive_gain_train = 100*(df_open[pd.Timestamp(correct_training_dates[-1]).strftime('%Y-%m-%d')] - df_open[pd.Timestamp(correct_training_dates[0]).strftime('%Y-%m-%d')])/df_open[pd.Timestamp(correct_training_dates[0]).strftime('%Y-%m-%d')]
passive_gain_test = 100*(df_open[pd.Timestamp(correct_testing_dates[-1]).strftime('%Y-%m-%d')] - df_open[pd.Timestamp(correct_testing_dates[0]).strftime('%Y-%m-%d')])/df_open[pd.Timestamp(correct_testing_dates[0]).strftime('%Y-%m-%d')]

print(correct_training_dates.shape)
print(training_pred_diff.shape)
print(training_target_diff.shape)

#greedy gain
greedy_gain_train,train_invest,training_greedy_returns,training_greedy_PDT = greedy_gain(correct_training_dates, training_pred_diff,training_target_diff,df_open,df_close)
greedy_gain_test, test_invest,testing_greedy_returns,testing_greedy_PDT= greedy_gain(correct_testing_dates, testing_pred_diff, testing_target_diff,df_open,df_close)

#threshold gain
threshold_gain_train,training_threshold_returns,training_threshold_PDT = threshold_gain(correct_training_dates, training_pred_diff,training_target_diff,df_open,df_close)
threshold_gain_test,testing_threshold_returns,testing_threshold_PDT = threshold_gain(correct_testing_dates, testing_pred_diff,testing_target_diff,df_open,df_close)

m = {0:{'RMSE':0,'CRPS':0,'MAPE':0},1:{'RMSE':0,'CRPS':0,'MAPE':0}}
t = rolling_training_stats
v = rolling_testing_stats
for i,set_type in enumerate([t,v]):
    for metric in ['RMSE','MAPE','CRPS']:
        m[i][metric] = (set_type['Open'][metric] + set_type['Close'][metric])/2

# 23 Printing the various metrics:

print(f"Training: RMSE: {round(m[0]['RMSE'],3)} MAPE: {round(m[0]['MAPE'],3)} CRPS: {round(m[0]['CRPS'],3)}"
      f" Only Up: {round(train_bias_acc_up,3)} Only Down: {round(train_bias_acc_down,3)} Accuracy: {round(train_acc,3)}"
      f" Initial Investment: {round(train_invest,2)}$ Passive Return: {round(passive_gain_train,1)}% Greedy Return: {round(greedy_gain_train,1)}%"
      f" Threshold Return: {round(threshold_gain_train,1)}%")
print(f"Testing: RMSE: {round(m[1]['RMSE'],3)} MAPE: {round(m[1]['MAPE'],3)} CRPS: {round(m[1]['CRPS'],3)}"
      f" Only Up: {round(test_bias_acc_up,3)} Only Down: {round(test_bias_acc_down,3)} Accuracy: {round(test_acc,3)}"
      f" Initial Investment: {round(test_invest,2)}$ Passive Return: {round(passive_gain_test,1)}% Greedy Return: {round(greedy_gain_test,1)}%"
      f" Threshold Return: {round(threshold_gain_test,1)}%")


ret_dict ={}
training_metrics = {}
testing_metrics = {}

for feat in ['RMSE','MAPE','CRPS']:
    training_metrics[feat] = m[0][feat]
    testing_metrics[feat] = m[1][feat]

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

