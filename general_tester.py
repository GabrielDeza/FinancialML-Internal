import argparse

parser = argparse.ArgumentParser(description='Main Script to Run one of the jobs')
parser.add_argument('--models', default='DeepAR', type=str, help='List of What Setting (DeepAR, DeepVAR, GPVAR)')
parser.add_argument('--frequencies', default='Daily', type=str, help='List of What Frequency (Daily, Hourly)')
parser.add_argument('--companies',nargs='+', default='CHTR', type=str, help='List of Company')

parser.add_argument('--train_length', default=[60], type=list, help='Length of training set')
parser.add_argument('--validation_length', default=[70], type=list, help='Length of validation set')
parser.add_argument('--test_length', default=[120], type=list, help='Length of testing set')

parser.add_argument('--prediction_length', default=[5], type=list, help='Prediction Length')
parser.add_argument('--epochs', default=[1], type=list, help='number of epochs')
parser.add_argument('--batch_size', default=[32], type=list, help='Batch size')
parser.add_argument('--nbpe', default=[30], type=list, help='Number of batches per epoch')

parser.add_argument('--seed', default=[6], type=int, help='seed')

parser.add_argument('--adv_dir', default = [1], type= list, help ="Direction of parameter to modify (-1 or +1)")
parser.add_argument('--epsilon', default=[0.5], type=list, help='Percent change in dataset at each iteration')
parser.add_argument('--max_iter', default=[4], type=list, help='number of iterations on the adv dataset algorithm')
parser.add_argument('--parameter', default=['mu'], type=list, help='parameter we want to change. its mu sigma nu for student-t',)

args = parser.parse_args()


companies = ["ADSK","BWA","CAH","CE","CHTR","FANG","XLNX"]
prediction_length = [5]
epochs = [1]
batch_size = [32]
nbpe = [50]
seed = [6]
adv_dir = [1]
epsilon = [0.01]
max_iter = [3]
parameter = ['mu']


################################
################################
################################

frequency = 'Hourly'
train_length = [50]
validation_length = [70]
test_length = [120]
########
model = 'DeepAR'
os.system(f'python submit_jobs.py --models {model} --frequencies {frequency} --companies {companies} --train_length {train_length}'
          f'--validation_length {validation_length} --test_length {test_length} --prediction_length {prediction_length}'
          f'--epochs {epochs} --batch_size {batch_size} --nbpe {nbpe} --seed {seed} --adv_dir {adv_dir} --epsilon {epsilon}'
          f'--max_iter {max_iter} --parameter {parameter}')
model = 'DeepVAR'
os.system(f'python submit_jobs.py --models {model} --frequencies {frequency} --companies {companies} --train_length {train_length}'
          f'--validation_length {validation_length} --test_length {test_length} --prediction_length {prediction_length}'
          f'--epochs {epochs} --batch_size {batch_size} --nbpe {nbpe} --seed {seed} --adv_dir {adv_dir} --epsilon {epsilon}'
          f'--max_iter {max_iter} --parameter {parameter}')
model = 'GPVAR'
os.system(f'python submit_jobs.py --models {model} --frequencies {frequency} --companies {companies} --train_length {train_length}'
          f'--validation_length {validation_length} --test_length {test_length} --prediction_length {prediction_length}'
          f'--epochs {epochs} --batch_size {batch_size} --nbpe {nbpe} --seed {seed} --adv_dir {adv_dir} --epsilon {epsilon}'
          f'--max_iter {max_iter} --parameter {parameter}')
########


