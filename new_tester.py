import argparse

parser = argparse.ArgumentParser(description='Main Script to Run one of the jobs')
parser.add_argument('--models', default='DeepAR', type=str, help='List of What Setting (DeepAR, DeepVAR, GPVAR)')
parser.add_argument('--frequencies', default='Daily', type=str, help='List of What Frequency (Daily, Hourly)')
parser.add_argument('--companies',nargs='+', default='CHTR', type=str, help='List of Company')

args = parser.parse_args()
print(args.companies)