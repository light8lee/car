import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input1')
parser.add_argument('input2')
parser.add_argument('output')
args = parser.parse_args()

input1 = pd.read_csv('../data/{}.csv'.format(args.input1))
input2 = pd.read_csv('../data/{}.csv'.format(args.input2))

output = pd.merge(input1, input2, left_index=True, right_index=True)
output.to_csv('../data/{}.csv'.format(args.output), index=False)