import pandas as pd
import h2o

# Initialize server
h2o.init()

# Load training data set from csv
data = h2o.import_file('dataset/train.csv')

# Split data into train and test
train, test = data.split_frame(ratios=[0.8])

print train.nrows
print test.nrows