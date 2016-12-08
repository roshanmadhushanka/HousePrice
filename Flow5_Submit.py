import pandas as pd

test = pd.read_csv('dataset/test.csv')
predict = pd.read_csv('prediction.csv')

df = pd.DataFrame()

df['Id'] = test['Id']
df['SalePrice'] = predict['predict']

df.to_csv('submit.csv', index=False)