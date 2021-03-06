import h2o
from h2o.estimators import H2ODeepLearningEstimator

h2o.init()

train = h2o.import_file('dataset/train.csv')
test = h2o.import_file('dataset/test.csv')

# define columns
response_column = 'SalePrice'
training_columns = train.col_names
training_columns.remove(response_column)

#train[training_columns].describe()
#OverallQual, OverallCond, GrLivArea

model = H2ODeepLearningEstimator(nfolds=10, epochs=100, hidden=[500, 500])
model.train(x=training_columns, y=response_column, training_frame=train)

h2o.export_file(frame=model.predict(test_data=test), path='prediction.csv', force=True)

print model.model_performance()
