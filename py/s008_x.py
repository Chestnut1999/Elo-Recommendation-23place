import xlearn as xl
# Training task
ffm_model = xl.create_ffm()  # Use field-aware factorization machine
ffm_model.setTrain("../input/ffm_train.txt")   # Training data
# ffm_model.setTrain("../input/titanic_train.txt")   # Training data
#  ffm_model.setValidate("../input/titanic_test.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate : 0.2
#  2. regular lambda : 0.002
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}

# Train model
ffm_model.fit(param, "../model.out")
