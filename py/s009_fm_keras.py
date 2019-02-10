import sys
import pandas as pd
import numpy as np
import datetime
import glob
import gc
import os
from tqdm import tqdm

HOME = os.path.expanduser('~')

sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from preprocessing import get_ordinal_mapping, get_dummies, outlier
from utils import logger_func
try:
    if not logger:
        logger=logger_func()
except NameError:
    logger=logger_func()

#========================================================================
# Data Args
#========================================================================
key = 'card_id'
target = 'target'
ignore_list = [key, target, 'merchant_id', 'first_active_month']
win_path = f'../features/ffm_winner/*.gz'
start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())

#========================================================================
# Model Args
gpu_count = 1
#  gpu_count = 8
batch_size = 128 * gpu_count
epoch = 5
learning_rate = 0

# Common Args
model_type = 'keras'
seed_list = [1208]
seed = seed_list[0]
#========================================================================


logger.info("Keras Setup Start!!")
# Keras
from sklearn.base import BaseEstimator
from keras.layers import Input, Embedding, Dense,Flatten, Activation, dot, add
from keras.models import Model
from keras.regularizers import l2 as l2_reg
from keras import initializers
if gpu_count>1:
    from keras.utils import multi_gpu_model
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def batch_generator(X,y,batch_size=128,shuffle=True):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = [X[i][batch_ids] for i in range(len(X))]
            y_batch = y[batch_ids]
            yield X_batch,y_batch


def test_batch_generator(X,y,batch_size=128):
    sample_size = X[0].shape[0]
    index_array = np.arange(sample_size)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        X_batch = [X[i][batch_ids] for i in range(len(X))]
        y_batch = y[batch_ids]
        yield X_batch,y_batch


def predict_batch(model,X_t,batch_size=128):
    outcome = []
    for X_batch,y_batch in test_batch_generator(X_t,np.zeros(X_t[0].shape[0]),batch_size=batch_size):
        outcome.append(model.predict(X_batch,batch_size=batch_size))
    outcome = np.concatenate(outcome).ravel()
    return outcome



def build_model(input_len, max_features,K=8,solver='adam',l2=0.0,l2_fm = 0.0):

    inputs = []
    flatten_layers=[]
    columns = range(len(max_features))
    for c in columns:
        inputs_c = Input(shape=(1,), dtype='int32',name = 'input_%s'%c)
        num_c = max_features[c]

        embed_c = Embedding(
                        input_dim=num_c, # 埋め込む特徴の次元
                        output_dim=K, # 何次元に埋め込むか
                        input_length=1,
#                         input_length=1,
                        name = 'embed_%s'%c,
                        W_regularizer=l2_reg(l2_fm)
                        )(inputs_c)


        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    fm_layers = []

    for emb1,emb2 in itertools.combinations(flatten_layers, 2):

#         dot_layer = merge([emb1,emb2], mode='dot', dot_axes=1)
        dot_layer = dot(inputs=[emb1, emb2], axes=1)

        fm_layers.append(dot_layer)


    for c in columns:
        num_c = max_features[c]

        embed_c = Embedding(
                        num_c,
                        1,
                        input_length=1,
#                         input_length=input_len,
                        name = 'linear_%s'%c,
                        W_regularizer=l2_reg(l2)
                        )(inputs[c])

        flatten_c = Flatten()(embed_c)

        fm_layers.append(flatten_c)

#     flatten = merge(fm_layers, mode='sum')
    flatten = add(fm_layers) 
    outputs = Activation('sigmoid',name='outputs')(flatten)

    model = Model(input=inputs, output=outputs)

    if gpu_count>1:
        model = multi_gpu_model(model, gpus=gpu_count) # add

    model.compile(
                optimizer=solver,
                loss= 'binary_crossentropy'
              )

    return model


class KerasFM(BaseEstimator):
    def __init__(self, input_len, max_features=[], K=8, solver='adam', l2=0.0, l2_fm=0.0):
        self.model = build_model(input_len, max_features,K,solver,l2=l2,l2_fm = l2_fm)

    def fit(self, X, y, batch_size=128, nb_epoch=10, shuffle=True, verbose=1, validation_data=None):
        self.model.fit(X,y,batch_size=batch_size,nb_epoch=nb_epoch,shuffle=shuffle,verbose=verbose,validation_data=None)

    def fit_generator(self,X,y,batch_size=128,nb_epoch=10,shuffle=True,verbose=1,validation_data=None,callbacks=None):
        tr_gen = batch_generator(X,y,batch_size=batch_size,shuffle=shuffle)
        if validation_data:
            X_test,y_test = validation_data
            te_gen = batch_generator(X_test,y_test,batch_size=batch_size,shuffle=False)
            nb_val_samples = X_test[-1].shape[0]
        else:
            te_gen = None
            nb_val_samples = None

        self.model.fit_generator(
                tr_gen, 
                samples_per_epoch=X[-1].shape[0], 
                nb_epoch=nb_epoch, 
                verbose=verbose, 
                callbacks=callbacks, 
                validation_data=te_gen, 
                nb_val_samples=nb_val_samples, 
                max_q_size=10
                )

    def predict(self,X,batch_size=128):
        y_preds = predict_batch(self.model,X,batch_size=batch_size)
        return y_preds

logger.info("Keras Setup Complete!!")

#========================================================================
# Data Load
base = utils.read_df_pkl('../input/base*')
win_path_list = glob.glob(win_path)
train_path_list = []
test_path_list = []
for path in win_path_list:
    if path.count('train'):
        train_path_list.append(path)
    elif path.count('test'):
        test_path_list.append(path)

# train_path_list = sorted(train_path_list)[:20]
# test_path_list  = sorted(test_path_list)[:20]
        
base_train = base[~base[target].isnull()].reset_index(drop=True)
base_test = base[base[target].isnull()].reset_index(drop=True)
train_feature_list = utils.parallel_load_data(path_list=train_path_list)
test_feature_list = utils.parallel_load_data(path_list=test_path_list)
train = pd.concat(train_feature_list, axis=1)
train = pd.concat([base_train, train], axis=1)
test = pd.concat(test_feature_list, axis=1)
test = pd.concat([base_test, test], axis=1)
train.set_index(key, inplace=True)
test.set_index(key, inplace=True)

train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)

num_list = [col for col in train.columns if (str(train[col].dtype).count('int') or 
                                             str(train[col].dtype).count('float')) and 
            col != target and not(col.count('amount'))]
y = train[target]

train = train[num_list]
test = test[num_list]

for col in tqdm(num_list):
    train = outlier(df=train, col=col, replace_inner=True)
    test = outlier(df=test, col=col, replace_inner=True)
    
train_test = pd.concat([train[num_list], test[num_list]], axis=0)
scaler = MinMaxScaler()
columns = train_test.columns
train_test = scaler.fit_transform(train_test)
train_test = pd.DataFrame(train_test, columns=columns)
train = train_test.iloc[:len(train), :]
test = train_test.iloc[len(train):, :]
print(train.shape, test.shape)
del train_test
gc.collect()
#========================================================================


cv_list = []

train[target] = y.values
pred_list = np.zeros(len(test))

train[target] = train[target].map(lambda x: 1 if x<-30 else 0)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
y = train[target]
kfold = list(folds.split(train, y.values))
# train.drop('outliers', axis=1, inplace=True)

# Test Set
len_test = len(test)
len_feats = len(num_list)
x_test = test.values.reshape(len_test, len_feats)
x_test = [i for i in x_test.T]

prediction = np.zeros(len_test)
stack_prediction = np.zeros(len(train))

for n_fold, (trn_idx, val_idx) in enumerate(kfold):
    tmp_train, y_train = train[num_list].iloc[trn_idx, :], y.iloc[trn_idx]
    x_val, y_val = train[num_list].iloc[val_idx, :], y.iloc[val_idx]
    len_train = len(tmp_train)
    len_valid = len(x_val)

    x_train = tmp_train.values.reshape(len_train, len_feats)
    y_train = y_train.values.reshape(len_train, 1)
    x_val = x_val.values.reshape(len_valid, len_feats)
    y_val = y_val.values.reshape(len_valid, 1)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    x_train = [i for i in x_train.T]
    x_val = [i for i in x_val.T]
    print(len(x_train),len(x_val), len(x_test))
    
    max_features = [len(tmp_train[col]) for col in num_list]
    model = KerasFM(input_len=len(tmp_train), max_features=max_features)
    
    model.fit(X=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=batch_size, nb_epoch=epoch)
    
    test_pred = model.predict(X=x_test, batch_size=batch_size)
    prediction += test_pred
    
    y_pred = model.predict(X=x_val, batch_size=batch_size)
    stack_prediction[val_idx] = y_pred
    
    sc_score = roc_auc_score(y_val, y_pred)
    logger.info(f'''
    #========================================================================
    # FOLD {n_fold} SCORE: {sc_score}
    #========================================================================''')
    cv_list.append(sc_score)
    
prediction /= len(kfold)
cv_score = np.mean(cv_list)
logger.info(f'''
#========================================================================
# CV SCORE: {cv_score}
#========================================================================''')

train_pred = pd.Series(stack_prediction, name='prediction').to_frame()
test_pred = pd.Series(prediction, name='prediction').to_frame()
train_pred[key] = list(train.index)
test_pred[key] = list(test.index)
df_pred = pd.concat([train_pred, test_pred], axis=0)

utils.to_pkl_gzip(path=f"../stack/{start_time[4:12]}_stack_{model_type}_lr{learning_rate}_{len(num_list)}feats_{len(seed_list)}seed_{batch_size/gpu_count}batch_OUT_CV{str(cv_score).replace('.', '-')}_LB", obj=df_pred)
