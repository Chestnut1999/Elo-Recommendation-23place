from .base_model import Model
import numpy as np
import lightgbm as lgb
import os
HOME = os.path.expanduser('~')
import sys
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
from parallel_utils import parallel_process


class lightgbm_ex(Model):

    def __init__(self, logger, metric, seed=1208, model_type='lgb', ignore_list=[]):
        self.__model = lgb
        self.estimator = None
        self.model_type = model_type
        self.logger = logger
        self.metric = metric
        self.seed = seed
        self.ignore_list = ignore_list
        self.fold_model_list = []
        self.viz_detail = False
        #  self.cv_feim = []
        #  self.prediction = []
        #  self.result_stack = []
        #  self.xray_list = []
        #  self.cv_score = None
        #  self.base_xray = []
        #  self.decoder = None

    def train(self, x_train, y_train, x_val, y_val, params={}, verbose_eval=100, gbdt_args={}):

        # Params
        params['bagging_seed'] = self.seed
        params['data_random_seed'] = self.seed
        params['feature_fraction_seed'] = self.seed
        params['random_seed'] = self.seed
        num_boost_round = gbdt_args['num_boost_round']
        early_stopping_rounds = gbdt_args['early_stopping_rounds']

        # Dataset
        lgb_train = self.__model.Dataset(data=x_train, label=y_train)
        lgb_eval = self.__model.Dataset(data=x_val, label=y_val)

        #  self.logger.info(params)
        estimator = self.__model.train(
            train_set=lgb_train,
            valid_sets=lgb_eval,
            params=params,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
            categorical_feature= []
        )
        return estimator

    def predict(self, X, Y):
        y_pred = self.__model.predict(X)
        return y_pred

    def feature_importance(self):
        return self.__model.feature_importance()
