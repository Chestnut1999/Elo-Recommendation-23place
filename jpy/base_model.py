parallel=True
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, r2_score, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, GroupKFold, KFold
import multiprocessing
import shutil
from copy import copy
import gc
from scipy.stats.mstats import mquantiles
from tqdm import tqdm
import sys
import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
from parallel_utils import parallel_process
from calculate_utils import round_size
from preprocessing import factorize_categoricals, get_dummies, ordinal_encode, get_ordinal_mapping
import category_encoders as ce

kaggle = 'home-credit-default-risk'

class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    #  @abstractmethod
    def predict_proba(self):
        pass

    #  @abstractmethod
    def accuracy(self):
        pass

    #  @abstractmethod
    def cross_val_score(self):
        pass

    def sc_metrics(self, y_test, y_pred):
        try:
            if self.metric.count('logloss'):
                score = log_loss(y_test, y_pred)
            elif self.metric == 'auc':
                score = roc_auc_score(y_test, y_pred)
            elif self.metric=='l2':
                score = r2_score(y_test, y_pred)
            elif self.metric=='rmse':
                score = np.sqrt(mean_squared_error(y_test, y_pred))
            elif self.metric=='mse':
                score = np.sqrt(mean_squared_error(y_test, y_pred))
            elif self.metric=='accuracy':
                y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする
                score = sum(y_test == y_pred_max) / len(y_test)
            else:
                print('SCORE CALICULATION ERROR!')
        except ValueError:
            self.logger.info(f"""
# ==============================
# WARNING!!!!!
# {self.target} is True Only.
# y_test Unique: {np.unique(y_test)}
# y_pred Unique: {np.unique(y_pred)}
# ==============================
            """)
        return score

    def sc_confusion_matrix(self, y_test, y_pred):
        #========================================================================
        # F1Scoreを最大化するポイントで混同行列を算出する
        #========================================================================
        if self.objective=='binary':
            threshold = 0.5
            binary_method = lambda x: 1 if x>=threshold else 0
            def to_binary_f1():
                bi_test = list(map(binary_method, y_test))
                bi_pred = list(map(binary_method, y_pred))
                f1 = f1_score(bi_test, bi_pred)
                return f1

            best_f1 = to_binary_f1()
            best_threshold = threshold
            tmp = copy(best_threshold)
            meta1 = 0
            meta2 = 1

            # 二分探索でF1 Scoreを最大化する閾値を探る
            while True:
                threshold = (tmp + meta1) / 2
                f1 = to_binary_f1()
                print(f"Best : {best_f1} | F1 : {f1}")
                print(f"Threshold : {threshold} | Tmp : {tmp} | Meta1 : {meta1} | Meta2 : {meta2}")
                tmp_f1 = copy(f1) # 1つ目の閾値におけるF1を保存しておく
                if f1>best_f1:
                    #========================================================================
                    # meta1を使い算出した閾値でBest F1を更新したら, 下記を更新して再ループ
                    # meta1: stay
                    # meta2: 元のtmp
                    # tmp  : Best F1を更新したthreshold
                    #========================================================================
                    best_f1 = f1
                    best_threshold = threshold
                    meta2 = copy(tmp)
                    tmp = copy(best_threshold)
                elif f1<best_f1:
                    threshold = (tmp + meta2) / 2
                    f1 = to_binary_f1()
                    if f1>best_f1:
                        #========================================================================
                        # meta2を使い算出したthresholdでBest F1を更新したら, 下記を更新して再ループ
                        # meta1: 元のtmp
                        # meta2: stay
                        # tmp  : Best F1を更新したthreshold
                        #========================================================================
                        best_f1 = f1
                        best_threshold = threshold
                        meta1 = copy(tmp)
                        tmp = copy(best_threshold)
                    elif f1<best_f1:
                        #========================================================================
                        # thresholdを更新してもF1が向上しなかった分岐
                        # 新しい作成した2つのthresholdを比較して、F1が向上する可能性がある分岐を決める
                        # meta1: stay or 元tmp
                        # meta2: stay or 元tmp
                        # tmp  : BestF1のthreshold
                        #========================================================================
                        if tmp_f1>=f1:
                            meta2 = copy(tmp)
                            tmp = copy(best_threshold)
                        elif tmp_f1<f1:
                            meta1 = copy(tmp)
                            tmp = copy(best_threshold)
                    elif f1==best_f1:
                        # 新しい閾値でF1が変わらなかったループ終了
                        break
                elif f1==best_f1:
                    # 新しい閾値でF1が変わらなかったループ終了
                    break


            if self.objective=='multiclass':
                y_pred = np.argmax(self.prediction, axis=1)  # 最尤と判断したクラスの値にする
                self.accuracy = sum(y_test == y_pred_max) / len(y_test)

            else:
                y_pred = list(map(binary_method, y_pred))
                tp, fn, fp, tn = confusion_matrix(y_test, y_pred).ravel()
                self.cmx = np.array([tp, fn, fp, tn])
                self.f1 = f1
                self.accuracy = accuracy_score(y_test, y_pred)
                self.true = accuracy_score(y_test, y_pred, normalize=False)
                self.best_threshold = best_threshold


    def auc(self, test_features, test_target):
        return roc_auc_score(test_target, self.predict(test_features))

    def feature_impact(self):
        pass


    def df_feature_importance(self, feim_name):
        ' Feature Importance '
        if self.model_type.count('lgb'):
            tmp_feim = pd.Series(self.estimator.feature_importance(importance_type='gain'), name=feim_name)
            feature_name = pd.Series(self.use_cols, name='feature')
            feim = pd.concat([feature_name, tmp_feim], axis=1)
        elif self.model_type.count('xgb'):
            tmp_feim = self.estimator.get_fscore()
            feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})
        elif self.model_type.count('ext'):
            tmp_feim = self.estimator.feature_importance_(importance_type='gain')
            feim = pd.Series(tmp_feim,  name=feim_name).to_frame().reset_index().rename(columns={'index':'feature'})
        return feim


    def move_feature(self, feature_name, move_path='../features/7_escape'):

        try:
            shutil.move(f'../features/4_winner/{feature_name}.gz', move_path)
        except FileNotFoundError:
            print(f'FileNotFound. : {feature_name}.gz')
            pass


    def data_check(self, train=[], test=[], target='', encode='', exclude_category=False):
        '''
        Explain:
            学習を行う前にデータに問題がないかチェックする
            カテゴリカルなデータが入っていたらエンコーディング or Dropする
        Args:
        Return:
        '''

        if len(test):
            df = pd.concat([train, test], axis=0)
        else:
            df = train

        try:
            #  categorical_list = [col for col in list(df.columns) if (df[col].dtype == 'object') and col not in self.ignore_list]
            categorical_list = []
            for col in df.columns:
                if (df[col].dtype == 'object') and col not in self.ignore_list:
                    categorical_list.append(col)
        except AttributeError:
            print(f"Duplicate Column: {col}")
            sys.exit()
        dt_list = [col for col in list(df.columns) if str(df[col].dtype).count('time') and col not in self.ignore_list]
        self.logger.info(f'''
#==============================================================================
# DATA CHECK START
# CATEGORICAL FEATURE: {categorical_list}
# DATETIME FEATURE   : {dt_list}
# CAT ENCODE         : {encode}
# ignore_list        : {self.ignore_list}
#==============================================================================
        ''')

        if encode=='label':
            df = factorize_categoricals(df, categorical_list)
        elif encode=='dummie':
            df = get_dummies(df, categorical_list)
        elif encode=='ordinal':
            df, decoder = ordinal_encode(df, categorical_list)
            self.decoder = decoder

        if len(test):
            train = df[~df[target].isnull()]
            test = df[df[target].isnull()]
        else:
            train = df
        ' Testsetで値のユニーク数が1のカラムを除外する '
        drop_list = dt_list
        if len(test):
            for col in test.columns:
                length = test[col].nunique()
                if length <=1 and col not in self.ignore_list and col!=target:
                    self.logger.info(f'''
***********WARNING************* LENGTH {length} COLUMN: {col}''')
                    self.move_feature(feature_name=col)
                    if col not in self.ignore_list:
                        drop_list.append(col)

        self.logger.info(f'''
#==============================================================================
# DATA CHECK END
# SHAPE: {df.shape}
#==============================================================================''')

        return train, test, drop_list


    def cross_validation(self, train, key, target, fold_type='stratified', fold=5, group_col_name='', params={}, num_boost_round=0, early_stopping_rounds=0, self_kfold=False, self_stop=[], params_tune=False):

        # Result Variables
        list_score = []
        self.cv_feim = pd.DataFrame([])

        # Y Setting
        y = train[target]

        ' KFold '
        if fold_type == 'stratified':
            folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=self.seed)  # 1
            kfold = folds.split(train, y)
        elif fold_type == 'group':
            if group_col_name == '':
                raise ValueError(f'Not exist group_col_name.')
            folds = GroupKFold(n_splits=fold)
            kfold = folds.split(train, y, groups=train[group_col_name].values)
        elif fold_type == 'kfold':
            folds = KFold(n_splits=fold, shuffle=True, random_state=self.seed)  # 1
            kfold = folds.split(train, y)
        elif fold_type == 'self':
            kfold = self_kfold

        use_cols = [f for f in train.columns if f not in self.ignore_list]
        self.use_cols = sorted(use_cols)  # カラム名をソートし、カラム順による学習への影響をなくす
        self.kfold = list(kfold)

        if key in train.columns:
            train.set_index(key, inplace=True)

        pred_val = train[target].copy()

        for n_fold, (trn_idx, val_idx) in enumerate(self.kfold):

            x_train, y_train = train[self.use_cols].iloc[trn_idx, :], y.iloc[trn_idx].values
            x_val, y_val = train[self.use_cols].iloc[val_idx, :], y.iloc[val_idx].values

            # GBDTのみ適用するargs
            gbdt_args = {}
            if num_boost_round:
                gbdt_args['num_boost_round'] = num_boost_round
                gbdt_args['early_stopping_rounds'] = early_stopping_rounds
            self.estimator = self.train(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                params=params,
                gbdt_args=gbdt_args
            )
            y_pred = self.estimator.predict(x_val)
            pred_val.iloc[val_idx] = y_pred

            self.fold_model_list.append(self.estimator)

            sc_score = self.sc_metrics(y_val, y_pred)

            stop_cnt = 0
            stop_score_list = []
            if len(self_stop)>0:
                if sc_score > self_stop[n_fold]:
                    stop_cnt+=1
                    stop_score_list.append(sc_score)
                if stop_cnt==3:
                    self.cv_score = np.mean(stop_score_list)
                    return self

            list_score.append(sc_score)
            self.logger.info(f'Fold No: {n_fold} | {self.metric}: {sc_score}')

            ' Feature Importance '
            feim_name = f'{n_fold}_importance'
            feim = self.df_feature_importance(feim_name=feim_name)

            if len(self.cv_feim) == 0:
                self.cv_feim = feim.copy()
            else:
                self.cv_feim = self.cv_feim.merge(feim, on='feature', how='inner')

        self.cv_score = np.mean(list_score)
        self.val_score_list = list_score
        self.prediction = pred_val.values

        self.logger.info(f'''
#========================================================================
# Train End.''')
        [self.logger.info(f'''
# Validation No: {i} | {self.metric}: {score}''') for i, score in enumerate(list_score)]
        self.logger.info(f'''
# Params   : {params}
# CV score : {self.cv_score}
#======================================================================== ''')

        # Parameter Tuningの場合はscoreがあればOK
        if params_tune:
            return self

        importance = []
        for fold_no in range(fold):
            if len(importance) == 0:
                importance = self.cv_feim[f'{fold_no}_importance'].values.copy()
            else:
                importance += self.cv_feim[f'{fold_no}_importance'].values

        self.cv_feim['avg_importance'] = importance / fold
        self.cv_feim.sort_values(by=f'avg_importance',
                            ascending=False, inplace=True)
        self.cv_feim['rank'] = np.arange(len(self.cv_feim))+1

        return self


    def cross_prediction(self, train, test, key, target, fold_type='stratified', fold=5, group_col_name='', params={}, num_boost_round=0, early_stopping_rounds=0, oof_flg=True, self_kfold=False, self_stop=[], comp_name='', scaler=False, self_predict=[]):

        if len(self_predict):
            df_base_id = pd.concat([self_predict[key].to_frame(), test.reset_index()[key].to_frame()], axis=0)
        else:
            df_base_id = pd.concat([train.reset_index()[key].to_frame(), test.reset_index()[key].to_frame()], axis=0)
        self.target = target
        list_score = []
        best_iter_list = []
        self.fold_pred_list = []
        self.fold_val_list = []
        self.cv_feim = pd.DataFrame([])
        self.prediction = np.array([])
        val_stack = pd.DataFrame()

        self.objective = params['objective']
        if self.objective=='multiclass':
            self.val_pred = np.zeros((len(train), 13))
        else:
            self.val_pred = np.zeros(len(train))

        # Y Setting
        y = train[target].copy()
        #  if self.objective.count('reg'):
        #      y_min = y.min()
        #      y -= y_min
        #      y = np.log1p(y)

        ' KFold '
        if fold_type == 'stratified':
            folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=self.seed)  # 1
            kfold = folds.split(train, y)
        elif fold_type == 'group':
            if group_col_name == '':
                raise ValueError(f'Not exist group_col_name.')
            folds = GroupKFold(n_splits=fold)
            kfold = folds.split(train, y, groups=train[group_col_name].values)
        elif fold_type == 'kfold':
            folds = KFold(n_splits=fold, shuffle=True, random_state=self.seed)  # 1
            kfold = folds.split(train, y)
        elif fold_type == 'self':
            kfold = self_kfold

        use_cols = [f for f in train.columns if f not in self.ignore_list]
        self.use_cols = sorted(use_cols)  # カラム名をソートし、カラム順による学習への影響をなくす

        if key in train.columns:
            train.set_index(key, inplace=True)
        if key in test.columns:
            test.set_index(key, inplace=True)

        #  if sys.argv[4]=='ods':
        if True:
            self.kfold = zip(*kfold)
        else:
            self.kfold = list(kfold)
        #  for n_fold, (trn_idx, val_idx) in enumerate(zip(*kfold)):

        result_list = []

        for n_fold, (trn_idx, val_idx) in enumerate(self.kfold):

            # card_id split
            #  if sys.argv[4]=='ods' or sys.argv[5]=='no_out':
            if True:
                x_train, y_train = train.loc[train.index.isin(trn_idx), :][use_cols], y.loc[train.index.isin(trn_idx)].values
                x_val, y_val = train.loc[train.index.isin(val_idx), :][use_cols], y.loc[train.index.isin(val_idx)].values
            else:
                x_train, y_train = train[self.use_cols].iloc[trn_idx, :], y.iloc[trn_idx].values
                x_val, y_val = train[self.use_cols].iloc[val_idx, :], y.iloc[val_idx].values
            print(x_train.shape, x_val.shape, len(val_idx))
            print(y_train.max(), y_train.min())

            if n_fold == 0:
                if params['objective']=='lambdarank':
                    test_cols = self.use_cols
                    test_cols.remove('rank')
                    x_test = test[test_cols]
                else:
                    x_test = test[self.use_cols]

            # GBDTのみ適用するargs
            gbdt_args = {}
            if num_boost_round:
                gbdt_args['num_boost_round'] = num_boost_round
                gbdt_args['early_stopping_rounds'] = early_stopping_rounds

            self.estimator = self.train(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                params=params,
                gbdt_args=gbdt_args
            )

            # Tmp Result
            y_pred = self.estimator.predict(x_val)
            best_iter_list.append(self.estimator.best_iteration)


            if len(self_predict):
                #========================================================================
                # StackしたいOOFの作成
                train_id_list = list(x_train.index)
                self_valid = self_predict[~self_predict[key].isin(train_id_list)]
                y_valid = self_valid[target].values
                x_valid = self_valid[self.use_cols]
                y_pred = self.estimator.predict(x_valid)
                self_valid[f'pred_{n_fold}'] = y_pred
                self_valid = self_valid[[key, f'pred_{n_fold}']].set_index(key)
                del x_valid
                gc.collect()
                #========================================================================

                best_iter_list.append(self.estimator.best_iteration)

                self.fold_model_list.append(self.estimator)

                #========================================================================
                # Scoring
                if scaler:
                    y_val = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
                    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                    sc_score = self.sc_metrics(y_val, y_pred)
                else:
                    sc_score = self.sc_metrics(y_valid, y_pred)
                #========================================================================

                list_score.append(sc_score)
                if self.viz_detail:
                    self.logger.info(f'Fold No: {n_fold} | {self.metric}: {sc_score}')

                ' OOF for Stackng '
                if len(val_stack):
                    val_stack = val_stack.join(self_valid.copy())
                else:
                    base_train = self_predict[[key, target]]
                    val_stack = base_train.set_index(key).join(self_valid.copy())

                print(f"Train Valid Stack: {val_stack.shape}")


            else:
                x_val['prediction'] = y_pred
                x_val[target] = y_val
                result_list.append(x_val.reset_index()[[key, 'prediction', target]])

                self.fold_pred_list.append(y_pred)
                self.fold_val_list.append(y_val)
                self.fold_model_list.append(self.estimator)

                if scaler:
                    y_val = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
                    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                    sc_score = self.sc_metrics(y_val, y_pred)
                else:
                    sc_score = self.sc_metrics(y_val, y_pred)

                list_score.append(sc_score)
                if self.viz_detail:
                    self.logger.info(f'Fold No: {n_fold} | {self.metric}: {sc_score}')

            test_pred = self.estimator.predict(x_test)

            if scaler:
                test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1)).ravel()

            # 対数化を解除
            #  if self.objective.count('reg'):
            #      test_pred = np.expm1(test_pred)
            #      test_pred += y_min + 5

            if len(self.prediction) == 0:
                self.prediction = test_pred
            else:
                self.prediction += test_pred

            ' Feature Importance '
            feim_name = f'{n_fold}_importance'
            feim = self.df_feature_importance(feim_name=feim_name)

            if len(self.cv_feim) == 0:
                self.cv_feim = feim.copy()
            else:
                self.cv_feim = self.cv_feim.merge(feim, on='feature', how='inner')

        #========================================================================
        # CV SCORE & F1SCORE
        #========================================================================
        self.val_score_list = list_score
        self.cv_score = np.mean(list_score)
        self.iter_avg = np.mean(best_iter_list)
        if len(self_predict):
            pred_cols = [col for col in val_stack.columns if col.count('pred_')]
            val_stack['prediction'] = val_stack[pred_cols].mean(axis=1)

        if params['objective']=='binary':
            val_stack = pd.concat(result_list, axis=0)
            y_train = val_stack[target].values
            y_allval = val_stack['prediction'].values
            self.train_stack = val_stack
            self.sc_confusion_matrix(y_train, y_allval)


        self.logger.info(f'''
#========================================================================
# Train End.''')
        [self.logger.info(f'''
# Validation No: {i} | {self.metric}: {score}''') for i, score in enumerate(list_score)]
        self.logger.info(f'''
# Params   : {params}
# CV score : {self.cv_score}
#======================================================================== ''')

        if self.objective=='binary':
            self.logger.info(f'''
# Accuracy : {self.accuracy}  {self.true}/{len(test)}
# F1 score : {self.f1}
# TP:{self.cmx[0]}  FP:{self.cmx[2]}
# FN:{self.cmx[1]}  TN:{self.cmx[3]}
#======================================================================== ''')

        ' fold数で平均をとる '
        self.prediction = self.prediction / fold


        ' OOF for Stackng '
        if oof_flg:
            pred_stack = test.reset_index()[[key, target]]
            pred_stack['prediction'] = self.prediction
            result_list.append(pred_stack)

            if len(self_predict):
                val_stack.reset_index(inplace=True)
                result_list.append(val_stack[pred_stack.columns])
            result_stack = pd.concat(result_list, axis=0, ignore_index=True)
            result_stack = df_base_id.merge(result_stack, how='inner', on=key)
            self.logger.info(
                f'result_stack shape: {result_stack.shape} | cnt_id: {len(result_stack[key].drop_duplicates())}')
        else:
            result_stack = []

        importance = []
        for fold_no in range(fold):
            if len(importance) == 0:
                importance = self.cv_feim[f'{fold_no}_importance'].values.copy()
            else:
                importance += self.cv_feim[f'{fold_no}_importance'].values

        self.cv_feim['avg_importance'] = importance / fold
        self.cv_feim.sort_values(by=f'avg_importance',
                            ascending=False, inplace=True)
        self.cv_feim['rank'] = np.arange(len(self.cv_feim))+1

        self.result_stack = result_stack

        return self


    def dist_prediction(self, train, test, key, target, fold_type='stratified', fold=5, group_col_name='', params={}, num_boost_round=0, early_stopping_rounds=0, oof_flg=True, self_kfold=False, self_stop=[], comp_name='', scaler=False, self_predict=[]):

        df_base_id = pd.concat([train.reset_index()[[key, target]], test.reset_index()[key].to_frame()], axis=0)
        self.target = target
        list_score = []
        best_iter_list = []
        self.fold_pred_list = []
        self.fold_val_list = []
        self.cv_feim = pd.DataFrame([])
        self.prediction = np.array([])
        val_stack = pd.DataFrame()

        self.objective = params['objective']
        self.val_pred = np.zeros(len(train))

        # Y Setting
        y = train[target].copy()

        ' KFold '
        if fold_type == 'stratified':
            folds = StratifiedKFold(n_splits=fold, shuffle=True, random_state=self.seed)  # 1
            kfold = folds.split(train, y)
        elif fold_type == 'group':
            if group_col_name == '':
                raise ValueError(f'Not exist group_col_name.')
            folds = GroupKFold(n_splits=fold)
            kfold = folds.split(train, y, groups=train[group_col_name].values)
        elif fold_type == 'kfold':
            folds = KFold(n_splits=fold, shuffle=True, random_state=self.seed)  # 1
            kfold = folds.split(train, y)
        elif fold_type == 'self':
            kfold = self_kfold

        use_cols = [f for f in train.columns if f not in self.ignore_list]
        self.use_cols = sorted(use_cols)  # カラム名をソートし、カラム順による学習への影響をなくす
        self.kfold = list(kfold)

        if key in train.columns:
            train.set_index(key, inplace=True)
        if key in test.columns:
            test.set_index(key, inplace=True)

        for n_fold, (trn_idx, val_idx) in enumerate(self.kfold):

            x_train, y_train = train[self.use_cols].iloc[trn_idx, :], y.iloc[trn_idx].values
            x_val, y_val = train[self.use_cols].iloc[val_idx, :], y.iloc[val_idx].values
            print(x_train.shape, x_val.shape)

            if n_fold == 0:
                x_test = test[self.use_cols]

            # GBDTのみ適用するargs
            gbdt_args = {}
            if num_boost_round:
                gbdt_args['num_boost_round'] = num_boost_round
                gbdt_args['early_stopping_rounds'] = early_stopping_rounds

            self.estimator = self.train(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                params=params,
                gbdt_args=gbdt_args
            )

            #========================================================================
            # StackしたいOOFの作成
            train_id_list = list(x_train.index)
            self_valid = self_predict[~self_predict[key].isin(train_id_list)]
            y_valid = self_valid[target].values
            x_valid = self_valid[self.use_cols]
            y_pred = self.estimator.predict(x_valid)
            self_valid[f'pred_{n_fold}'] = y_pred
            self_valid = self_valid[[key, f'pred_{n_fold}']].set_index(key)
            del x_valid
            gc.collect()
            #========================================================================

            best_iter_list.append(self.estimator.best_iteration)

            self.fold_model_list.append(self.estimator)

            #========================================================================
            # Scoring
            if scaler:
                y_val = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
                y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                sc_score = self.sc_metrics(y_val, y_pred)
            else:
                sc_score = self.sc_metrics(y_valid, y_pred)
            #========================================================================

            list_score.append(sc_score)
            if self.viz_detail:
                self.logger.info(f'Fold No: {n_fold} | {self.metric}: {sc_score}')

            ' OOF for Stackng '
            if len(val_stack):
                val_stack = val_stack.join(self_valid.copy())
            else:
                base_train = self_predict[[key, target]]
                val_stack = base_train.set_index(key).join(self_valid)

            test_pred = self.estimator.predict(x_test)

            if scaler:
                test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1)).ravel()

            # 対数化を解除
            #  if self.objective.count('reg'):
            #      test_pred = np.expm1(test_pred)
            #      test_pred += y_min + 5

            if len(self.prediction) == 0:
                self.prediction = test_pred
            else:
                self.prediction += test_pred

            ' Feature Importance '
            feim_name = f'{n_fold}_importance'
            feim = self.df_feature_importance(feim_name=feim_name)

            if len(self.cv_feim) == 0:
                self.cv_feim = feim.copy()
            else:
                self.cv_feim = self.cv_feim.merge(feim, on='feature', how='inner')

        #========================================================================
        # Result Viz
        # CV SCORE & F1SCORE
        self.val_score_list = list_score
        self.cv_score = np.mean(list_score)
        self.iter_avg = np.mean(best_iter_list)

        self.logger.info(f'''
#========================================================================
# Train End.''')
        [self.logger.info(f'''
# Validation No: {i} | {self.metric}: {score}''') for i, score in enumerate(list_score)]
        self.logger.info(f'''
# Params   : {params}
# CV score : {self.cv_score}
#======================================================================== ''')
        if self.objective=='binary':
            self.sc_confusion_matrix(y_train, y_allval)
            self.logger.info(f'''
# Accuracy : {self.accuracy}  {self.true}/{len(test)}
# F1 score : {self.f1}
# TP:{self.cmx[0]}  FP:{self.cmx[2]}
# FN:{self.cmx[1]}  TN:{self.cmx[3]}
#======================================================================== ''')
        #========================================================================

        ' fold数で平均をとる '
        self.prediction = self.prediction / fold
        pred_cols = [col for col in val_stack.columns if col.count('pred_')]
        val_stack['prediction'] = val_stack[pred_cols].mean(axis=1)
        val_stack.reset_index(inplace=True)

        ' OOF for Stackng '
        pred_stack = test.reset_index()[key].to_frame()
        pred_stack['prediction'] = self.prediction

        result_stack = pd.concat([val_stack, pred_stack], axis=0, ignore_index=True)
        self.logger.info(
            f'result_stack shape: {result_stack.shape} | cnt_id: {len(result_stack[key].drop_duplicates())}')

        importance = []
        for fold_no in range(fold):
            if len(importance) == 0:
                importance = self.cv_feim[f'{fold_no}_importance'].values.copy()
            else:
                importance += self.cv_feim[f'{fold_no}_importance'].values

        self.cv_feim['avg_importance'] = importance / fold
        self.cv_feim.sort_values(by=f'avg_importance',
                            ascending=False, inplace=True)
        self.cv_feim['rank'] = np.arange(len(self.cv_feim))+1

        self.result_stack = result_stack

        return self
