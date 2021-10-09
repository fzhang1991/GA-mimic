import numpy as np
from keras.utils.vis_utils import plot_model
from keras.layers import *
import keras
from keras import regularizers
from sklearn.model_selection import KFold
import sklearn
from tensorflow.keras import activations
import argparse
import psycopg2
from sklearn import metrics
import os

# keras 2.4.3;  tensorflow: 2.2

DATA_PATH = "/home/fzhang/Benchmarking_DL_MIMICIII-master/Data_org/admdata_99p/24hrs_raw/series/imputed-normed-ep_1_24.npz"
# DATA_PATH = "/home/fzhang/Benchmarking_DL_MIMICIII-master/Data_org/admdata_17f/24hrs/non_series/sapsii.npz"
# DATA_PATH = "/home/fzhang/Benchmarking_DL_MIMICIII-master/Data_org/admdata_17f/24hrs/series/imputed-normed-ep_1_24.npz"
# DATA_PATH = "/home/fzhang/Benchmarking_DL_MIMICIII-master/Data_org/admdata_99p/24hrs_raw/non_series/tsmean_24hrs.npz"
# DATA_PATH = "/home/fzhang/Benchmarking_DL_MIMICIII-master/Data_org/admdata_17f/24hrs_raw/non_series/tsmean_24hrs.npz"
# DATA_PATH = "/Users/zhangfan/Documents/workspace/MIMIC-III/Benchmarking_DL_MIMICIII-master/Data/admdata_99p/24hrs_raw/non_series/tsmean_24hrs.npz"
DD = ""
from sklearn import linear_model, ensemble, neural_network, svm, tree, neighbors, naive_bayes
from sklearn import model_selection


def get_acc(y, p):
    def is_int(a):
        for e in a:
            if int(e) != e:
                return False
        return True

    if is_int(p):
        acc = metrics.accuracy_score(y, p)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y, p)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        p_01 = [1 if e > optimal_threshold else 0 for e in p]
        acc = metrics.accuracy_score(y, p_01)
    return acc


def lr(X, y, testx, testy, epoch, model):
    if 'tree' in model:
        # parameters = {'criterion': ('gini', 'entropy'),
        #               'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 10, 20, 30]}
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=30)
    elif 'bayes' in model:
        # clf = naive_bayes.GaussianNB()
        clf = linear_model.BayesianRidge()  # default 300
    elif 'lasso' in model:
        # clf = linear_model.LogisticRegression(max_iter=10000)  # max_iter=epoch
        clf = linear_model.LassoLarsIC()  # default 500
    else:
        0 / 0
        print("MODEL ERROR : {} ".format(model))

    clf.fit(X, y)
    testp = clf.predict(testx)
    train_p = clf.predict(X)
    # prob = clf.predict_proba(testx)

    acc = get_acc(testy, testp)
    auc_roc = metrics.roc_auc_score(testy, testp)
    auc_pr = metrics.average_precision_score(testy, testp)

    train_acc = get_acc(y, train_p)
    train_roc = metrics.roc_auc_score(y, train_p)
    train_pr = metrics.average_precision_score(y, train_p)

    print("Test :{} Train  :{}".format([acc, auc_roc, auc_pr], [train_acc, train_roc, train_pr]))

    return acc, auc_roc, auc_pr


class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc, auc_roc, auc_pr = self.model.evaluate({"x_series": x['x_series'], 'x_non_series': x['x_non_series']},
                                                         {"y_predict": y['y_predict']})
        print(
            '\nEpoch: {}, test LOSS: {}, ACC: {}, AUC-ROC: {}, AUC-PR: {}\n'.format(epoch + 1, loss, acc, auc_roc,
                                                                                    auc_pr))


class TestCallback_FNN(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc, auc_roc, auc_pr = self.model.evaluate({"x_series": x['x_series']},
                                                         {"y_predict": y['y_predict']})
        print(
            '\nEpoch: {}, test LOSS: {}, ACC: {}, AUC-ROC: {}, AUC-PR: {}\n'.format(epoch + 1, loss, acc, auc_roc,
                                                                                    auc_pr))


def getConnection():
    return psycopg2.connect(
        "dbname='mimic' user='mimicuser' host='localhost' password='mimicuser' port='5432' options='-c search_path=mimiciii'")


class MMDL():
    def __init__(self, args):
        self.args = args
        print(
            "lr:{}, bz:{}, dense_act:{}, dense_drop:{}, lstm_drop:{}, lstm_act:{}, unitsn:{}, epoch:{}, task_name:{},ga_iter:{},stand01:{}, model:{} ".format(
                args.lr,
                args.batch_size,
                args.dense_act,
                args.dense_drop,
                args.lstm_drop,
                args.lstm_act,
                args.lstm_n,
                args.epoch,
                args.task_name,
                args.ga_iter,
                args.stand01,
                args.model
            ))
        self.iteartion = 0

    def get_feature_n(self):
        data = np.load(self.args.data_path, allow_pickle=True)
        # x_matrix = np.concatenate((data['ep_tdata'], data_0['ep_tdata']), axis=-1)
        if self.args.model in ["lstm", "rnn"]:
            x_matrix = data['ep_tdata']
            feature_n = len(x_matrix[0][0])
        elif 'saps' in DATA_PATH:
            x_matrix = data['sapsii']
            feature_n = len(x_matrix[0])
        elif 'sk' in self.args.model or 'fnn' in self.args.model:
            return len(data['ep_tdata'][0][0]) * 3 + 5

        return feature_n

    def lstm_concat(self, sub_f, show_log=False):

        data = np.load(DATA_PATH, allow_pickle=True)
        x_matrix = []
        y_train = []

        if self.args.model in ['lstm', 'rnn']:
            # tempn = 132
            # sub_f_s = [e for e in sub_f if e < tempn]
            # sub_f_n = [e - tempn for e in sub_f if e >= tempn]
            x_matrix = data['ep_tdata']
            x_matrix = x_matrix[:, :, sub_f]
            y_train = data['y_icd9']
            labeln = len(y_train[0]) if type(y_train[0]) == np.ndarray else 1
            x_non_series = data['adm_features_all']
            time_n = len(x_matrix[0])
            feature_n = len(x_matrix[0][0])
            non_series_n = len(x_non_series[0])
            x_matrix = np.nan_to_num(x_matrix)
            x_non_series = np.nan_to_num(x_non_series)

        elif 'sk' in self.args.model or 'fnn' in self.args.model:
            x_matrix = data['ep_tdata']
            x_matrix_mean = np.nanmean(x_matrix, axis=1)
            x_matrix_max = np.nanmax(x_matrix, axis=1)
            x_matrix_min = np.nanmin(x_matrix, axis=1)
            assert len(x_matrix[0][0]) == len(x_matrix_mean[0]) == len(x_matrix_max[0]) == len(x_matrix_min[0])
            x_matrix = np.concatenate([x_matrix_mean, x_matrix_max, x_matrix_min], axis=-1)

            if self.args.stand01 > 0:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                x_matrix = scaler.fit_transform(x_matrix)

            x_non_series = data['adm_features_all']
            x_matrix = np.concatenate([x_matrix, x_non_series], axis=-1)
            x_matrix = np.nan_to_num(x_matrix)
            x_matrix = x_matrix[:, sub_f]
            y_train = data['y_icd9']
            # y_train = data['y_icd9'][:, [self.args.mor]]
            feature_n = len(x_matrix[0])

            if 'sk' in self.args.model:
                y_train = np.array([e[0] for e in y_train])
            elif 'fnn' in self.args.model:
                labeln = len(y_train[0]) if type(y_train[0]) == np.ndarray else 1


        else:
            0 / 0
            print("Model type Error {}".format(self.args.model))

        if self.iteartion == 0:
            print(
                "matrix[{},{}],non_series_n: {}, label_1_n: {}, sub_index:{}".format(len(y_train), feature_n,
                                                                                     non_series_n if 'lstm' in self.args.model else 0,
                                                                                     np.sum(y_train), sub_f))
            self.iteartion = 1

        kfold = KFold(n_splits=3, shuffle=True)
        LOSS, ACC, AUC_ROC, AUC_PR = [], [], [], []
        for f, (traini, testi) in enumerate(kfold.split(x_matrix, y_train)):
            if self.args.model == 'lstm':
                series_input = keras.Input(shape=(time_n, feature_n), name='x_series')
                non_series_input = keras.Input(shape=(non_series_n,), name='x_non_series')
                if self.args.lstm_act == "GRU":
                    x_s = GRU(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True)(series_input)
                elif self.args.lstm_act == "B_GRU":
                    x_s = Bidirectional(
                        GRU(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True))(series_input)
                elif self.args.lstm_act == "LSTM":
                    x_s = LSTM(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True)(series_input)
                elif self.args.lstm_act == "B_LSTM":
                    x_s = Bidirectional(
                        LSTM(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True))(series_input)
                elif self.args.lstm_act == "SimpleRNN":
                    x_s = SimpleRNN(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True)(series_input)
                elif self.args.lstm_act == "B_SimpleRNN":
                    x_s = Bidirectional(
                        SimpleRNN(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True))(series_input)
                else:
                    print("arg_lstm_activation ERROR!")
                    1 / 0
                x_s = Flatten()(x_s)
                x_s = Dropout(self.args.dense_drop)(x_s)

                x_n = Dense(10, activation=self.args.dense_act)(non_series_input)
                x_n = Dropout(self.args.dense_drop)(x_n)
                x_n = Dense(10, activation=self.args.dense_act)(x_n)
                x_n = Dropout(self.args.dense_drop)(x_n)

                x = concatenate([x_n, x_s])
                # x = Dense(240, activation=dense_act)(x)
                # x = Dropout(n_dropout)(x)
                x = Dense(40, activation=self.args.dense_act)(x)
                x = Dropout(self.args.dense_drop)(x)

                x = Dense(labeln, activation=self.args.dense_act)(x)
                x = BatchNormalization()(x)

                if self.args.task_name in "y_mor y_icd9":
                    y_predict = Activation(activation='sigmoid', name='y_predict')(x)

                elif self.args.task_name in "y_loss":
                    y_predict = x

                model = keras.Model(inputs=[series_input, non_series_input], outputs=[y_predict])
                plot_model(model, to_file='{}concat_model.png'.format(DD), show_shapes=True);
                optimizer = keras.optimizers.RMSprop(learning_rate=self.args.lr)
                model.compile(loss='binary_crossentropy',  # mean_squared_error
                              optimizer=optimizer,
                              metrics=['accuracy', keras.metrics.AUC(curve='ROC'),
                                       keras.metrics.AUC(curve='PR')])  # , auroc

                # model.fit({"x_series": x_matrix[traini], 'x_non_series': x_non_series[traini]},
                #           {"y_predict": y_train[traini]},
                #           epochs=epoch, batch_size=batch_size,
                #           callbacks=[TestCallback(({"x_series": x_matrix[testi], 'x_non_series': x_non_series[testi]},
                #                                    {"y_predict": y_train[testi]}))])
                callbacks = [TestCallback(({"x_series": x_matrix[testi], 'x_non_series': x_non_series[testi]},
                                           {"y_predict": y_train[testi]}))] if show_log else None

                model.fit({"x_series": x_matrix[traini], 'x_non_series': x_non_series[traini]},
                          {"y_predict": y_train[traini]},
                          epochs=self.args.epoch, batch_size=self.args.batch_size, verbose=0, callbacks=callbacks)

                layer_weights = {}
                for layer in model.layers:
                    weights = layer.get_weights()  # list of numpy array
                    layer_weights.update({layer.name: weights})

                r = model.evaluate({"x_series": x_matrix[testi], 'x_non_series': x_non_series[testi]},
                                   {"y_predict": y_train[testi]})

                LOSS.append(r[0])
                ACC.append(r[1])
                AUC_ROC.append(r[2])
                AUC_PR.append(r[3])
            elif self.args.model == 'rnn':
                series_input = keras.Input(shape=(time_n, feature_n), name='x_series')
                x_s = GRU(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True)(series_input)
                x_s = Flatten()(x_s)
                x_s = Dropout(self.args.dense_drop)(x_s)

                x = Dense(40, activation=self.args.dense_act)(x_s)
                x = Dropout(self.args.dense_drop)(x)

                x = Dense(labeln, activation=self.args.dense_act)(x)
                x = BatchNormalization()(x)

                if self.args.task_name in "y_mor y_icd9":
                    y_predict = Activation(activation='sigmoid', name='y_predict')(x)

                elif self.args.task_name in "y_loss":
                    y_predict = x

                model = keras.Model(inputs=[series_input], outputs=[y_predict])
                plot_model(model, to_file='{}rnn_model.png'.format(DD), show_shapes=True);
                optimizer = keras.optimizers.RMSprop(learning_rate=self.args.lr)
                model.compile(loss='binary_crossentropy',  # mean_squared_error
                              optimizer=optimizer,
                              metrics=['accuracy', keras.metrics.AUC(curve='ROC'),
                                       keras.metrics.AUC(curve='PR')])  # , auroc

                callbacks = [TestCallback_FNN(({"x_series": x_matrix[testi]},
                                               {"y_predict": y_train[testi]}))] if show_log else None

                model.fit({"x_series": x_matrix[traini]},
                          {"y_predict": y_train[traini]},
                          epochs=self.args.epoch, batch_size=self.args.batch_size, verbose=0, callbacks=callbacks)

                layer_weights = {}
                for layer in model.layers:
                    weights = layer.get_weights()  # list of numpy array
                    layer_weights.update({layer.name: weights})

                r = model.evaluate({"x_series": x_matrix[testi]},
                                   {"y_predict": y_train[testi]})

                LOSS.append(r[0])
                ACC.append(r[1])
                AUC_ROC.append(r[2])
                AUC_PR.append(r[3])
            elif self.args.model == 'fnn':
                series_input = keras.Input(shape=(feature_n), name='x_series')
                # x_s = GRU(self.args.lstm_n, dropout=self.args.lstm_drop, return_sequences=True)(series_input)
                x = Dense(self.args.lstm_n, activation=self.args.dense_act)(series_input)
                x = Dropout(self.args.dense_drop)(x)

                x = Dense(40, activation=self.args.dense_act)(x)
                x = Dropout(self.args.dense_drop)(x)

                y_predict = Dense(labeln, activation='sigmoid', name='y_predict')(x)
                # x = BatchNormalization()(x)

                # if self.args.task_name in "y_mor y_icd9":
                #     y_predict = Activation(activation='sigmoid', name='y_predict')(x)
                #
                # elif self.args.task_name in "y_loss":
                #     y_predict = x

                model = keras.Model(inputs=[series_input], outputs=[y_predict])
                plot_model(model, to_file='fnn_model.png'.format(DD), show_shapes=True);
                optimizer = keras.optimizers.RMSprop(learning_rate=self.args.lr)
                model.compile(loss='binary_crossentropy',  # mean_squared_error
                              optimizer=optimizer,
                              metrics=['accuracy', keras.metrics.AUC(curve='ROC'),
                                       keras.metrics.AUC(curve='PR')])  # , auroc
                callbacks = [TestCallback_FNN(({"x_series": x_matrix[testi]},
                                               {"y_predict": y_train[testi]}))] if show_log else None

                model.fit({"x_series": x_matrix[traini]},
                          {"y_predict": y_train[traini]},
                          epochs=self.args.epoch, batch_size=self.args.batch_size, verbose=0, callbacks=callbacks)

                input_dense_weights = model.layers[1].get_weights()

                layer_weights = {}
                for layer in model.layers:
                    weights = layer.get_weights()  # list of numpy array
                    layer_weights.update({layer.name: weights})

                r = model.evaluate({"x_series": x_matrix[testi]},
                                   {"y_predict": y_train[testi]})

                LOSS.append(r[0])
                ACC.append(r[1])
                AUC_ROC.append(r[2])
                AUC_PR.append(r[3])
            elif 'sk' in self.args.model:
                acc, roc, pr = lr(x_matrix[traini], y_train[traini], x_matrix[testi], y_train[testi], self.args.epoch,
                                  self.args.model)
                ACC.append(acc)
                AUC_ROC.append(roc)
                AUC_PR.append(pr)
                print(
                    '\nEpoch: {}, test LOSS: {}, ACC: {}, AUC-ROC: {}, AUC-PR: {}\n'.format(0, 0, acc,
                                                                                            roc,
                                                                                            pr))
            else:
                0 / 0
                print("MODEL Error! :{}".format(self.args.model))

        return np.mean(AUC_ROC)


import datetime
import random


class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness


class GeneticAlgorithm():
    geneset = []

    def guess_password(self, classifier, geneset, targetn, ga_feature, ga_iter):
        print("Featurn N: {}, target N :{}".format(len(geneset), targetn))
        self.classifier = classifier
        self.geneset = geneset
        self.startTime = datetime.datetime.now()
        self.parent = None
        self.ga_iter = ga_iter
        if ga_feature != "" and ga_feature != "all":
            index = [int(e) for e in ga_feature.strip().split(",")]
            fitness = self.get_fitness(index)
            self.parent = Chromosome(index, fitness)
            print("Parent: fitness {} ,{}".format(fitness, index))

        optimalFitness = 1
        best = self.get_best(targetn, optimalFitness, geneset)
        assert best.Fitness == optimalFitness

    def get_best(self, targetLen, optimalFitness, geneSet):
        def _generate_parent(length, geneSet):
            genes = []
            while len(genes) < length:
                sampleSize = min(length - len(genes), len(geneSet))
                genes.extend(random.sample(geneSet, sampleSize))
            assert len(set(genes)) == len(genes)
            return Chromosome(genes, self.get_fitness(genes))

        def _mutate(parent, geneSet):
            index = random.randint(0, len(parent.Genes) - 1)
            childGenes = list(parent.Genes)
            diffGeneSet = set(geneSet).difference(childGenes)
            newGene = random.sample(diffGeneSet, 1)[0]
            childGenes[index] = newGene
            fitness = self.get_fitness(childGenes)
            return Chromosome(childGenes, fitness)

        random.seed()
        bestParent = _generate_parent(targetLen, geneSet) if self.parent == None else self.parent
        self.display(bestParent, 0)
        if bestParent.Fitness >= optimalFitness:
            return bestParent
        mutate_i = self.ga_iter
        while True:
            child = _mutate(bestParent, geneSet)
            mutate_i += 1
            if mutate_i % 100 == 0:
                print("GA mutate N: {}".format(mutate_i))
            if mutate_i >= 20000:
                return
            if bestParent.Fitness >= child.Fitness:
                continue
            self.display(child, mutate_i)
            if child.Fitness >= optimalFitness:
                return child
            bestParent = child

    def get_fitness(self, guess):
        auc = self.classifier.lstm_concat(guess)
        return auc

    def display(self, candidate, n):
        timeDiff = datetime.datetime.now() - self.startTime
        print("GA UP N {},\t{},\t{},\t{}".format(n, candidate.Fitness, timeDiff, candidate.Genes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--data-path', type=str,
                        default=DATA_PATH)

    parser.add_argument('--batch-size', default=256, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lstm-drop', default=0.0, type=float, help='LSTM dropout')
    parser.add_argument('--lstm-act', default='GRU', type=str, help='GRU,B_GRU,LSTM,B_LSTM,SimpleRNN,B_SimpleRNN')
    parser.add_argument('--lstm-n', default=128, type=int, help='Batch size for training')
    parser.add_argument('--dense-drop', default=0.1, type=float, help='dense dropout')
    parser.add_argument('--dense-act', default=None, type=str, help='dense act')
    parser.add_argument('--task-name', default='y_icd9', type=str, help='y_mor y_icd9 y_los')
    # parser.add_argument('--mor', default=0, type=int)
    parser.add_argument('--epoch', default=300, type=int, help='epoch')

    parser.add_argument('--feature-n', default=0, type=int)
    parser.add_argument('--ga-feature', default="all", type=str, help="Init GA feature index")  # eg."1,2"
    parser.add_argument('--ga-iter', default=0, type=int, help="Init GA iteration n")
    parser.add_argument('--stand01', default=0, type=int)
    parser.add_argument('--model', default='lstm', type=str, help="lstm,fnn,rnn,sk-tree")
    parser.add_argument('--ga', default=0, type=int, help='1:ga , 0: test all')

    args = parser.parse_args()
    if args.ga > 0:
        args.epoch = 50
    classifier = MMDL(args)

    if args.ga > 0:
        # ga
        geneset = list(range(classifier.get_feature_n()))
        g = GeneticAlgorithm()
        if args.ga_feature != 'all':
            args.feature_n = len(args.ga_feature.split(","))
        g.guess_password(classifier, geneset, args.feature_n, args.ga_feature, args.ga_iter)
    else:
        # once lstm
        sub_f = [int(e) for e in args.ga_feature.split(",")] if args.ga_feature != "all" else list(
            range(classifier.get_feature_n()))
        classifier.lstm_concat(sub_f, True)

