import numpy as np
import itertools
import pandas as pd
import time


class HDMR:
    '''
    Find feature interactions from a model.

    '''
    def __init__(self, model, X, sample_size, candidate_size, fea_comb_local_distri=[]):
        '''

        :param model: the interpreted model.
        :param X: the dataset used to interpret model.
        :param sample_size: the size of sample randomly selected from X.
        :param candidate_size: the number of instances for which the conditional means are calculated.
        :param fea_comb_local_distri: the set of feature combinations whose local feature interactions will be saved
        '''
        self.X = X
        self.model = model
        self.sample_size = sample_size
        self.candidate_size = candidate_size
        self.fea_comb_local_distri = fea_comb_local_distri
        self.nfea = self.X.shape[1]  # number of features
        self.fea_interaction = []
        self.fea_comb = []
        self.dist_results = []

        self.Y = self.__pred_output__(self.X)
        if np.ndim(self.Y) > 1:
            self.nclass = self.Y.shape[1]  # number of classes
        else:
            self.nclass = 1
        np.random.seed(10)
        self.data = self.__generate_sample__()
        self.candidates = self.__generate_candidates__()

    def __generate_sample__(self):
        '''
        Generate a random sample from dataset.

        :return: a sample dataset.
        '''
        data = np.zeros([self.sample_size, self.nfea])
        for i in range(self.nfea):
            data[:, i] = np.random.choice(self.X[:, i], self.sample_size, replace=True)
        return data

    def __generate_candidates__(self):
        '''
        generate a candidate set for which the conditional means are calculated.

        :return: a candidate set.
        '''
        candidates = np.zeros([self.candidate_size, self.nfea])
        for i in range(self.nfea):
            temp = np.unique(self.X[:, i])
            if len(temp) >= self.candidate_size:
                rep = False
            else:
                rep = True
            candidates[:, i] = np.random.choice(temp, self.candidate_size, replace=rep)
        return candidates

    def __pred_output__(self, x):
        ''' Predict the outputs.

        :param x: the data for prediction.
        :type x: np.array
        :return: the prediction results.
        :rtype: array

        '''
        if np.ndim(x) == 1:
            x = [x]
        pred = self.model(x)
        return pred

    def __cal_cond_mean__(self, feas):
        '''
        Calculate conditional mean.

        :param feas: the conditional features.
        :return: conditional mean.
        '''
        cond_mean = np.zeros([self.candidate_size, self.nclass])
        for i in range(self.candidate_size):
            data = self.data.copy()
            data[:, feas] = self.candidates[i, feas]
            cond_mean[i] = np.mean(self.__pred_output__(data), axis=0)
        return cond_mean

    def detect_interaction(self, threshold=0.01, max_order=3):
        '''
        Find feature interactions in a model, whose f_s are greater than a threshold for all candidate instances.

        :param threshold: only the feature interactions whose g_S are greater than the threshold are found.
        :param max_order: the maximum length of feature interactions.
        :return: a series of feature interactions and their values of g_S.
        '''
        f0 = np.mean(self.__pred_output__(self.data), axis=0)
        fea_comb = [[[]]]  # feature combination
        fea_inte = [[[np.array(f0)]]]  # strength of feature interaction
        fea_comb.append([[i] for i in range(self.nfea)])
        fea_inte.append([self.__cal_cond_mean__([i]) - f0 for i in range(self.nfea)])  # 1-order interaction
        for len_comb in range(1, max_order+1):
            print("=======", len_comb)
            cur_comb = fea_comb[len_comb]  # get the last feature interactions
            new_comb = []
            new_inte = []
            for item in cur_comb:
                cnt = 0
                for x in range(max(item) + 1, self.nfea):
                    # judge if the new combination is an valid solution,
                    # i.e., all of its sub-items are valid
                    available = True
                    for subset in itertools.combinations(item, len_comb - 1):
                        if list(subset) + [x] not in fea_comb[len_comb]:
                            available = False
                            break

                    if available:
                        # calculate feature interaction
                        new_item = item + [x]
                        cur_inte = self.__cal_cond_mean__(new_item)
                        for L in range(0, len_comb + 1):
                            for subset in itertools.combinations(new_item, L):
                                indx = fea_comb[L].index(list(subset))
                                cur_inte -= fea_inte[L][indx]
                        # judge if the median of the feature interaction values
                        # satisfies the threshold
                        if np.median(np.abs(cur_inte)) >= threshold:
                            new_comb.append(new_item)
                            new_inte.append(cur_inte)
                            cnt += 1
                            # save the local interactions for the feature combinations that are
                            # in the self.distribution
                            if new_item in self.fea_comb_local_distri:
                                self.dist_results.append(cur_inte)

            if len(new_comb) > 0:
                fea_comb.append(new_comb)
                fea_inte.append(new_inte)
            else:
                break

        self.fea_comb = fea_comb
        self.fea_interaction = fea_inte

        index = []
        feaint_eval = []

        # sorting the feature interactions based on their values of g_S
        for i in range(2, len(self.fea_comb)):
            if self.nclass == 1:
                index += [str(x) for x in self.fea_comb[i]]
                feaint_eval += [np.median(np.abs(x)) for x in self.fea_interaction[i]]
            else:
                index += [str(x) for x in self.fea_comb[i]]
                feaint_eval += [np.max(np.median(np.abs(x), axis=0)) for x in self.fea_interaction[i]]
        feaInt = pd.Series(feaint_eval, index=index)
        return feaInt.sort_values()

    def detect_max_set(self):
        '''
        Find the list of feature interactions with maximum order, i.e., for interactions
        [1,2] , [1,3], [2,3] and [1,2,3], only [1,2,3] is kept.

        :return: a list of feature interactions.
        '''
        feaint_all = []
        for i in range(1, len(self.fea_comb)):
            feaint_all += [x for x in self.fea_comb[i]]

        labels = [True] * len(feaint_all)
        for i in range(len(feaint_all) - 1, 2, -1):
            if labels[i]:
                for j in range(0, i):
                    if set(feaint_all[i]) > set(feaint_all[j]):
                        labels[j] = False
        return [feaint_all[i] for i in range(len(feaint_all)) if labels[i]]


class H_STATISTIC:
    def __init__(self, model, X):
        '''
            Find 2- and 3-way feature interactions based on H_Statistics

            :param model: the interpreted model.
            :param X: the dataset used to interpret model.

            Reference: Friedman, J. H., & Popescu, B. E. (2008).
            Predictive learning via rule ensembles. The Annals of Applied Statistics, 2(3), 916-954.
            doi: 10.1214/07-AOAS148
        '''
        self.X = X
        self.model = model
        self.nfea = self.X.shape[1]
        self.nobs = self.X.shape[0]

    def __pdp_1way__(self):
        res = np.zeros([self.nfea, self.nobs])
        for fea in range(self.nfea):
            tmpData = np.tile(self.X, [self.nobs, 1])
            tmpData[:, fea] = np.repeat(self.X[:, fea], self.nobs, axis=0)
            tmpPred = self.model(tmpData)
            for i in range(self.nobs):
                res[fea, i] = np.mean(tmpPred[i*self.nobs: (i+1)*self.nobs])
        self.pdp_1way = res

    def __pdp_2way__(self):
        res = np.zeros([self.nfea, self.nfea, self.nobs])
        for fea_1 in range(self.nfea - 1):
            for fea_2 in range(fea_1+1, self.nfea):
                tmpData = np.tile(self.X, [self.nobs, 1])
                comb = [fea_1, fea_2]
                tmpData[:, comb] = np.repeat(self.X[:, comb], self.nobs, axis=0)
                tmpPred = self.model(tmpData)
                for i in range(self.nobs):
                    res[fea_1, fea_2, i] = np.mean(tmpPred[i*self.nobs: (i+1)*self.nobs])
        self.pdp_2way = res

    def __pdp_3way__(self):
        res = np.zeros([self.nfea, self.nfea, self.nfea, self.nobs])
        for fea_1 in range(self.nfea-2):
            for fea_2 in range(fea_1+1, self.nfea-1):
                for fea_3 in range(fea_2+1, self.nfea):
                    tmpData = np.tile(self.X, [self.nobs, 1])
                    comb = [fea_1, fea_2, fea_3]
                    tmpData[:, comb] = np.repeat(self.X[:, comb], self.nobs, axis=0)
                    tmpPred = self.model(tmpData)
                    for i in range(self.nobs):
                        res[fea_1, fea_2, fea_3, i] = np.mean(tmpPred[i*self.nobs: (i+1)*self.nobs])
        self.pdp_3way = res

    def detect_interaction(self):
        ''''''
        self.__pdp_1way__()
        self.__pdp_2way__()
        self.__pdp_3way__()

        res = []
        # 2-way
        for fea_1 in range(self.nfea - 1):
            for fea_2 in range(fea_1 + 1, self.nfea):
                imp = np.sum((self.pdp_2way[fea_1, fea_2, :] - self.pdp_1way[fea_1, :]
                              - self.pdp_1way[fea_2, :]) ** 2) / \
                      np.sum(self.pdp_2way[fea_1, fea_2, :] ** 2)
                res.append([fea_1, fea_2, -1, imp])
        # 3-way
        for fea_1 in range(self.nfea-2):
            for fea_2 in range(fea_1+1, self.nfea-1):
                for fea_3 in range(fea_2+1, self.nfea):
                    imp = np.sum((self.pdp_3way[fea_1, fea_2, fea_3, :]
                                  - self.pdp_2way[fea_1, fea_2, :]
                                  - self.pdp_2way[fea_1, fea_3, :]
                                  - self.pdp_2way[fea_2, fea_3, :]
                                  + self.pdp_1way[fea_1, :]
                                  + self.pdp_1way[fea_2, :]
                                  + self.pdp_1way[fea_3, :]) ** 2) / \
                          np.sum(self.pdp_3way[fea_1, fea_2, fea_3, :] ** 2)
                    res.append([fea_1, fea_2, fea_3, imp])

        res = np.array(res)
        indx = list(np.argsort(res[:, 3]))
        indx.reverse()
        return res[indx, :]


class COND_FEAIMP:
    def __init__(self, model, X):
        '''
            Find 2-way feature interactions based on H_Statistics

            :param model: the interpreted model.
            :param X: the dataset used to interpret model.

            Reference: Greenwell, B. M., Boehmke, B. C., & McCarthy, A. J. (2018).
            A Simple and Effective Model-Based Variable Importance Measure.
            arXiv preprint arXiv:1805.04755
        '''
        self.X = X
        self.model = model
        self.nfea = self.X.shape[1]
        self.nobs = self.X.shape[0]

    def detect_interaction(self):
        std_cond_feaimp = np.zeros([self.nfea, self.nfea])
        cond_fea_imp = np.zeros([self.nobs])
        tmp_res = np.zeros([self.nobs])

        for fea_1 in range(self.nfea):
            for fea_2 in range(self.nfea):
                if fea_1 != fea_2:
                    for i in range(self.nobs):
                        fea_val = self.X[i, fea_2]
                        tmpData = np.tile(self.X, [self.nobs, 1])
                        tmpData[:, fea_2] = fea_val
                        tmpData[:, fea_1] = np.repeat(self.X[:, fea_1], self.nobs, axis=0)
                        tmpPred = self.model(tmpData)
                        for j in range(self.nobs):
                            tmp_res[j] = np.mean(tmpPred[j * self.nobs: (j + 1) * self.nobs])
                        cond_fea_imp[i] = np.std(tmp_res)
                    std_cond_feaimp[fea_1, fea_2] = np.std(cond_fea_imp)

        res = []
        for fea_1 in range(self.nfea - 1):
            for fea_2 in range(fea_1+1, self.nfea):
                interaction = (std_cond_feaimp[fea_1, fea_2] + std_cond_feaimp[fea_2, fea_1]) / 2
                res.append([fea_1, fea_2, interaction])
        res = np.array(res)
        indx = list(np.argsort(res[:, 2]))
        indx.reverse()
        return res[indx, :]


class MODEL_ERROR:
    def __init__(self, model, X, Y):
        '''
            Find the feature interactions based on model prediction errors.

            :param model: the interpreted model.
            :param X: the input features used to interpret model.
            :param Y: the output feature

            Reference: Oh, S. (2019).
            Feature interaction in terms of prediction performance.
            Applied Sciences, 9(23), 5191. doi: 10.3390/app9235191
        '''
        self.X = X
        self.model = model
        self.nfea = self.X.shape[1]
        self.nobs = self.X.shape[0]
        self.Y = Y

        self.shuffleIndx = np.array(range(self.nobs))
        np.random.shuffle(self.shuffleIndx)

        self.error = np.sqrt(np.mean((self.Y - self.model(X)) ** 2))

    def __model_error_(self):
        res_1way = np.zeros([self.nfea])
        for fea in range(self.nfea):
            tmpData = self.X.copy()
            np.random.shuffle(self.shuffleIndx)
            tmpData[:, fea] = self.X[self.shuffleIndx, fea]
            res_1way[fea] = np.sqrt(np.mean((self.Y - self.model(tmpData)) ** 2))
        res_2way = np.zeros([self.nfea, self.nfea])
        for fea_1 in range(self.nfea - 1):
            for fea_2 in range(fea_1+1, self.nfea):
                tmpData = self.X.copy()
                np.random.shuffle(self.shuffleIndx)
                tmp = self.X[:, [fea_1, fea_2]]
                tmpData[:, [fea_1, fea_2]] = tmp[self.shuffleIndx, :]
                res_2way[fea_1, fea_2] = np.sqrt(np.mean((self.Y - self.model(tmpData)) ** 2))
        self.res_1way = res_1way
        self.res_2way = res_2way

    def detect_interaction(self):
        self.__model_error_()
        res = []
        for fea_1 in range(self.nfea - 1):
            for fea_2 in range(fea_1+1, self.nfea):
                imp = self.res_2way[fea_1, fea_2] - self.res_1way[fea_1] - \
                      self.res_1way[fea_2] + self.error
                res.append([fea_1, fea_2, imp])
        res = np.array(res)
        indx = list(np.argsort(np.abs(res[:, 2])))
        indx.reverse()
        return res[indx, :]


class ANOVA:
    def __init__(self, model, X):
        '''
                Find 2-way feature interactions based on ANOVA

                :param model: the interpreted model.
                :param X: the dataset used to interpret model.

                Reference: Hooker, G. (2004). Discovering additive structure in black box functions.
                Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery
                and Data Mining (pp. 425–435).
        '''
        self.X = X
        self.model = model
        self.nfea = self.X.shape[1]
        self.nobs = self.X.shape[0]

        self.pred = model(X)

    def __cond_1way__(self):
        res = np.zeros([self.nfea, self.nobs])
        for fea in range(self.nfea):
            tmpData = np.repeat(self.X, self.nobs, axis=0)
            tmpData[:, fea] = np.tile(self.X[:, fea], [self.nobs])
            tmpPred = self.model(tmpData)
            for i in range(self.nobs):
                res[fea, i] = np.mean(tmpPred[i * self.nobs: (i + 1) * self.nobs])
        self.cond_1way = res

    def __cond_2way__(self):
        res = np.zeros([self.nfea, self.nfea, self.nobs])
        for fea_1 in range(self.nfea - 1):
            for fea_2 in range(fea_1+1, self.nfea):
                tmpData = np.repeat(self.X, self.nobs, axis=0)
                comb = [fea_1, fea_2]
                tmpData[:, comb] = np.tile(self.X[:, comb], [self.nobs, 1])
                tmpPred = self.model(tmpData)
                for i in range(self.nobs):
                    res[fea_1, fea_2, i] = np.mean(tmpPred[i*self.nobs: (i+1)*self.nobs])
        self.cond_2way = res

    def detect_interaction(self):
        self.__cond_1way__()
        self.__cond_2way__()
        res = []
        for fea_1 in range(self.nfea - 1):
            for fea_2 in range(fea_1+1, self.nfea):
                imp = self.pred - self.cond_1way[fea_1, :] - self.cond_1way[fea_2, :] + \
                      self.cond_2way[fea_1, fea_2, :]
                imp = np.mean(imp ** 2)
                res.append([fea_1, fea_2, imp])
        res = np.array(res)
        indx = list(np.argsort(res[:, 2]))
        indx.reverse()
        return res[indx, :]
