from featureInteraction import *
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np
from sklearn.datasets import load_boston, load_iris


# ----- synthetic model
def F1(ds):
    return 3.14 ** (ds[:, 0] * ds[:, 1]) * np.sqrt(2 * ds[:, 2]) - \
           1 / np.sin(ds[:, 3]) + \
           np.log(ds[:, 2] + ds[:, 4]) - \
           ds[:, 8] / ds[:, 9] * np.sqrt(ds[:, 6] / ds[:, 7]) - \
           ds[:, 1] * ds[:, 6]


# generate data
np.random.seed(3)
temp = np.hstack([np.random.uniform(0, 1, 1200).reshape([200, 6]),
                  np.random.uniform(0.6, 1, 800).reshape([200, 4])])
data = temp[:, [0, 1, 2, 6, 7, 3, 4, 8, 5, 9]]

# HDMR
hdmr = HDMR(model=F1, X=data, sample_size=100, candidate_size=30)
hdmr_res = hdmr.detect_interaction(threshold=0.0001, max_order=4)
print(hdmr_res)
hdmr_max = hdmr.detect_max_set()

# H-Statistics
h_stat = H_STATISTIC(F1, data)
hstat_res = h_stat.detect_interaction()
print(hstat_res)

# Conditional feature importance
con_feaimp = COND_FEAIMP(F1, data)
confeaimp_res = con_feaimp.detect_interaction()
print(confeaimp_res)

# Model error-based
np.random.seed(3)
model_err = MODEL_ERROR(F1, data, F1(data))
moderr_res = model_err.detect_interaction()
print(moderr_res)

# Additive Structure
anova = ANOVA(model=F1, X=data)
anova_res = anova.detect_interaction()
print(anova_res)

# ----- Boston housing Data
# build a regression model based on xgboost
ds = load_boston()
X_train, X_test, Y_train, Y_test = \
    train_test_split(scale(ds.data), ds.target, test_size=0.3, random_state=3)
xgb = xgboost.XGBRegressor()
xgb.fit(X_train, Y_train)

hdmr = HDMR(model=xgb.predict, X=X_test, sample_size=100, candidate_size=50)
res = hdmr.detect_interaction(threshold=0.1, max_order=4)
print(res)

# ----- Iris data
# build a classification model based on xgboost
ds = load_iris()
X_train, X_test, Y_train, Y_test = \
    train_test_split(scale(ds.data), ds.target, test_size=0.3, random_state=3)
xgb = xgboost.XGBClassifier()
xgb.fit(X_train, Y_train)

hdmr = HDMR(model=xgb.predict, X=X_test, sample_size=100, candidate_size=50)
res = hdmr.detect_interaction(threshold=0.05, max_order=4)
print(res)

