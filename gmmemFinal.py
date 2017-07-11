"""
================================
Gaussian Mixture Model Selection
================================

Source: http://docs.w3cub.com/scikit_learn/auto_examples/mixture/plot_gmm_selection/
        with appropriate modifications by Dario H. Romero - Dataset from Kaggle

This example shows that model selection can be performed with
Gaussian Mixture Models using information-theoretic criteria (BIC).
Model selection concerns both the covariance type
and the number of components in the model.
In that case, AIC also provides the right result (not shown to save time),
but BIC is better suited if the problem is to identify the right model.
Unlike Bayesian procedures, such inferences are prior-free.

In that case, the model with 8 components and full covariance
(which corresponds to the true generative model) is selected.
"""

import pandas as pd
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import silhouette_score

print(__doc__)

# command on the output console to extend linesize
desired_width = 320
pd.set_option('display.width', desired_width)

# Load Dataset (Source: from Kaggle)
X = pd.read_csv('HR_data.csv')
X = X[X.columns[2:len(X.columns)]]
X.sales[X.sales == 'RandD'] = 'RnD'

# split column 'sales' into several binary columns
data_sales = pd.get_dummies(X['sales'])
# number = LabelEncoder()
# X['sales'] = number.fit_transform(X['sales'].astype('str'))
# split column 'salary' into several binary columns
data_salary = pd.get_dummies(X['salary'])
# X['salary'] = number.fit_transform(X['salary'].astype('str'))

# data_sales.head()
X.drop('sales', axis=1, inplace=True)
X.drop('salary', axis=1, inplace=True)
X = X.join(data_sales)
X = X.join(data_salary)
col_names = list(X)
#print(X.head())

# Initialize parameters
lowest_bic = np.infty
# lowest_aic = np.infty
bic = []
# aic = []
n_components_range = range(1, 13)

# cv_types is cross validation types
cv_types = ['spherical', 'tied', 'diag', 'full']

# Generate random seed, for reproducibility
np.random.seed(123)

for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        # to avoid the risk of over/under fitting the data we need
        # to run an optimization loop on the number of components
        # using BIC (Bayesian Information Criteria) as a cost function.
        bic.append(gmm.bic(X))
        # aic.append(gmm.aic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_bic_gmm = gmm
        # if aic[-1] < lowest_aic:
        #     lowest_aic = aic[-1]
        #     best_aic_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange',
                              'skyblue', 'fuchsia', 'green', 'mediumorchid',
                              'salmon', 'turquoise', 'seagreen', 'dodgerblue'])
# color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
clf_bic = best_bic_gmm
bars = []
Y_ = clf_bic.predict(X)

# print("======== number of components ========")
# print(" n_components: {}.".format(clf_bic.n_components))

# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

#
# print("No of Components: {}".format(clf_bic.n_components))
# print("Means: {}".format(clf_bic.means_))
# print("Variance type: {}".format(clf_bic.covariance_type))
# print("Variances: {}".format(clf_bic.covariances_))
# print("Converged [True/False]: {}".format(clf_bic.converged_))

# Plot the winner
splot = plt.subplot(2, 1, 2)
X = np.array(X)

for i, (mean, cov, color) in enumerate(zip(clf_bic.means_, clf_bic.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model, {} components'.format(clf_bic.n_components))
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.savefig('gmmBIC_getDummy_values.png')
plt.show()

# Print results to the console
ds = pd.DataFrame(X)
# Assign column names to ds DataFrame
ds.columns = col_names
# convert Y_ predicted cluster into dataframe
Y_df = pd.DataFrame(Y_)
Y_df.rename(columns={0: 'cluster'}, inplace=True)
# Concat both DataFrames
ds = pd.concat([Y_df, ds], axis=1)
ds = ds.sort_values(by=['cluster'])
ds.head()
ds.tail()

# Plot 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# dss = pd.DataFrame([])

# n = 10
#
# for i in range(max(ds['cluster'].unique() + 1)):
#     dss_x = pd.DataFrame(ds[ds.cluster == i][:(max(ds['cluster'].unique()))][['cluster',
#                                                                                   'satisfaction_level',
#                                                                                   'last_evaluation',
#                                                                                   'average_monthly_hours']])
#     dss = pd.concat([dss, dss_x])
#
# print(dss)
# colors = ('navy', 'turquoise', 'cornflowerblue', 'darkorange', 'skyblue', 'fuchsia',
#           'green', 'mediumorchid', 'salmon', 'turquoise', 'seagreen', 'dodgerblue')
# # for i in range(max(ds['cluster'].unique() + 1)):
# for dss, color in zip(dss[:3], colors):
#     print(color)
#     # ax.scatter(dss['satisfaction_level'], dss['last_evaluation'], dss['average_monthly_hours'],
#     #            c=color[i], alpha=0.8)
#
# ax.set_xlabel("satisfaction level")
# ax.set_ylabel("last evaluation")
# ax.set_zlabel("avg monthly hours")
# plt.title("3D Plot - Only 3 Columns from Dataset")
# plt.show()
#
# # Plot 3D - test 2
# # group = pd.DataFrame([])
# # for i in range(max(ds['cluster'].unique() + 1)):
# #     print("Cluster {}".format(i))
# #     df = pd.DataFrame(ds[ds.cluster == i][:10][['satisfaction_level', 'last_evaluation', 'average_monthly_hours']])
# #     group = pd.concat([group, df])
# #     print(df)
# #     print(" ======== ")
# # print(group)
#
# # pd.DataFrame(ds[ds.cluster == 0][:10])[['satisfaction_level', 'last_evaluation', 'average_monthly_hours']]

