import numpy as np

from sklearn import manifold

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import distance
import numpy as np


def OneSentenceEval(x, sets, eval=distance.cosine):
    return np.array([eval(x,  s) for s in sets])


def totalEval(sets, eval=distance.cosine):
    return np.array([OneSentenceEval(x, sets, eval) for x in sets])


def two_dim_sim(matrix, LIM=.1):
    return [(i, np.where(matrix[i] < LIM))
            for i, _ in enumerate(matrix) if len(np.where(matrix[i] < LIM)[0]) > 1]


def PrintSimSentence(sets, simArr, lim=300):
    for i in simArr:
        print(sets[i][:lim])


def GetTfidfMatrix(string_content):
    from sklearn import feature_extraction
    tfidf = feature_extraction.text.TfidfVectorizer()
    tf_idf = tfidf.fit_transform(string_content)
    return tf_idf.toarray()


def LowDimProject(data, projector=manifold.t_sne.TSNE()):
    return projector.fit_transform(data)


def

for cent in range(k):
    one = ans[nonzero(clusterAssment[:, 0].A == cent)[0]]
    plt.scatter(one[:, 0], one[:, 1])
plt.scatter(ans[:])
