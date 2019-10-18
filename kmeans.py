import lemmatize as lem
import pca_tsne as pt

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i+1))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


data = pd.read_csv('select_no_plural.csv', sep=';', quotechar='"')
data = lem.perform_lemmatize_dataset(data)

general_words = ["data", "use", "using", "used", "paper", "method", "analysis", "different",
                 "based", "result", "problem", "furthermore", "propose", "important", "general",
                 "approach", "present", "aim", "work", "make", "xxxix", "rio", "grande"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(general_words)

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = my_stop_words
)

matrix = tfidf.fit_transform(data.setting_value)

means_clusters = KMeans(n_clusters=17, random_state=20).fit_predict(matrix)
    
pt.plot_tsne_pca(matrix, means_clusters)

get_top_keywords(matrix, means_clusters, tfidf.get_feature_names(), 10)

print("\nClusters Size")
print(np.bincount(means_clusters))