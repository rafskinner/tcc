import lemmatize as lem
import pca_tsne as pt

import re
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    top_keywords = ""
    
    for i,r in df.iterrows():
        top_keywords += '\nCluster {}'.format(i+1)
        top_keywords += ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])

    return top_keywords


clusters_size_keywords = []

data = pd.read_csv('select_no_plural.csv', sep=';', quotechar='"')
data = lem.perform_lemmatize_dataset(data)

general_words = ["data", "use", "using", "used", "paper", "method", "analysis", "different",
                 "based", "result", "problem", "furthermore", "propose", "important", "general",
                 "approach", "present", "aim", "work", "make", "xxxix", "rio", "grande", "nbsp"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(general_words)

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = my_stop_words
)

while (data.size > 0):
    print("Applying TFIDF...\n")
    matrix = tfidf.fit_transform(data.setting_value)
    
    for k in range(1, 101, 2):
        print("Clustering with k = {}...".format(k))
        means_clusters = KMeans(n_clusters=k, random_state=20).fit_predict(matrix)
        
        cluster_size = np.bincount(means_clusters)
        print("Clusters sizes = {}".format(cluster_size))
        
        min_sizes = sorted(i for i in cluster_size if i <= 50)
        print("Min cluster sizes = {}\n".format(min_sizes))
        
        if min_sizes:
            rows_removal = []
            counts = Counter(means_clusters)
            print("Counter occurrences = {}\n".format(counts))
            
            for min_size in min_sizes:
                min_element = list(counts.keys())[list(counts.values()).index(min_size)]
                print("Current min_element = {}".format(min_element))
    
                del counts[min_element]                
                print("Removed min_element {} from counter {}\n".format(min_element, counts))
                
                print("Removing smallest cluster elements = {} with occurrences = {}".format(min_element, min_size))
                print("Getting element indexes...")
                min_element_positions = [index for index, value in enumerate(means_clusters) if value == min_element]
                
                rows_removal.extend(min_element_positions)
                
                min_size_position = list(cluster_size).index(min_size) + 1
                print("Getting cluster {} size and top keywords...".format(min_size_position))
                top_keywords = get_top_keywords(matrix, means_clusters, tfidf.get_feature_names(), 10)
                regex = "{}(.*)".format(min_size_position)
                min_cluster_keywords = re.search(regex, top_keywords).group(1)
                print("Top keywords are {}\n".format(min_cluster_keywords))
                
                clusters_size_keywords.append([min_size, min_cluster_keywords])
                
            print("Being removed {} elements...".format(len(rows_removal)))
            
            print("Old data size = {}".format(data.index))
            data = data.drop(data.index[rows_removal]).reset_index(drop=True)
            print("New data size = {}\n".format(data.index))
            
            break
                    
        print("k = {} failed\n".format(k))

#pt.plot_tsne_pca(matrix, means_clusters)
#
#print(get_top_keywords(matrix, means_clusters, tfidf.get_feature_names(), 10))
#
#print("\nClusters Size")
#print(np.bincount(means_clusters))