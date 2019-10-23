import keywords as kw

import re
import numpy as np

from collections import Counter
from sklearn.cluster import KMeans


def iteractive_kmeans(data, tfidf, clusters_size_keywords):
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
                    top_keywords = kw.get_top_keywords(matrix, means_clusters, tfidf.get_feature_names(), 10)
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