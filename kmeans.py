import lemmatize as lem
#import pca_tsne as pt
import clustering as cl

import pandas as pd

#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn import metrics
#from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


clusters_size_keywords = []
kmeans_size_keywords = []

data = pd.read_csv('select_no_plural.csv', sep=';', quotechar='"')
data = lem.perform_lemmatize_dataset(data)

general_words = ["data", "use", "using", "used", "paper", "method", "analysis", "area",
                 "proper", "total", "different", "based", "result", "problem", "furthermore",
                 "propose", "important", "general", "approach", "present", "aim", "work",
                 "make", "goal", "exist", "like", "new", "12", "xxxix", "rio", "grande", "nbsp"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(general_words)

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = my_stop_words
)

cl.iteractive_kmeans(data, tfidf, clusters_size_keywords)
iteractive_kmeans_kw = []
for csk in clusters_size_keywords:
    iteractive_kmeans_kw.append("'" + csk[1] + "'")

### READS SAVED RESULT FROM ITERACTIVE CLUSTERIZATION
#buffered_iteractive_kmeans = open("iteractive_kmeans_data.txt").read().splitlines()
#iteractive_kmeans_kw = []
#for bik in buffered_iteractive_kmeans:
#    iteractive_kmeans_kw.append(bik[bik.find("'"):-1])

cl.conventional_kmeans(data, tfidf, kmeans_size_keywords, 72)
conventional_kmeans_kw = []
for ksk in kmeans_size_keywords:
    conventional_kmeans_kw.append("'" + ksk[1] + "'")

print("\nComparing clusters from both approaches...")
for kw in conventional_kmeans_kw:
    if kw in iteractive_kmeans_kw:
        print("Cluster Keywords match!\n{}\n".format(kw))




#matrix = tfidf.fit_transform(data.setting_value)
#kmeans_model = KMeans(n_clusters=72, random_state=20).fit(matrix)
#labels = kmeans_model.labels_
#silhouette = metrics.silhouette_samples(matrix, labels, metric='euclidean')
#print(silhouette)




#matrix = tfidf.fit_transform(data.setting_value)
#km = KMeans(n_clusters = 72, random_state = 20).fit(matrix)
#dists = euclidean_distances(km.cluster_centers_)
#print(dists)





#pt.plot_tsne_pca(matrix, means_clusters)