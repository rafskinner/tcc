import lemmatize as lem
import clustering as cl

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


clusters_size_keywords = []
kmeans_size_keywords = []

data = pd.read_csv('select.csv', sep=';', quotechar='"')
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

#cl.iteractive_kmeans(data, tfidf, clusters_size_keywords, 50)

### READS SAVED RESULT FROM ITERACTIVE CLUSTERIZATION SAVING TIME
#buffered_iteractive_kmeans = open("iteractive_kmeans_data.txt").read().splitlines()
#iteractive_kmeans_kw = []
#for bik in buffered_iteractive_kmeans:
#    iteractive_kmeans_kw.append(bik[bik.find("'"):-1])

cl.conventional_kmeans(data, tfidf, kmeans_size_keywords, 72)