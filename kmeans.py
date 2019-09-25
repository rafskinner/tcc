import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction import text


data = pd.read_csv('select_no_plural.csv', sep=';', quotechar='"')

wordnet_lemmatizer = WordNetLemmatizer()
#porter_stemmer = PorterStemmer()


def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None


def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:            
      res_words.append(word)
    else:
      res_words.append(wordnet_lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)


i = 0
for setting_value in data['setting_value']:
    data.at[i, 'setting_value'] = lemmatize_sentence(setting_value)
    i += 1

#i = 0
#for setting_value in data['setting_value']:
#    words = nltk.word_tokenize(setting_value)
#    abstract = ''
#    for word in words:
##        stem = porter_stemmer.stem(word)
#        lem = wordnet_lemmatizer.lemmatize(word)
#        abstract += lem + " "
##        abstract += stem + " "
#    data.at[i, 'setting_value'] = abstract.strip()
#    i += 1

general_words = ["data", "use", "using", "used", "paper", "method", "analysis",
                 "based", "result", "problem", "furthermore", "propose",
                 "approach", "present"]

my_stop_words = text.ENGLISH_STOP_WORDS.union(general_words)

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = my_stop_words
)
tfidf.fit(data.setting_value)
matrix = tfidf.transform(data.setting_value)


def find_optimal_clusters(data, max_k):
    iters = range(3, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')


#find_optimal_clusters(matrix, 99)

#clusters = MiniBatchKMeans(n_clusters=19, init_size=1024, batch_size=2048, random_state=20).fit_predict(matrix)
means_clusters = KMeans(n_clusters=17, random_state=20).fit_predict(matrix)


def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=2702, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
#plot_tsne_pca(matrix, clusters)
plot_tsne_pca(matrix, means_clusters)


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i+1))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
#get_top_keywords(matrix, clusters, tfidf.get_feature_names(), 10)
get_top_keywords(matrix, means_clusters, tfidf.get_feature_names(), 10)

print("\nClusters Size")
print(np.bincount(means_clusters))