import numpy as np
import pandas as pd


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    top_keywords = ""
    
    for i,r in df.iterrows():
        top_keywords += '\nCluster {}'.format(i+1)
        top_keywords += ','.join([labels[t] for t in np.argsort(r)[-n_terms:]])

    return top_keywords