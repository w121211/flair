import dataclasses
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, DocumentRNNEmbeddings
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, MeanShift, AgglomerativeClustering


from .es import Lang

# Docs to cluster (as event)
# off-line clustering docs: doc -> vec -> knn -> clusters(as events)
# save to elasticsearch: for each event
# on-line clustering


@dataclasses.dataclass
class Doc:
    text: str
    vec: List[float]
    published_at: datetime
    lang: Lang


def text_to_doc(text: str):
    # doc embedding
    pass


def extract_entities(doc):
    """use BLINK, currently only EN available"""
    pass


def extract_events(docs: List[str], date: datetime):
    # k-means clustering

    # for each cluster is either majority composed by the same previous cluster members or not
    # extend the previous cluster if yes, create new cluster if not
    return


def count_comentioned():
    """count entities comentioned in a doc, for all docs"""
    # ticker-tag
    # ticker-ticker
    # tag-ticker
    pass


def update_comentioned_count(cur):
    pass


def slide_window_clustering():
    pass


def run_every(pages):
    """
    定期執行，處理一定時期間收集的docs，包括:
    * doc entities
    * event (doc clustering)，包含延續性event
    * entity count tick: 這期間entity出現的次數
    * doc count tick: 這期間的docs數
    * event count tick
    """
    symbol_count = {}  # {symbol_name: count} for ticks
    return symbol_count


def run_onetime(start, end):
    pass


def main():
    df = pd.read_csv("/workspace/flair/data/abcnews-date-text.csv")
    df = df.head(1000)
    df.head

    document_embeddings = DocumentPoolEmbeddings([WordEmbeddings('glove')])

    X = []
    for i, d in enumerate(df["headline_text"]):
        sent = Sentence(d, use_tokenizer=True)
        document_embeddings.embed(sent)
        x = sent.get_embedding()
        x = x.detach().numpy()
        X.append(x)
    X = np.stack(X)

    # 將doc vectors依照日期分成不同的組

    # df.groupby(by="publish_date").agg(["count"])

    groups = df.groupby(by="publish_date")
    groups.groups.keys()
    keys = [20030219, 20030220, 20030221, 20030222, 20030223, 20030224]

    # groups[20030219]
    # groups.get_group(20030219)
    d0 = groups.get_group(keys[0])
    d1 = groups.get_group(keys[1])
    d2 = groups.get_group(keys[2])

    # for i, d in enumerate(df["headline_text"]):
    #     print(i, d)
    #     break

    X0 = X[d0.index[0]:d0.index[-1]+1]
    X1 = X[d1.index[0]:d1.index[-1]+1]
    # X2 = X[d2.index[0]:d2.index[-1]+1]
    # X0_1 = X[d0.index[0]:d1.index[-1]+1]

    clustering = KMeans(n_clusters=50, random_state=0).fit(X0)
    # clustering.predict(X1)
    # y = np.concatenate([c0.labels_, clustering.predict(X1)])
    # c1 = KMeans(n_clusters=45, random_state=0).fit(X1)

    # # z = np.zeros(df.shape[0], dtype=np.int32)
    # z = np.empty((df.shape[0],)) * np.nan
    # z[:c0.labels_.shape[0]] = c0.labels_
    # df["c0"] = z

    # z = np.empty((df.shape[0],)) * np.nan
    # z[:c1.labels_.shape[0]] = c1.labels_
    # df["c1"] = z
    # df.head(100)

    # _df = df[0:len(y)]
    # _df["y"] = y
    # # _df.loc[_df["y"] == 20, ["headline_text", "y"]]
    # _df.groupby("y").agg(["count"])

    df = pd.DataFrame({"a": [4, 5, 6]})
    # df.insert(loc=1, column="b", value=[1,2])
    c = np.full((3,), np.nan).astype(np.uint8)
    c[1:3] = [1, 2]
    df["c"] = c
    df

    prev = [1, 1, 1, np.nan]
    cur = [1, 2, 3, 4]

    cluster = {}
    # docs = pd.DataFrame({"doc" : [})
    df = pd.DataFrame({
        "prev": ["t0-1", "t0-1", "t0-1", "NA", "NA"],
        "cur": [1, 1, 2, 1, 2]
    }, index=[5, 6, 7, 8, 9])

    # g = grouped.get_group(1)
    # list(grouped.groups)
    # grouped
    # print(grouped.get_group(2))

    # df = grouped.get_group(1)
    # print(df)
    # g = grouped.get_group(1).groupby('prev')

    # g.apply(lambda x: )
    # [c for c in g.count()]
    # g.count()

    # len(df)

    # grouped.get_group(2)
    # for g in grouped:
    #     print(g.)

    def set_cluster(df: pd.DataFrame, cur_cluster: int):
        count = {}

        for i, r in df.iterrows():
            try:
                count[r["prev"]].append(i)
            except KeyError:
                count[r["prev"]] = [i]

        # 找出最大的prev_cluster
        cmax = (None, 0)  # (prev_label, count)
        total = 0  # number of prev items
        for k, v in count.items():
            if len(v) > cmax[1]:
                cmax = (k, len(v))
            if k != -1:
                total += len(v)

        print(cmax)

        # 判斷該cluster是否為已建立event的延續
        for i, r in df.iterrows():
            c = cmax[0] if cmax[1] / total > 0.5 else f't1-{cur_cluster}'
            try:
                cluster[i].append(c)
            except KeyError:
                cluster[i] = [c]

    # k = 1
    # g = grouped.get_group(1)
    # df.iloc(1)
    # cluster[0].append(1)
    # cluster.values()

    grouped = df.groupby("cur")
    for k, g in grouped:
        set_cluster(g, cur_cluster=k)

    cluster


if __name__ == "__main__":
    main()
