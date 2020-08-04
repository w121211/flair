import dataclasses
import json
from typing import List

from elasticsearch_dsl import Q

# from . import es

# doc(paragraph > article) -> tickers[]
data_folder = '/workspace/flair/data/@me-news-tickers'

# doc(description > article) -> tags[]


def related_tags(ticker):
    """給ticker，返回相關的tags（依照？？？排序）"""
    symbols = dict()
    symbols["AAA"] = []  # (ticker, tag)


def related_links(ticker):
    pass


def related_events(ticker):
    pass


def all_ticers():
    pass


def all_tags():
    pass


def generate_dataset():
    pass


# @dataclasses.dataclass
# class Ticker:
#     name: str
#     corr_tickers: List[str] = dataclasses.Field(default_factory=list)
#     corr_tags: List[str] = dataclasses.Field(default_factory=list)
#     mentioned_docs: List[str] = dataclasses.Field(default_factory=list)
ticker_template = {
    "count": 10,
    "co_tags": {
        "tag1": 2,
        "tag2": 10,
    },
    "co_tickers": {
        "tag1": 2,
        "tag2": 10,
    },
    # "docs": ["id1", "id2"],
    # "text": ["text 1 ....", "..."],
}

tickers = {}
tags = {}

ticker_name = {
    "AAA": {"AAA Company", "AAA Comp."},
}

ticker_count = {}
keyword_count = {}
ticker_name = {}


def add(tickers: List[str], keywords: List[str]) -> None:
    for tk in tickers:
        if tk not in ticker_count:
            ticker_count[tk] = {"co_tickers": {}, "co_keywords": {}}
        for ctk in tickers:
            try:
                ticker_count[tk]["co_tickers"][ctk] += 1
            except KeyError:
                ticker_count[tk]["co_tickers"][ctk] = 1
        for ckw in keywords:
            try:
                ticker_count[tk]["co_keywords"][ckw] += 1
            except KeyError:
                ticker_count[tk]["co_keywords"][ckw] = 1

    for kw in keywords:
        if kw not in keyword_count:
            keyword_count[kw] = {"co_tickers": {}, "co_keywords": {}}
        for ctk in tickers:
            try:
                keyword_count[kw]["co_tickers"][ctk] += 1
            except KeyError:
                keyword_count[kw]["co_tickers"][ctk] = 1
        for ckw in keywords:
            try:
                keyword_count[kw]["co_keywords"][ckw] += 1
            except KeyError:
                keyword_count[kw]["co_keywords"][ckw] = 1


def one_time_proc():
    es.init()

    domain = "cnbc"

    s = es.Page.search()
    # q = Q('wildcard', from_url=f'*{domain}*') \
    #     & Q("term", http_status=200)
    q = Q('wildcard', resolved_url=f'*{domain}*') \
        & Q("term", http_status=200)

    docs = []
    labels = []
    tickers = []

    for i, page in enumerate(s.filter(q).scan()):
        if i > 1000:
            break
        print(i)

        try:
            parsed = json.loads(page.parsed)
            _tickers = []
            for e in parsed["tickers"]:
                for label in e["labels"]:
                    name, ticker = label
                    try:
                        ticker_name[ticker].add(name)
                    except:
                        ticker_name[ticker] = set([name])
                    _tickers.append(ticker)
            add(_tickers, parsed["keywords"])
        except KeyError:
            pass

    # tk = Ticker(name="aaa")
    # print(ticker_count)

    def serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        return obj

    with open('ticker_count.json', 'w', encoding='utf-8') as f:
        json.dump(ticker_count, f, ensure_ascii=False, default=serializer)
    with open('keyword_count.json', 'w', encoding='utf-8') as f:
        json.dump(keyword_count, f, ensure_ascii=False,  default=serializer)
    with open('ticker_name.json', 'w', encoding='utf-8') as f:
        json.dump(ticker_name, f, ensure_ascii=False,  default=serializer)


if __name__ == '__main__':
    print('start')
    one_time_proc()
