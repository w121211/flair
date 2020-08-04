from __future__ import annotations
import datetime
import json
from enum import Enum
from typing import Iterable

import elasticsearch
from elasticsearch_dsl import connections, Document, Date, Keyword, Q, Search, Text, Range, Integer, Boolean


class Lang(Enum):
    EN = "EN"
    ZH = "ZH"


class Entity(Document):
    wiki_id = Keyword()  # nullable (rare case)
    symbol = Keyword()  # either ticker, tag, null


class Doc(Document):
    page_id = Keyword()  # in es index news-page
    title = Text()
    summary = Text()
    text = Text()  # array
    lang = Keyword()
    extracted_event_ids = Keyword()
    extracted_entity_ids = Keyword()  # array
    user_event_ids = Keyword()
    user_entity_ids = Keyword()

    class Index:
        name = "learning-flair"
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0
        }


class Event(Document):
    started_at = Date()
    ended_at = Date()
    count_ticks = Integer()  # array
    similar_event_ids = Integer()  # array
    # 隨時間而變化的entities？

    class Index:
        name = "learning-flair"
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0
        }


# class Page(Document):
#     from_url = Keyword(required=True)
#     resolved_url = Keyword()
#     http_status = Integer()
#     entry_urls = Keyword()  # array
#     entry_tickers = Keyword()  # array
#     entry_title = Text()
#     entry_summary = Text()
#     entry_published_at = Date()
#     entry_meta = Text()
#     article_metadata = Text()
#     article_published_at = Date()
#     article_title = Text()
#     article_text = Text()
#     article_html = Text()  # clean_top_node
#     parsed = Text()  # JSON for flexible data format
#     created_at = Date(required=True)
#     fetched_at = Date()  # null for not-yet-fetched

#     class Index:
#         name = "news_page"
#         settings = {
#             'number_of_shards': 1,
#             'number_of_replicas': 0
#         }

#     def save(self, **kwargs):
#         if 'id' not in self.meta:
#             self.meta.id = self.from_url
#         if self.created_at is None:
#             self.created_at = datetime.datetime.now()
#         return super().save(**kwargs)

#     # @classmethod
#     # def is_existed(cls, src_url: str) -> bool:
#     #     s = cls.search()
#     #     s.query = Q({"match": {"src_url": src_url}})
#     #     resp = s.execute()
#     #     if resp.hits.total.value > 0:
#     #         return True
#     #     return False

#     @classmethod
#     def is_fetched(cls):
#         pass

#     @classmethod
#     def get_or_create(cls, url) -> Page:
#         try:
#             page = cls.get(id=url)
#         except elasticsearch.NotFoundError:
#             page = cls(from_url=url)
#             page.save()
#         return page

#     @classmethod
#     def scan_urls(cls, domain: str) -> Iterable[str]:
#         s = cls.search()
#         q = Q('wildcard', from_url=f'*{domain}*') \
#             & ~Q("term", http_status=200)
#         for page in s.filter(q).scan():
#             yield page.from_url


# def scan_twint(user: str,
#                since="2000-01-01 00:00:00",
#                until="2025-01-01 00:00:00"):
#     q = Q({
#         "range": {
#             "date": {
#                 "gte": since,
#                 "lt": until,
#             }
#         }
#     })
#     client = elasticsearch.Elasticsearch(['es:9200'])
#     s = Search(using=client, index="twinttweets").query(
#         q).filter("terms", username=[user])
#     return s.scan()


def init():
    connections.create_connection(hosts=['es:9200'])
    Page.init()
