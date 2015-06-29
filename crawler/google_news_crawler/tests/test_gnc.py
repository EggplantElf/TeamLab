import os
from unittest import skip

import feedparser

from google_news_crawler.gnc import GoogleNewsCrawler


with open(os.path.join(os.path.dirname(__file__),
                       "news?cf=all&ned=nl_nl&hl=nl&output=rss&topic=h&"
                       "sort=newest")) as f:
    first_level_feed = f.read()


def test_gnc():
    feed = feedparser.parse(first_level_feed)
    desc = feed.entries[0].description

    expected = ('http://news.google.com/news/story?'
                'ncl=dUTYRKrdHfmCJuM_r0EncAhyE_yzM&ned=nl_nl&topic=h')
    actual = GoogleNewsCrawler._get_click_through_link(desc)

    assert expected == actual
