"""Google News Crawler.

Module for fetching news items from Google News by fetching its RSS
feed(s).

"""
import sys
from urlparse import urlsplit, urlunsplit, parse_qs, parse_qsl
import datetime
import logging
from urllib import urlencode

import requests
import feedparser
from lxml import html
import pytz

from util import now


LOG = logging.getLogger(__name__)


class GoogleBlockedUsException(Exception):
    """Gluttony _is_ one of the deadly sins... Better backoff or you might
    get banned (do they even do that? Wouldn't suprise me.)

    """
    pass


class GoogleNewsCrawler(object):
    """Crawler for Google News.

    Crawls the RSS item feed at `name` and stores the items (entries)
    in the supplied `datastore`.

    """
    def __init__(self, name, datastore, user_agent=None):
        self.name = name
        self.datastore = datastore
        if user_agent is None:
            # copy-n-paste of the User-Agent header my chromium
            # browser spit out (on debian/unstable, 2013-04-22ish)
            self.user_agent = ("Mozilla/5.0 (X11; Linux x86_64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/32.0.1700.123 Safari/537.36")
        else:
            self.user_agent = user_agent

        # seems very global to me...
        feedparser.USER_AGENT = self.user_agent

    @staticmethod
    def link_cleanup(link):
        """
        Item links in Google News RSS feeds have the following form:
        http://news.google.com/news/url?...&url=<realurl>&...

        We extract and return the <realurl> part.
        """
        return parse_qs(urlsplit(link).query)['url'][0]

    @staticmethod
    def get_feed_entry_metadata(entry):
        """Assemble a dict of key value pairs of metadata for the given
        `entry`"""
        md = {}

        md['retrieved'] = now().isoformat()

        for f in ['published', 'updated', 'created']:
            alt = "%s_parsed" % f
            if alt in entry:
                # tstamp = time.mktime(entry[alt])
                # dt = datetime.datetime.fromtimestamp(tstamp)
                sex_tuple = entry[alt][:6]
                dt = datetime.datetime(*sex_tuple)
                dt = dt.replace(tzinfo=pytz.utc)
                dt_formatted = dt.isoformat()
                md[f] = dt_formatted

        # GN's classification
        md['tags'] = [tag.term for tag in entry['tags']]

        # copy specific fields from the feed's entry
        md.update({f: entry[f] for f in ['author', 'license', 'publisher',
                                         'title', 'guid'] if f in entry})

        return md

    def download_article(self, entry, metadata=None):
        """Downloads and stores this item.

        First checks if the datastore already contains this item
        (based on its url), if not downloads the item and stores it in
        the datastore.

        """
        url = GoogleNewsCrawler.link_cleanup(entry.link)

        if self.datastore.contains(url):
            # we already have this article, skip it...
            LOG.debug("skipping existing item: %s", url)
            return

        request_headers = {"User-Agent": self.user_agent}
        session = requests.Session()
        try:
            res = session.get(url, headers=request_headers, timeout=5)
            article_data = res.text
        except:
            t, v, tb = sys.exc_info()
            LOG.warn("ignoring exception %s for %s", t, url)
            return

        LOG.debug("storing new item: %s", url)
        if metadata is None:
            metadata = {}

        metadata['url'] = url
        if res.url != url:
            # Store the final redirect url if we encounter one (or
            # more) redirect(s) while fetching the article.
            metadata['final_url'] = res.url

        metadata.update(GoogleNewsCrawler.get_feed_entry_metadata(entry))

        try:
            self.datastore.store(article_data, url, metadata=metadata)
        except:
            t, v, tb = sys.exc_info()
            LOG.warn("failed to store %s, ignoring exception %s", url, t)
            raise

    @staticmethod
    def _get_feed(url):
        feed = feedparser.parse(url)
        if feed.status != 200:
            LOG.warn('uh-oh feed status is not 200 (but %d)', feed.status)
            if '/sorry/' in feed.href:
                # Uh-oh, google is probably asking us to solve a capcha
                raise GoogleBlockedUsException()

        return feed

    def process_feed(self):
        """Crawls the Google News RSS feed.

        This performs a two-stage retrieval:
        - first stage (category feed):
          - retrieve linked article
          - extract url of second level feed
        - second stage (article cluster):
          - retrieve all linked articles

        """
        # process first level
        # FYI: tacking on the parameter '&num=100' to the feed url will get you
        # blocked sooner rather than later...
        first = GoogleNewsCrawler._get_feed(self.name)
        LOG.debug("  1st: %s (%s)", first.feed.title, self.name)
        for entry in first.entries:
            # self.download_article(entry)

            # get second level feed
            second_name, ncl = GoogleNewsCrawler.extract_url(entry.description)
            # process second level
            second = GoogleNewsCrawler._get_feed(second_name)
            LOG.debug("    2nd: %s (%s)", entry.title, second_name)

            metadata = {
                'cluster_title': entry.title,
                'cluster_url': second_name,
                'ncl': ncl
            }
            for entry in second.entries:
                self.download_article(entry, metadata=metadata)

    @staticmethod
    def _get_click_through_link(description):
        """
        The click through link is the last <a> in the 'description' field
        of a Google News feed entry.
        """
        html_fragment = html.fromstring(description)
        elt = html_fragment.xpath('(//a)[last()]')[0]
        url = elt.attrib['href']

        return url

    @staticmethod
    def extract_url(description):
        """NB: there is a `num` parameter down below which controls how many
        results Google will yield for the feed. I don't know what the
        safe limit is, but the default value of 10 seems to be low
        enough to stay below Google Radar(tm).

        """
        # get link to all news items in the cluster for this entry
        url = GoogleNewsCrawler._get_click_through_link(description)

        # contruct and return the RSS feed url for that cluster
        parsed = urlsplit(url)
        queries = parse_qsl(parsed.query)
        ncl = [tup[1] for tup in queries if tup[0] == 'ncl'][0]
        new_queries = [('cf', 'all'), ('output', 'rss'), ('num', 10)] + queries

        new_tuple = (parsed.scheme, parsed.netloc, '/news',
                     urlencode(new_queries), parsed.fragment)
        new_url = urlunsplit(new_tuple)

        return new_url, ncl
