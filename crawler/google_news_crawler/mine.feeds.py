#!/usr/bin/env python

import sys
import os
from os.path import join
from lxml import html
from urlparse import urlparse, urljoin
from collections import namedtuple, defaultdict
import re
import warc

Feed = namedtuple('Feed', 'domain, url')

# Lots of links to things that look like an rss feed are actually a
# human-readable description of what RSS is. We filter those
# descriptions.
exception_fragments = """
/hln/nl/1441/rss/
/vk/nl/2764/rss/
/ad/nl/1401/home/
www.standaard.be/Info.aspx
www.rtl.nl/components/actueel/rtlnieuws/services/RSS_Feeds.xml
/dm/nl/1361/rss/
/parool/nl/106/SERVICE/
www.nu.nl/socialtools.html
"""

RSS_EXCEPTIONS = "(?:" + "|".join(fragment for fragment in exception_fragments.split("\n") if fragment) + ")"


def get_base(doc_url, doc):
    try:
        url = doc.xpath('(/head/base)[1]')[0].attrib["href"]
    except:
        return doc_url

    url = urljoin(doc_url, url)

    return url


def mine_feeds(fname, doc_uri):
    doc = html.parse(fname)
  
    base = get_base(doc_uri, doc)

    feeds = set([])

    for feed in doc.xpath('//*[@type="application/rss+xml"] | //*[@type="application/atom+xml"]'):
        try:
            href = feed.attrib["href"]
        except KeyError:
            continue

        abs_url = urljoin(base, href)
        domain = urlparse(abs_url).netloc
    
        feeds.add(Feed(domain, abs_url))

    for a in doc.xpath('//a'):
        try:
            text = unicode(a.xpath('string()'))
        except UnicodeDecodeError:
            # skip this mofo
            continue

        if re.search(r'\brss\b', text, re.IGNORECASE):
            title = a.attrib["title"] if "title" in a.attrib else text
            feed_type = a.attrib["type"] if "type" in a.attrib else ""
            href = a.attrib["href"]

            abs_url = urljoin(base, href)
            # www.hln.be crap
            if re.search(RSS_EXCEPTIONS, abs_url):
                continue
            domain = urlparse(abs_url).netloc
    
            feeds.add(Feed(domain, abs_url))

    # baggertelegraaf
    for elt in doc.xpath('//span[@onclick]'):
        text = elt.attrib["onclick"]
        m = re.search(r"self.location\s*=\s*'/feeds/\?rss=([^']*)'", text, re.IGNORECASE)
        if m:
            abs_url = 'http://www.telegraaf.nl/rss/' + m.group(1)
            # abs_url = urljoin(base, url)
            domain = urlparse(abs_url).netloc
            f = Feed(domain, abs_url)
            feeds.add(f)

    return feeds


def get_warc_uri(fname):
    uri = ""
    f = None
    try:
        f = warc.WARCFile(fname, "r")
        for record in f:
            try:
                uri = record['WARC-Target-URI'].strip()
            except KeyError:
                continue
            if uri:
                break
    finally:
        f.close()

    return uri

all_feeds = set([])
all_feed_urls = set([])
per_domain = defaultdict(int)
per_feed_url = defaultdict(int)

for root, dirs, files in os.walk(sys.argv[1]):
    for fname in files:
        path = join(root, fname)

        doc_uri = get_warc_uri(path)
        feeds = mine_feeds(path, doc_uri)

        for feed in feeds:
            domain = feed.domain
            feed_url = feed.url

            per_domain[domain] += 1
            per_feed_url[feed_url] += 1

            all_feeds.add(feed)
            all_feed_urls.add(feed.url)
        # print "\n".join("\t".join(feed) for feed in feeds)


for feed_url, count in sorted(per_feed_url.items(), key=lambda t:t[1], reverse=True):
    print "\t".join([str(count), feed_url])
