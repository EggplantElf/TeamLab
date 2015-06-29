#!/usr/bin/env python

"""Google News Crawler

Retrieves news articles from Google News' feed and stores them in
ElasticSearch or on disk.

Usage:
  google_news_crawler --feed=FEED --datastore=DATASTORE [options]

Options:
  -f=FEED, --feed=FEED   Name/url of the Google News feed to crawl
  -d=DS, --datastore=DS  Datastore backend to use, options are: FS, ES
  --log-config=FNAME     YAML file containing the logging configuration
                         [default: logging.yaml]
  -h, --help             This help.
"""

from logging.config import dictConfig

import yaml
from docopt import docopt

from .gnc import GoogleNewsCrawler


def setup_logging(fname):
    with open(fname) as f:
        dictConfig(yaml.load(f))


def main(argv=None):
    version = '0.3.5'
    if argv is None:
        args = docopt(__doc__, version=version)
    else:
        args = docopt(__doc__, version=version, argv=argv)

    setup_logging(args['--log-config'])

    feed_name = args['--feed']
    ds_name = args['--datastore']

    # TODO: DSFactory anyone?
    if ds_name == 'FS':
        from google_news_crawler.datastore.fs_datastore import FileSystemDatastore
        output_dir = 'output'
        ds = FileSystemDatastore(output_dir)
    elif ds_name == 'ES':
        from google_news_crawler.datastore.es_datastore import ElasticSearchDatastore
        index = 'docdocdocs'
        ds = ElasticSearchDatastore(index, create_index=False)
    else:
        from google_news_crawler.datastore import UnknownDatastoreException
        raise UnknownDatastoreException(ds_name)

    crawler = GoogleNewsCrawler(feed_name, ds)
    crawler.process_feed()


if __name__ == '__main__':
    main()
