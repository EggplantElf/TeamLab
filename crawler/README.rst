Google News Crawler
===================

A utility to fetch news articles from `Google News`_.

GNC retrieves the latest items from the Google News feeds and stores
them in ElasticSearch_ or on disk.

Written by Isaac Sijaranamual, copyright 2013 University of Amsterdam/ILPS_,
licensed under the `Apache License, Version 2.0`_.

.. _`Google News`: http://news.google.com/
.. _ILPS: http://ilps.science.uva.nl/
.. _`Apache License, Version 2.0`: https://www.apache.org/licenses/LICENSE-2.0
.. _ElasticSearch: http://www.elasticsearch.org/


Installation
------------

Google News Crawler can be installed with ``pip`` as usual::

    pip install google_news_crawler


Usage
-----

Retrieve news items belonging to the 'science/technology' topic for
the region Botswana from Google News, storing the articles in an
ElasticSearch instance::

    google_news_crawler --datastore=ES --feed="http://news.google.com/news?cf=all&ned=en_bw&output=rss&topic=t&sort=newest"

You would typically want to run a command like the one above in a
``crontab`` to periodically fetch all the items::

    # m h  dom mon dow   command
    01-59/10 * * * * google_news_crawler --log-config=/path/to/gnc/logging.yaml --datastore=ES --feed="http://news.google.com/news?cf=all&ned=en_bw&output=rss&topic=t&sort=newest"

The complete list of usage options can be obtained with the ``--help``
argument::

    google_news_crawler --help


Nota Bene
---------

The store-to-disk backend is still available, but has been dropped as
a dependency because of a license incompatibility since warc_ licensed
under the GPL (version 2).

.. _warc: https://pypi.python.org/pypi/warc


TODO
----

* general

  * make user-agent configurable
  * expand documentation

* Elasticsearch backend

  * set up proper index mapping for the documents
  * make all ES related settings conigurable
  * update metadata for retrieved documents instead of skipping them
    entirely
