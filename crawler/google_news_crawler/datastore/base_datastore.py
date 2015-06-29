"""BaseDatastore for Google News Crawler"""


class BaseDatastore(object):
    """Abstract base class for storing web pages."""
    def contains(self, url):
        """Returns boolean whether this datastore already contains the data at
        the specified url.
        Override this method in the subclass(es).

        """
        raise NotImplementedError("bla")

    def store(self, data, url, **kwargs):
        """Store the `data` in the datastore using `url` as a key/id/locator.

        Override this method in the subclass(es).

        """
        raise NotImplementedError("blie")
