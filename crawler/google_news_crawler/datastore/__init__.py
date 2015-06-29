"""Datastore for Google News Crawler"""
import logging


class UnknownDatastoreException(BaseException):
    """We raise this if anybody asks for an unknown datastore type"""
    pass


LOG = logging.getLogger(__name__)
