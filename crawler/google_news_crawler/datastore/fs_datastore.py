import os
import errno

from urlparse import urlsplit, urlunsplit
import posixpath
import warc

from . import LOG
from base_datastore import BaseDatastore
from util import now


class FileSystemDatastore(BaseDatastore):
    """Datastore which stores web pages directly on disk as WARC files.

    Retians the original web url in the paths to the stored files.

    """
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def contains(self, url):
        """Check whether or not we already downloaded this article url
        relative to a archive root.

        """
        return os.path.exists(self.url_to_path(url))

    def url_to_path(self, url):
        """Converts a url to a valid path name relative to `base_dir`."""

        lst = list(urlsplit(url))
        # unset the 'scheme' part of the url ('protocol' in tcp lingo)
        lst[0] = ''
        output_path = os.path.join(self.base_dir,
                                   urlunsplit(lst).strip('/'))
        return output_path

    def store(self, data, url, metadata=None, **kwargs):
        output_path = self.url_to_path(url)
        dirname, _ = posixpath.split(output_path)

        FileSystemDatastore.mkdir_p(dirname)
        FileSystemDatastore._write_warc(output_path, data, metadata=metadata)

    @staticmethod
    def mkdir_p(path):
        """Python equivalent of `mkdir -p`"""
    # This method is taken from http://stackoverflow.com/a/600612, user
    # tzot's answer to "mkdir -p functionality in python"
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

    @staticmethod
    def _write_warc(fname, data, metadata=None, warc_date=None):
        """Create an uncompressed WARC file named 'fname' to store the
        (downloaded) data 'data'.

        If 'metadata' is not None then we will also create a WARC metadata
        record, storing a subset of the metadata we got from the feed.

        `warc_date`: datetime to record in the warc, defaults to
        `now()`

        """
        if warc_date is None:
            warc_date = now()

        with open(fname, "w") as warc_file:
            out = warc.WARCFile(fileobj=warc_file)

            warc_date = warc_date.isoformat()

            headers = {
                "WARC-Type": "resource",
                # let the warc library generate a uuid...
                # "WARC-Record-ID": fname,
                "WARC-Date": warc_date
            }

            try:
                headers['WARC-Target-URI'] = metadata['url']
            except KeyError:
                LOG.warn("no url key in metadata dict, "
                         "cannot create WARC-Target URI header")
                pass

            warc_resource = warc.WARCRecord(payload=data.encode("utf-8"),
                                            headers=headers, defaults=True)

            if metadata is not None:
                resource_id = warc_resource.header.record_id

                md_headers = {
                    'WARC-Type': 'metadata',
                    'WARC-Date': warc_date,
                    'WARC-Refers-To': resource_id
                    }

                res = []
                for k, v in metadata.items():
                    if isinstance(v, list):
                        res.extend("%s:%s" % (k, e) for e in v)
                    else:
                        res.append("%s:%s" % (k, v))

                res = "\n".join(res)

                warc_metadata = warc.WARCRecord(payload=res.encode("utf-8"),
                                                headers=md_headers)
                out.write_record(warc_metadata)

            out.write_record(warc_resource)
            out.close()
