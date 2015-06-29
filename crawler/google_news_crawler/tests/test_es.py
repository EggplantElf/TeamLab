from ..datastore.es_datastore import ElasticSearchDatastore


def test_es():
    index = 'test_docs'
    data = '<p>abc</p>'
    url = 'http://www.example.com/index.html'

    try:
        esd = ElasticSearchDatastore(index)
        esd.es.indices.create(index)
        esd.store(data, url)

        assert esd.contains(url)
    except:
        raise
    finally:
        esd.es.indices.delete(index)
