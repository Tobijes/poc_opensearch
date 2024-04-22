from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

class Searcher:
    def __init__(self, region="us-east-1"):
        es_endpoint = "https://localhost:9200"
        es_username = "admin"
        es_password = "yourStrongPassword123!"

        self.client = OpenSearch(
            hosts=[es_endpoint],
            http_auth=(es_username, es_password),
            verify_certs=False,
            # connection_class = RequestsHttpConnection,
        )
        self.index_name = "image"

    def create_index(self):
        knn_index = {
            "settings": {
                "index.knn": True,
            },
            "mappings": {
                "properties": {
                    "feature_vector": {
                        "type": "knn_vector",
                        "dimension": 512,
                    }
                }
            },
        }
        return self.client.indices.create(index=self.index_name, body=knn_index, ignore=400)

    def bulk_ingest(self, generate_data, chunk_size=128):
        return bulk(self.client, generate_data, chunk_size=chunk_size)

    def knn_search(self, query_features, k=10):
        body = {
            "size": k,
            "_source": {
                "exclude": ["feature_vector"],
            },
            "query": {
                "knn": {
                    "feature_vector": {
                        "vector": query_features,
                        "k": k,
                    }
                }
            },
        }
        return self.client.search(index=self.index_name, body=body)
