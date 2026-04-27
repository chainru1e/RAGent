from ragent.models.chunk import Chunk

class Retriever:
    def __init__(self, vectordb, embedder):
        self.vectordb = vectordb
        self.embedder = embedder

    def retrieve(self, query: str) -> list[Chunk]:
        query_vector = self.embedder.embed(query)
        search_results = self.vectordb.hybrid_search(query_vector=query_vector)
        
        return search_results