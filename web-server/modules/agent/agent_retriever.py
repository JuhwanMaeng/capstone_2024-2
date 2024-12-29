import datetime
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class AgentRetriever(BaseRetriever):
    vectorstore: VectorStore # 유사도 검색을 위한 Vectordb
    name : str # Agent별 Retriever
    # 장기기억 score와 관련된 가중치 할당
    alpha : float = Field(default=1) 
    beta : float = Field(default=0.01)
    psi : float = Field(default=0.01) 
    search_kwargs: dict = Field(default_factory=lambda: dict(k=100,)) # 유사도 검색 수행 시 가져올 Memory의 수(Default : k=100)
    index_stage: List[Document] = Field(default_factory=list)
    default_salience: Optional[float] = None
    k: int = 15

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _get_times_passed(time: datetime.datetime, ref_time: datetime.datetime) -> float:
        return (time - ref_time).total_seconds() / 3600

    def _document_get_date(self, field: str, document: Document) -> datetime.datetime:
        if field in document.metadata:
            if isinstance(document.metadata[field], float):
                return datetime.datetime.fromtimestamp(document.metadata[field])
            return document.metadata[field]
        return 0.0

    def _get_combined_score(self, document: Document, vector_relevance: Optional[float], current_time: datetime.datetime) -> float:
        times_passed = self._get_times_passed(current_time, self._document_get_date("last_accessed_at", document))
        search_count = document.metadata.get("search_count", 0)
        importance = document.metadata.get("importance", 1)
        score = self.alpha * importance * search_count * (1 + self.beta * times_passed) ** -self.psi
        if vector_relevance is not None:
            score += vector_relevance
        return score

    def get_salient_docs(self, query: str) -> Dict[int, Tuple[Document, float]]:
        docs_and_scores: List[Tuple[Document, float]]
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(query,namespace = self.name ,**self.search_kwargs)
        results = {}
        for fetched_doc, relevance in docs_and_scores:
            if "buffer_idx" in fetched_doc.metadata:
                buffer_idx = int(fetched_doc.metadata["buffer_idx"])
                doc = self.index_stage[buffer_idx]
                results[buffer_idx] = (doc, relevance)
        return results

    def _get_rescored_docs(self, now : datetime,docs_and_scores: Dict[Any, Tuple[Document, Optional[float]]]) -> List[Document]:
        current_time = now
        rescored_docs = [(doc, self._get_combined_score(doc, relevance, current_time)) for doc, relevance in docs_and_scores.values()]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        for doc, _ in rescored_docs[: self.k]:
            buffered_doc = self.index_stage[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            if "search_count" in buffered_doc.metadata:
                buffered_doc.metadata["search_count"] += 1
            else:
                buffered_doc.metadata["search_count"] = 1
            result.append(buffered_doc)
        return result
    
    def _get_relevant_documents(self, query: str, now : datetime,*, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        docs_and_scores = {doc.metadata["buffer_idx"]: (doc, self.default_salience) for doc in self.index_stage[-self.k :]}
        docs_and_scores.update(self.get_salient_docs(query))
        return self._get_rescored_docs(now, docs_and_scores)
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        current_time = kwargs.get("current_time")
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = current_time
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = current_time
            if "search_count" not in doc.metadata:
                doc.metadata["search_count"] = 0
            doc.metadata["buffer_idx"] = len(self.index_stage) + i
        self.index_stage.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)