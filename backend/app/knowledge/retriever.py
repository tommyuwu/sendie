import logging
import os
from typing import List

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from ..core.config import settings

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    def __init__(self):
        self.knowledge_base_path = settings.knowledge_base_path
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        try:
            if not os.path.exists(self.knowledge_base_path):
                logger.warning(f"Knowledge base path {self.knowledge_base_path} does not exist")
                return None

            loader = DirectoryLoader(
                self.knowledge_base_path,
                loader_cls=TextLoader,
                glob="**/*.txt"
            )
            documents = loader.load()

            if not documents:
                logger.warning("No documents found in knowledge base")
                return None

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings
            )

            logger.info(f"Knowledge base loaded with {len(splits)} chunks")
            return vectorstore

        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            return None

    def get_relevant_documents(self, query: str, k: int = 3) -> List[str]:
        if not self.vectorstore:
            return []

        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
