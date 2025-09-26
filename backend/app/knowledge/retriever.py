import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import nltk
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from nltk.corpus import stopwords

from ..core.config import settings

nltk.download("stopwords")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("spanish"))

logger = logging.getLogger(__name__)


def _flatten_dict(d: Dict[str, Any], indent: int = 0) -> str:
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_flatten_dict(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{prefix}{k}: {', '.join(map(str, v))}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


def _format_default(item: Dict[str, Any]) -> str:
    tipo = item.get("tipo", "general").upper()
    return f"{tipo}\n{_flatten_dict(item)}"


def _process_json_to_documents(json_path: str) -> List[Document]:
    documents = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            tipo = item.get("tipo", "general")

            content = _format_default(item)

            doc = Document(
                page_content=content,
                metadata={
                    "tipo": tipo,
                    "tags": item.get("tags", ""),
                    "modalidad": item.get("modalidad", ""),
                    "source": json_path,
                },
            )
            documents.append(doc)

    except Exception as e:
        logger.error(f"Error procesando JSON {json_path}: {e}")

    return documents


def rewrite_query_simple(query: str) -> str:
    query = query.lower()
    query = re.sub(r"\b(buenas|hola|tardes|días|noches)\b", "", query)
    tokens = nltk.word_tokenize(query)
    filtered = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(filtered)


class KnowledgeRetriever:
    def __init__(self):
        self.knowledge_base_path = settings.knowledge_base_path
        self.knowledge_base_json_file = settings.knowledge_base_json_file
        self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        try:
            if not os.path.exists(self.knowledge_base_path):
                logger.warning(f"La ruta {self.knowledge_base_path} no existe")
                return None

            documents = []

            json_file_path = f"{self.knowledge_base_path}/{self.knowledge_base_json_file}"
            if os.path.exists(json_file_path):
                json_documents = _process_json_to_documents(json_file_path)
                documents.extend(json_documents)
                logger.info(f"Se procesaron {len(json_documents)} elementos del archivo JSON")

            txt_files = Path(self.knowledge_base_path).glob("**/*.txt")
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if len(content) > 1000:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=100
                        )
                        chunks = text_splitter.split_text(content)
                        for chunk in chunks:
                            doc = Document(
                                page_content=chunk,
                                metadata={'source': str(txt_file), 'tipo': 'texto'}
                            )
                            documents.append(doc)
                    else:
                        doc = Document(
                            page_content=content,
                            metadata={'source': str(txt_file), 'tipo': 'texto'}
                        )
                        documents.append(doc)

                except Exception as e:
                    logger.error(f"Error al procesar {txt_file}: {e}")

            if not documents:
                logger.warning("No se encontraron documentos en la base de conocimientos")
                return None

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
            logger.info(f"Se cargaron {len(documents)} documentos en la base de conocimientos")
            return vectorstore

        except Exception as e:
            logger.error(f"Error al iniciar la base de conocimientos: {e}")
            return None

    def get_relevant_documents(self, query: str, k: int = 3) -> List[str]:
        if not self.vectorstore:
            return []

        try:
            docs = self.vectorstore.similarity_search_with_score(rewrite_query_simple(query), k=k)

            results = []
            for doc, score in docs:
                snippet = doc.page_content.strip()
                results.append(snippet)
                logger.info(f"doc: {snippet}")
                logger.info(f"Se agrego el doc (tipo: {doc.metadata.get('tipo')}), con score: {score}")

            return results

        except Exception as e:
            logger.error(f"Error al recuperar los documentos: {e}")
            return []
