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


def _format_envios(item: Dict[str, Any]) -> str:
    subtipo = item.get('subtipo', '')
    detalle = item.get('detalle', {})

    content = f"INFORMACIÓN DE ENVÍOS - {subtipo.upper()}\n"

    if subtipo == 'aéreo':
        content += f"Origen: {detalle.get('origen', 'N/A')}\n"
        content += f"Días de salida desde Miami: {', '.join(detalle.get('dias_salida', []))}\n"
        content += f"Días de entrega en Asunción: {', '.join(detalle.get('entregas_asuncion', []))}\n"
        content += f"Tiempo estimado de entrega: {detalle.get('tiempo_estimado', 'N/A')}\n"
    elif subtipo == 'marítimo':
        content += f"Tiempo estimado de entrega: {detalle.get('tiempo_estimado', 'N/A')}\n"

    return content


def _format_horarios(item: Dict[str, Any]) -> str:
    return (f"HORARIOS DE ATENCIÓN\n"
            f"Ubicación: {item.get('ubicacion', 'N/A')}\n"
            f"Horario: {item.get('horario', 'N/A')}\n")


def _format_tarifas(item: Dict[str, Any]) -> str:
    modalidad = item.get('modalidad', '')
    content = f"TARIFAS DE ENVÍO - {modalidad.upper()}\n"

    if modalidad == 'aéreo':
        content += f"Peso mínimo: {item.get('peso_minimo', 'N/A')}\n"
        content += f"Precio base mínimo: ${item.get('precio_minimo', 'N/A')} USD/kg\n"
        content += "Tarifas por sucursal (USD/kg):\n"
    elif modalidad == 'marítimo':
        content += f"Peso mínimo: {item.get('peso_minimo', 'N/A')}\n"
        content += f"Precio base mínimo: ${item.get('precio_minimo', 'N/A')} USD/kg\n"
        content += "Tarifas por sucursal (USD/kg):\n"

    sucursales = item.get('sucursales', {})
    for sucursal, precio in sucursales.items():
        content += f"  - {sucursal}: ${precio} USD/kg\n"

    return content


def _format_nota(item: Dict[str, Any]) -> str:
    content = "NOTAS IMPORTANTES:\n"
    detalles = item.get('detalle', [])
    for i, detalle in enumerate(detalles, 1):
        content += f"{i}. {detalle}\n"
    return content


def _format_faq(item: Dict[str, Any]) -> str:
    pregunta = item.get('pregunta', '')
    respuesta = item.get('respuesta', '')
    return f"PREGUNTA FRECUENTE:\nP: {pregunta}\nR: {respuesta}\n"


def _process_json_to_documents(json_path: str) -> List[Document]:
    documents = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if item.get('tipo') == 'envíos':
                content = _format_envios(item)
            elif item.get('tipo') == 'horarios':
                content = _format_horarios(item)
            elif item.get('tipo') == 'tarifas':
                content = _format_tarifas(item)
            elif item.get('tipo') == 'nota':
                content = _format_nota(item)
            elif item.get('tipo') == 'FAQ':
                content = _format_faq(item)
            else:
                content = json.dumps(item, ensure_ascii=False, indent=2)

            doc = Document(
                page_content=content,
                metadata={
                    'tipo': item.get('tipo', 'general'),
                    'tags': item.get('tags', ''),
                    'modalidad': item.get('modalidad', ''),
                    'source': json_path
                }
            )
            documents.append(doc)

    except Exception as e:
        logger.error(f"Error procesando JSON: {e}")

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
                logger.info(f"Se agrego el doc (tipo: {doc.metadata.get('tipo')}) con score {score}")

            return results

        except Exception as e:
            logger.error(f"Error al recuperar los documentos: {e}")
            return []
