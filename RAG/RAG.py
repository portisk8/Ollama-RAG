import os
from typing import List
import json
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

# Configuracion de los modelos y URL de Ollama
LLAMA_MODEL = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"

# Funcion personalizada para cargar archivos con manejo de errores de codificacion
def load_text_with_error_handling(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    print(f"No se pudo decodificar el archivo {file_path} con ninguna codificacion conocida.")
    return ""
def load_json_with_error_handling(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue
    print(f"No se pudo decodificar el archivo {file_path} con ninguna codificacion conocida.")
    return ""

# Loader personalizado
class CustomLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        if self.file_path.endswith('.json'):
            return self.load_json()
        else:  # Asumimos que es un archivo de texto o markdown
            return self.load_text()

    def load_text(self) -> List[Document]:
        text = load_text_with_error_handling(self.file_path)
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

    def load_json(self) -> List[Document]:
        data = load_json_with_error_handling(self.file_path)
        # Asumimos que el JSON tiene una estructura específica. Ajusta según sea necesario.
        documents = []
        for item in data:
            if isinstance(item, dict) and 'content' in item:
                content = item['content']
                metadata = {k: v for k, v in item.items() if k != 'content'}
                metadata['source'] = self.file_path
                documents.append(Document(page_content=content, metadata=metadata))
        return documents

# Cargar documentos
loader = DirectoryLoader("./knowledge_base", glob=["**/*.md", "**/*.json"], loader_cls=CustomLoader)
documents = loader.load()

# Dividir documentos en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Crear embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

# Crear base de datos vectorial
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Configurar el modelo de lenguaje
llm = Ollama(model=LLAMA_MODEL, base_url=OLLAMA_URL)

# Crear un prompt template personalizado
template = """Eres un asistente util y amigable. Utiliza la siguiente informacion para responder a la pregunta del usuario.
Si no puedes encontrar una respuesta especifica en la informacion proporcionada, di que no lo sabes.

Contexto: {context}

Pregunta: {question}

Respuesta:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Crear la cadena de recuperacion y respuesta
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def answer_question(question):
    result = qa_chain({"query": question})
    answer = result['result']
    sources = [doc.metadata['source'] for doc in result['source_documents']]
    return answer, sources

# Interfaz de usuario simple
if __name__ == "__main__":
    print("Bienvenido al asistente RAG. Escribe 'salir' para terminar.")
    while True:
        user_question = input("\nPregunta: ")
        if user_question.lower() == 'salir':
            print("Hasta luego!")
            break
        answer, sources = answer_question(user_question)
        print(f"\nRespuesta: {answer}")
        print("\nFuentes:")
        for source in set(sources):
            print(f"- {source}")