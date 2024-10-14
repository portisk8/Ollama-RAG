from flask import Flask, request, jsonify
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

app = Flask(__name__)

# Configuración de los modelos y URL de Ollama
LLAMA_MODEL = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"

# Función personalizada para cargar archivos con manejo de errores de codificación
def load_text_with_error_handling(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    print(f"No se pudo decodificar el archivo {file_path} con ninguna codificación conocida.")
    return ""

def load_json_with_error_handling(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return json.load(file)
        except UnicodeDecodeError:
            continue
    print(f"No se pudo decodificar el archivo {file_path} con ninguna codificación conocida.")
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
template = """Eres un asistente útil y amigable. Utiliza la siguiente información para responder a la pregunta del usuario.
Si no puedes encontrar una respuesta específica en la información proporcionada, di que no lo sabes.

Contexto: {context}

Pregunta: {question}

Respuesta:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Crear la cadena de recuperación y respuesta
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

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No se proporcionó una pregunta"}), 400

    answer, sources = answer_question(question)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
