from flask import Flask, request, jsonify
import os
from typing import List
import json
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import pyodbc
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Importa CORS

app = Flask(__name__)
CORS(app)  # Esto habilitar� CORS para todas las rutas

with open('appsettings.json') as config_file:
    config = json.load(config_file)
    
# Configuration
LLAMA_MODEL = config["Ollama"]["ChatModel"]
EMBEDDING_MODEL = config["Ollama"]["EmbbeddingModel"]
OLLAMA_URL = config["Ollama"]["Url"]
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'md'}
encodings = ['utf-8', 'latin-1', 'ascii']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SQL Server connection
# SQL Server connection
conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={config["Database"]["Server"]};'
    f'DATABASE={config["Database"]["Database"]};'
    f'UID={config["Database"]["User"]};'
    f'PWD={config["Database"]["Password"]}'
)
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Create tables if not exist
cursor.execute('''
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='projects' AND xtype='U')
    CREATE TABLE projects (
        id INT PRIMARY KEY IDENTITY(1,1),
        name NVARCHAR(255),
        system_context NVARCHAR(MAX)
    )
''')
cursor.execute('''
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='documents' AND xtype='U')
    CREATE TABLE documents (
        id INT PRIMARY KEY IDENTITY(1,1),
        project_id INT,
        file_path NVARCHAR(255),
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )
''')
conn.commit()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_document(file_path):
    if file_path.endswith('.pdf'):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith('.json'):
        return load_json(file_path)
    else:
        return TextLoader(file_path).load()

def load_json(file_path):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                 data = json.load(file)
            documents = []
            for item in data:
                if isinstance(item, dict) and 'content' in item:
                    content = item['content']
                    metadata = {k: v for k, v in item.items() if k != 'content'}
                    metadata['source'] = file_path
                    documents.append(Document(page_content=content, metadata=metadata))
            return documents
        except UnicodeDecodeError:
            continue
    print(f"No se pudo decodificar el archivo {file_path} con ninguna codificacion conocida.")
    return None

@app.route('/project', methods=['POST'])
def create_or_update_project():
    data = request.json
    name = data.get('name')
    system_context = data.get('system_context')
    project_id = data.get('project_id')

    if project_id:
        cursor.execute("UPDATE projects SET name = ?, system_context = ? WHERE id = ?",
                       name, system_context, project_id)
    else:
        cursor.execute("INSERT INTO projects (name, system_context) VALUES (?, ?)",
                       name, system_context)
        project_id = cursor.execute("SELECT @@IDENTITY").fetchone()[0]

    conn.commit()
    return jsonify({"project_id": project_id, "message": "Project saved successfully"})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    project_id = request.form.get('project_id')
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Crear la carpeta de destino si no existe
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        cursor.execute("INSERT INTO documents (project_id, file_path) VALUES (?, ?)",
                       (project_id, file_path))
        conn.commit()
        return jsonify({"message": "File uploaded successfully"}), 200
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/train', methods=['POST'])
def train_project():
    project_id = request.json.get('project_id')
    cursor.execute("SELECT file_path FROM documents WHERE project_id = ?", project_id)
    documents = []
    for row in cursor.fetchall():
        documents.extend(load_document(row.file_path))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=f"chroma/DB_{project_id}")

    return jsonify({"message": "Project trained successfully"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get("question", "")
    project_id = data.get("project_id")

    if not question or not project_id:
        return jsonify({"error": "Question and project_id are required"}), 400

    cursor.execute("SELECT system_context FROM projects WHERE id = ?", project_id)
    system_context = cursor.fetchone().system_context

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    vectorstore = Chroma(persist_directory=f"chroma/DB_{project_id}", embedding_function=embeddings)

    llm = Ollama(model=LLAMA_MODEL, base_url=OLLAMA_URL)

    template = f"""You are a helpful assistant. Use the following information to answer the user's question.
If you can't find a specific answer in the provided information, say you don't know.

System Context: {system_context}

Context: {{context}}

Question: {{question}}

Answer:"""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain({"query": question})
    answer = result['result']
    sources = [doc.metadata['source'] for doc in result['source_documents']]

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources
    })

@app.route('/projects', methods=['GET'])
def get_projects():
    try:
        # Crear un nuevo cursor para esta operaci�n
        cursor = conn.cursor()
        
        # Ejecutar la consulta SQL para obtener todos los proyectos
        cursor.execute("""
            SELECT id, name, system_context 
            FROM projects 
            ORDER BY name
        """)
        
        # Obtener todos los resultados
        projects = cursor.fetchall()
        
        # Cerrar el cursor
        cursor.close()
        
        # Convertir los resultados a una lista de diccionarios
        project_list = []
        for project in projects:
            project_list.append({
                'id': project.id,
                'name': project.name,
                'system_context': project.system_context
            })
        
        # Devolver la lista de proyectos como JSON
        return jsonify(project_list), 200
    
    except Exception as e:
        # En caso de error, devolver un mensaje de error
        return jsonify({'error': str(e)}), 500

@app.route('/project/<int:project_id>', methods=['GET'])
def get_project_name(project_id):
    try:
        # Crear un nuevo cursor para esta operaci�n
        cursor = conn.cursor()
        
        # Ejecutar la consulta SQL para obtener el nombre del proyecto seg�n el ID
        cursor.execute("""
            SELECT name
            FROM projects 
            WHERE id = ?
        """, (project_id,))
        
        # Obtener el resultado
        project = cursor.fetchone()
        
        # Cerrar el cursor
        cursor.close()

        # Verificar si se encontr� un proyecto con el ID proporcionado
        if project:
            # Devolver el nombre del proyecto
            return jsonify({'name': project.name}), 200
        else:
            # Si no se encuentra el proyecto, devolver un error 404
            return jsonify({'error': 'Project not found'}), 404

    except Exception as e:
        # En caso de error, devolver un mensaje de error
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)