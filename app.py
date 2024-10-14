from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

llm = ChatOllama(model="llama3.2") # Modelo de lenguaje

file_path = "c#.pdf" # Ruta del archivo PDF

loader = PyMuPDFLoader(file_path)

data_pdf = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500) # Tamaño de los chunks y el overlap (superposición de los chunks)

chunks = text_splitter.split_documents(data_pdf)

embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_db = "chroma_db_dir" # Directorio donde se guardará la información
collection_db = "chroma_collection" # Nombre de la colección

vs = Chroma.from_documents(
    documents=chunks,
    embedding=embed_model,
    persist_directory=persist_db,
    collection_name=collection_db
)

vectorstore = Chroma(
    embedding_function=embed_model,
    persist_directory=persist_db,
    collection_name=collection_db
)

retriever = vectorstore.as_retriever(
    search_kwargs={'k': 5} # Cantidad de chunks a retornar
)

custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

Contexto: {context}
Pregunta: {question}

Solo devuelve la respuesta útil a continuación y nada más. Responde siempre en español
Respuesta útil:
"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

quest = input("Ingrese su pregunta: ")

resp = qa.invoke({"query": quest})
print(resp['result'])
