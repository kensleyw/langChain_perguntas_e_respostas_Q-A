from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import chroma

def question_answer(file_path, query, chain_type, n_chunks):
    # carrega documento
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # divide o documento em n chunks
    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spliter.split_documents(documents)
    
    # seleciona qual embedding queremos usar
    embeddings = OpenAIEmbeddings()
    
    # cria cria o vectorstore para usar como um indice
    db = chroma.Chroma.from_documents(texts, embeddings)
    
    #expoe esse indice em uma interface de recuperação
    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k' : n_chunks})
    
    #cria uma cadeia para responder perguntas
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type=chain_type,
                                     retriever=retriever, return_source_documents=True)
    
    result = qa({"query" : query})
    
    return result
    