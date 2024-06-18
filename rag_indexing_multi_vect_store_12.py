import os
from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain.storage.file_system import LocalFileStore
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import ByteStore, InMemoryStore, BaseStore
from langchain.retrievers import MultiVectorRetriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Tuple
import json

doc_id_name = "doc_id"

def set_env():
    def set_proxy():
        # a function to test the connectivity to the proxy
        def get_proxy():
            ora_proxy = 'http://www-proxy-ash7.us.oracle.com:80'
            import requests
            try:
                requests.get(ora_proxy, timeout=1)
                return ora_proxy
            except Exception as e:
                return ""
        proxy = get_proxy()
        os.environ['http_proxy'] = proxy
        os.environ['https_proxy'] = proxy
        os.environ['no_proxy'] = proxy

    from dotenv import load_dotenv
    load_dotenv()
    os.environ['LANGCHAIN_PROJECT'] = 'rang_from_scratch_indexing'
    set_proxy()

def download_docs(doc_list_json:str, num_to_download:int=None)->List[Document]:
    from langchain_community.document_loaders import WebBaseLoader
    with open(doc_list_json, 'r') as f:
        doc_list = json.load(f)
    docs = WebBaseLoader(web_paths=list(doc_list.values())[:num_to_download]).load()
    import uuid
    for doc in docs:
        doc.metadata[doc_id_name] = str(uuid.uuid4())
    return docs

def summarize_docs(docs:List[Document], num_to_summarize:int=None)->List[Document]:
    template = """Please summarize the following document: 
    
    Document title: {title}
    Document description: {description}
    Document text:
    {doc_string}
    """
    summary_prompt = PromptTemplate(name="Summary Prompt", template=template, input_variables=['doc_string', 'title', 'description'])
    to_summary = [{"title": doc.metadata.get('title', ""), "description": doc.metadata.get("description", "") , "doc_string": doc.page_content} for doc in docs]
    summary_chain = summary_prompt | ChatOpenAI(name="OpenAI Summary", model="gpt-3.5-turbo", temperature=0)
    summary_chain.name = "Summary Chain - RAG from scratch - Indexing_multi-vector-store"
    responses = summary_chain.batch(to_summary[:num_to_summarize], {"max_concurrency": 5})
    summaries = []
    for doc, summary in zip(docs, responses):
        summary_doc = Document(page_content=summary.content, metadata=doc.metadata)
        summaries.append(summary_doc)
    return summaries

def create_multi_vectore_retriever(name:str, docs:List[Document], 
                                   summaries:List[str], 
                                   id_key:str=doc_id_name, 
                                   num_to_store:int=None)->MultiVectorRetriever:
    summary_vec_store = Chroma(collection_name="summary",
                          embedding_function=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
                          persist_directory="data/data_store/indexing/summaries")
    doc_store = InMemoryStore()
    retriver = MultiVectorRetriever(name=name,
                         vectorstore=summary_vec_store,
                         docstore=doc_store,
                         byte_store=LocalFileStore(root_path="data/data_store/indexing/docs"),
                         id_key=id_key, k=3)
    doc_id_list = [doc.metadata.get(doc_id_name) for doc in docs[:num_to_store]]
    retriver.vectorstore.add_documents(summaries)
    retriver.docstore.mset(list(zip(doc_id_list, docs[:num_to_store]))) # Setting "docstore" will automatically add files in "byte_store" location
    # retriver.byte_store.mset(list(zip(doc_id_list, [doc.page_content.encode() for doc in docs[:num_to_store]]))) # This is not needed as "docstore" will take care of it

    return retriver

def get_rag_chain(retriever:MultiVectorRetriever)->Chain:
    template = """Please answer question based on the give context.
    
    Context:
    {context}
    
    Question: {question}"""
    prompt = PromptTemplate.from_template(template)
    prompt.input_variables = ['context', 'question']
    prompt.name = "RAG Prompt"
    llm = ChatOpenAI(name="OpenAI LLM", model="gpt-3.5-turbo", temperature=0)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    chain.name = "RAG from scratch - Indexing_multi-vector-store"
    return chain

def main():
    set_env()
    limit_to = 5
    docs = download_docs("data/document_list.json", num_to_download=limit_to)
    summaries = summarize_docs(docs, limit_to)
    multi_vect_retriever = create_multi_vectore_retriever(name="Multi Vector Retriever", 
                                                          docs=docs, 
                                                          summaries=summaries,
                                                          id_key=doc_id_name, 
                                                          num_to_store=limit_to)
    chain = get_rag_chain(multi_vect_retriever)
    response = chain.invoke("What is attention?")
    print(response)

if __name__ == "__main__":
    main()