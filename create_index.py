
import langchain
from langchain.vectorstores import FAISS

import os

#os.environ["openai_secret_key"] == st.secrets["openai_secret_key"]

from langchain.document_loaders import WebBaseLoader
urls = []
urls.append('https://en.wikipedia.org/wiki/Taylor_Swift')
urls.append('https://www.grammy.com/artists/taylor-swift/15450')
loader = WebBaseLoader(urls)

data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print (f'Now you have {len(texts)} documents')

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Download OpenAI embeddings and map document chunks to embeddings in vector db
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss_index")