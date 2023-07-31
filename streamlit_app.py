import streamlit as st
from streamlit_chat import message
import langchain
import OpenAI
import tiktoken
import unstructured
import faiss_cpu
import pytesseract
import openai
import os

os.environ["openai_secret_key"] == st.secrets["openai_secret_key"]

from langchain.chains.summarize import load_summarize_chain

def generate_response2(query, db):
  docs = db.similarity_search(query, k=20)
  chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=QUESTION_PROMPT1, combine_prompt=COMBINE_PROMPT1)
  answer = chain.run(input_documents=docs, question=query)
  answer.replace ('. ', '.\n')
  return answer

from langchain.document_loaders import UnstructuredURLLoader
urls = []
urls.append('https://en.wikipedia.org/wiki/Taylor_Swift')
urls.append('https://www.grammy.com/artists/taylor-swift/15450')
loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print (f'Now you have {len(texts)} documents')

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

# Download OpenAI embeddings and map document chunks to embeddings in vector db
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
llm = OpenAI(temperature=0)

from langchain.prompts import PromptTemplate

question_prompt_template1 = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
Return any relevant text in English.
{text}
Question: {question}
Relevant text, if any, in English:"""
QUESTION_PROMPT1 = PromptTemplate(
    template=question_prompt_template1, input_variables=["text", "question"]
)

combine_prompt_template1 = """Given the following extracted parts of a long document and a question, create a 1000 word essay with references ("SOURCES") for each sentence.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Respond in English.

QUESTION: {question}
=========
{text}
=========
FINAL ANSWER IN English:"""
COMBINE_PROMPT1 = PromptTemplate(
    template=combine_prompt_template1, input_variables=["text", "question"]
)

# We will get the user's input by calling the get_text function
def get_text():
  input_text = st.text_input("You: ", "Hello, how are you?", key="input")
  return input_text

import streamlit as st
from streamlit_chat import message

st.title("chatBot : Streamlit + openAI")

if 'generated' not in st.session_state:
  st.session_state['generated'] = []

if 'past' not in st.session_state:
  st.session_state['past'] = []

user_input = get_text()


if user_input:
  #output = generate_response(user_input)
  # store the output
  output = generate_response2(user_input, db)
  st.session_state.past.append(user_input)
  st.session_state.generated.append(output)

if st.session_state['generated']:
  for i in range(len(st.session_state['generated']) - 1, -1, -1):
    message(st.session_state["generated"][i], key=str(i))
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
