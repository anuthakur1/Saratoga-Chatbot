import streamlit as st
from streamlit_chat import message
import langchain
from langchain.llms import OpenAI
import tiktoken
import unstructured
from langchain.vectorstores import FAISS
import pytesseract
import openai
import os
import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import re
import langchain
from langchain.document_loaders import WebBaseLoader

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)

class Crawler:
    
    def get_domain(self, url):
        """Gets the domain of a URL.

        Args:
            url (str): The URL.

        Returns:
            str: The domain of the URL.
        """

        match = re.match(r"https?://(.[^/:]+)(:\d+)?/?.*", url)
        if match:
            return match.group(1)
        else:
            return None
    
    def is_same_domain(self, url1, url2):
        """Checks if two URLs have the same domain.

        Args:
          url1 (str): The first URL.
         url2 (str): The second URL.

        Returns:
          bool: True if the two URLs have the same domain, False otherwise.
        """

        domain1 = self.get_domain(url1)
        domain2 = self.get_domain(url2)

        return domain1 == domain2

    def __init__(self, urls=[]):
        self.visited_urls = []
        self.urls_to_visit = urls
        self.count = 0

    def download_url(self, url):
        print("attempting to download " + url)
        return requests.get(url).text

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path
            
    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def crawl(self, url):
        print("crawling url " + url)
        html = self.download_url(url)
        #print(html)
        for url1 in self.get_linked_urls(url, html):
            if (url1 and self.is_same_domain(url, url1)):
                print("queuing up " + url1)
                self.add_url_to_visit(url1)

    def run(self):
        while self.urls_to_visit and self.count <= 10:
            self.count = self.count+1
            url = self.urls_to_visit.pop(0)
            logging.info(f'Crawling: {url}')
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f'Failed to crawl: {url}')
            finally:
                self.visited_urls.append(url)

#os.environ["openai_secret_key"] == st.secrets["openai_secret_key"]

from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

# Download OpenAI embeddings and map document chunks to embeddings in vector db
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)
print("Loaded db from file")
#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
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

def generate_response(query, db):
  docs = db.similarity_search(query, k=20)
  chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=QUESTION_PROMPT1, combine_prompt=COMBINE_PROMPT1)
  answer = chain.run(input_documents=docs, question=query)
  answer.replace ('. ', '.\n')
  return answer

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
  output = generate_response(user_input, db)
  st.session_state.past.append(user_input)
  st.session_state.generated.append(output)

if st.session_state['generated']:
  for i in range(len(st.session_state['generated']) - 1, -1, -1):
    message(st.session_state["generated"][i], key=str(i))
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
