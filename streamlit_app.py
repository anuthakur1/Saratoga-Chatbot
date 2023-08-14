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

#os.environ["openai_secret_key"] == st.secrets["openai_secret_key"]

from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class AppHelper(object):
    initialized = False
    def __new__(cls):
        """ creates a singleton object, if it is not created,
        or else returns the previous singleton object"""
        if not hasattr(cls, 'instance'):
            cls.instance = super(AppHelper, cls).__new__(cls)

        return cls.instance
    
    def __init__(self):
        if (not self.initialized):
          # Download OpenAI embeddings and map document chunks to embeddings in vector db
          embeddings = OpenAIEmbeddings()
          self.db = FAISS.load_local("faiss_index", embeddings)
          print("Loaded db from file ")
          print(self.initialized)
          #os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
          self.llm = OpenAI(temperature=0)
          
          question_prompt_template1 = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
          Return any relevant text in English.
          {text}
          Question: {question}
          Relevant text, if any, in English:"""
          self.QUESTION_PROMPT1 = PromptTemplate(
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
          self.COMBINE_PROMPT1 = PromptTemplate(
          template=combine_prompt_template1, input_variables=["text", "question"]
          )
          self.initialized = True
          print(self.initialized)

helper = AppHelper()

# We will get the user's input by calling the get_text function
def get_text():
  input_text = st.text_input("You: ", "Hello, how are you?", key="input")
  return input_text

def generate_response(query, db):
  docs = db.similarity_search(query, k=20)
  chain = load_summarize_chain(helper.llm, chain_type="map_reduce", map_prompt=helper.QUESTION_PROMPT1, combine_prompt=helper.COMBINE_PROMPT1)
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
  output = generate_response(user_input, helper.db)
  st.session_state.past.append(user_input)
  st.session_state.generated.append(output)

if st.session_state['generated']:
  for i in range(len(st.session_state['generated']) - 1, -1, -1):
    message(st.session_state["generated"][i], key=str(i))
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
