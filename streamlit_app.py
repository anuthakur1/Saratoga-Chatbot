# A streamlit based frontend that reads a vector db and creates a simple web interface
# for a chatbot. The prompting is done via a simple langchain based chain.
# Note that we use st.cache_resource to make sure the vectordb and other large resources like embeddings
# and the language model are loaded only once.

# To run:
# streamlit run streamlit_app.py

from copy import deepcopy
import streamlit as st
from streamlit_chat import message
import langchain
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
import tiktoken
import unstructured
from langchain.vectorstores import FAISS
import pytesseract
import openai
import os

from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

@st.cache_resource
class CompatibleChatOpenAI(ChatOpenAI):
    def _combine_llm_outputs(self, llm_outputs):
        cleaned_outputs = []

        for output in llm_outputs:
            if output is None:
                cleaned_outputs.append(None)
                continue

            cleaned = dict(output)
            cleaned["token_usage"] = {
                k: v
                for k, v in output.get("token_usage", {}).items()
                if isinstance(v, (int, float))
            }

            cleaned_outputs.append(cleaned)

        return super()._combine_llm_outputs(cleaned_outputs)
    
def init_resources():
  # Download OpenAI embeddings and map document chunks to embeddings in vector db
  embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
  )
  db = FAISS.load_local("faiss_index", embeddings)
  print("Loaded db from file ")
  llm = CompatibleChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
  )
  
  question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
  Return any relevant text in English.
  {text}
  Question: {question}
  Relevant text, if any, in English:"""
  QUESTION_PROMPT = PromptTemplate(
  template=question_prompt_template, input_variables=["text", "question"]
  )
  
  combine_prompt_template = """Given the following extracted parts of a long document and a question, create around 1000 word essay with references ("SOURCES") for each sentence.
  If you don't know the answer, just say that you don't know. Don't try to make up an answer. Always end the 
  answer with a full sentence.
  ALWAYS return a "SOURCES" part in your answer.
  Respond in English.
  
  QUESTION: {question}
  =========
  {text}
  =========
  FINAL ANSWER IN English:"""
  COMBINE_PROMPT = PromptTemplate(
  template=combine_prompt_template, input_variables=["text", "question"]
  )

  return (db, llm, QUESTION_PROMPT, COMBINE_PROMPT)

# We will get the user's input by calling the get_text function
def get_text():
  input_text = st.text_input("You: ", " ", key="input")
  return input_text

def generate_response(query, db, llm, QUESTION_PROMPT, COMBINE_PROMPT):
  docs = deepcopy(db.similarity_search(query, k=20))

  for doc in docs:
      source = doc.metadata.get("source", "Unknown")
      doc.page_content = (
          f"Source: {source}\n\n{doc.page_content}"
      )

  chain = load_summarize_chain(
      llm,
      chain_type="map_reduce",
      map_prompt=QUESTION_PROMPT,
      combine_prompt=COMBINE_PROMPT
  )

  answer = chain.run(
      input_documents=docs,
      question=query
  )

(db, llm, QUESTION_PROMPT, COMBINE_PROMPT) = init_resources()

import streamlit as st
from streamlit_chat import message

st.title("Saratoga Chatbot")

if 'generated' not in st.session_state:
  st.session_state['generated'] = []

if 'past' not in st.session_state:
  st.session_state['past'] = []

user_input = get_text()


if user_input:
  #output = generate_response(user_input)
  # store the output
  output = generate_response(user_input, db, llm, QUESTION_PROMPT, COMBINE_PROMPT)
  st.session_state.past.append(user_input)
  st.session_state.generated.append(output)

if st.session_state['generated']:
  for i in range(len(st.session_state['generated']) - 1, -1, -1):
    message(st.session_state["generated"][i], key=str(i))
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
