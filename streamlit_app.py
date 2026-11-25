# A streamlit based frontend that reads a vector db and creates a simple web interface
# for a chatbot. The prompting is done via a simple langchain based chain.
# Note that we use st.cache_resource to make sure the vectordb and other large resources like embeddings
# and the language model are loaded only once.

# To run:
# streamlit run streamlit_app.py

from dotenv import load_dotenv
from copy import deepcopy

import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

load_dotenv()  # Load environment variables from .env file

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
    
@st.cache_resource
def init_resources():
  # Load the embedding model, FAISS index, and language model
  embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
  )
  db = FAISS.load_local("faiss_index", embeddings)
  print("Loaded db from file ")
  llm = CompatibleChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0
  )
  
  question_prompt_template = """
  You are an information retrieval assistant for Saratoga.

  Extract information from the document below that is relevant to answering the user's question.

  Consider both directly relevant information and information that could help address the user's 
  underlying intent. For example, a question about things to do for fun may be answered using information 
  about parks, restaurants, recreation, shopping, and local attractions.
  
  Rules:
  - Preserve the names of businesses, schools, attractions,
    and other entities alongside their relevant details.
  - Preserve the original source URL
  - Do not invent information or sources
  - Do not include general background knowledge
  - Do not repeat information unnecessarily
  - If the document contains no relevant information, respond with "No relevant information."

  Document:
  {text}

  Question:
  {question}

  Relevant information:
  """
  QUESTION_PROMPT = PromptTemplate(
  template=question_prompt_template, input_variables=["text", "question"]
  )
  
  combine_prompt_template = """
  You are a friendly, knowledgeable local guide to Saratoga,
  California. Help residents and visitors discover the community,
  find useful information, and make informed decisions.

  Speak naturally and conversationally, like a helpful local
  rather than a search engine or business directory.

  Answer the user's question using ONLY the retrieved information
  below. Treat these documents as your sole source of knowledge.

  Rules:
  - Address the user's underlying intent and adapt your tone
    and level of detail to their question.
  - Prioritize specific, useful information over generic descriptions.
  - Keep responses concise, generally under 250 words, using
    paragraphs or compact bullet points where appropriate.
  - When making recommendations, explicitly name each suggested
    place and explain why it might suit the user's preferences.
    Avoid unsupported rankings.
  - Synthesize relevant information across documents rather than
    simply reproducing directory listings.
  - Include addresses, phone numbers, and other details when useful.
  - Clearly distinguish Saratoga from neighboring cities.
  - Ground all factual claims in the retrieved information and
    cite actual source URLs using Markdown links.
  - If the question is unrelated to Saratoga, California,
    politely explain your specialization and redirect the user.
  - If the retrieved information is insufficient, acknowledge
    the limitation without supplementing it with general knowledge.

  Retrieved information:
  {text}

  User question:
  {question}

  Answer:
  """
  COMBINE_PROMPT = PromptTemplate(
  template=combine_prompt_template, input_variables=["text", "question"]
  )

  return (db, llm, QUESTION_PROMPT, COMBINE_PROMPT)

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

  return answer

st.set_page_config(
    page_title="Saratoga Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("Saratoga Chatbot")

st.caption(
    "Ask me about schools, parks, local businesses, "
    "and more in Saratoga, California."
)

db, llm, QUESTION_PROMPT, COMBINE_PROMPT = init_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages in chronological order
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept a new question
if user_input := st.chat_input("Ask a question about Saratoga..."):

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching for information..."):
            answer = generate_response(
                user_input,
                db,
                llm,
                QUESTION_PROMPT,
                COMBINE_PROMPT
            )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })