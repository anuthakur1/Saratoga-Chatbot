# Saratoga Chatbot

<p align="center">
    <img src="https://github.com/anuthakur1/Saratoga-Chatbot/assets/85850320/fdece36d-6682-4b11-81ab-4d7fd3b18608"
        width = "650"
        height = "600">
</p>

The Saratoga Chatbot can be used to answer all the questions you have about the city of Saratoga, CA. When you run the program, you will be taken to a domain where you will see the chatbot and be able to ask your questions.

First, the project will run the file *create_index.py*, which crawls all of the imported websites and splits them based on the information each sentence gives. This ensures that the chatbot gives the most accurate and concise information to the user. *create_index.py* is run only once to acquire the necessary information quickly and to keep the chatbot from lagging. The fetched information is broken into chunks and stored in a vector database.

Next, the file *streamlit_app.py* will be run, creates the domain for the chatbot. This is the file that makes the chatbot appear for the user. The vector database created in *create_index.py* is loaded up, and queries are run against it. The fetched chunks are provided as context to the large language model (LLM) to summarize. At this point, the chatbot is completely fuctional.

In this implementation we use OpenAI for the large language model and HuggingFace embeddings and FAISS vector database to create the vector index.

The Saratoga Chatbot can be immensely useful to those considering a move to Saratoga. They can find out about what the city has to offer in terms of education, activites, and local businesses. The chatbot can be equally useful to current Saratoga residents, as they can find new restaurants or events to visit.
