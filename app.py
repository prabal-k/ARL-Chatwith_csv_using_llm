import streamlit as st
from streamlit_chat import message  # to create conversational interface
from langchain.document_loaders.csv_loader import CSVLoader  # to load csv file
from langchain.embeddings import HuggingFaceEmbeddings  # for embeddings
from langchain.vectorstores import FAISS  # to store the vectors
from langchain.llms import CTransformers  # to load the llama2 model
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss  # Import faiss library
import numpy as np

DB_FAISS_DIR = "vectorstore/db_faiss"  # To store the embedding locally
DB_FAISS_PATH = os.path.join(DB_FAISS_DIR, "index.faiss")  # Full path to the FAISS index file
CSV_FILE_PATH = "preprocessed-hr-data.csv"  # Predefined CSV file location

# Function for Loading the model from the local storage
def load_llm():
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama", config={'temperature': 0.7, 'max_new_tokens': 500})
    return llm

st.title("Welcome to chat-chat-chat-gossip")

# Mark down
st.markdown("<h3 style='text-align: center; color: green;'>Guff garam ekxin aau</h3>",
            unsafe_allow_html=True)

# Ensure the vectorstore directory exists
os.makedirs(DB_FAISS_DIR, exist_ok=True)

# Check if the embeddings already exist
if not os.path.exists(DB_FAISS_PATH):
    try:
        st.title("Creating the embeddings on the fly")
        loader = CSVLoader(file_path=CSV_FILE_PATH, encoding='utf-8')

        data = loader.load()  # This loads the CSV file data into the data variable
        st.write(data[0])

        # Splitting text into chunks with overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n", ","])
        split_docs = text_splitter.split_documents(data)

        # CREATING THE EMBEDDINGS
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                           model_kwargs={'device': 'cpu'})
        
        # Generate embeddings for documents
        doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in split_docs]
        
        # Initialize FAISS index with IndexFlatL2
        index = faiss.IndexFlatL2(len(doc_embeddings[0]))
        index.add(np.array(doc_embeddings).astype(np.float32))  # Convert embeddings to numpy array of type float32
        
        # Save FAISS index
        faiss.write_index(index, DB_FAISS_PATH)
        
        # Create LangChain FAISS vector store
        db = FAISS.from_documents(split_docs, embeddings)
        db.index = index  # Replace the index with our custom FAISS index
        db.save_local(DB_FAISS_DIR)
        
    except Exception as e:
        st.error(f"An error occurred during embedding creation: {e}")

# Load the embeddings
try:
    st.title("Embeddings are being loaded")
    
    # Load FAISS index
    index = faiss.read_index(DB_FAISS_PATH)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    
    # Load documents and create LangChain FAISS vector store
    loader = CSVLoader(file_path=CSV_FILE_PATH, encoding='utf-8')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n", ","])
    split_docs = text_splitter.split_documents(data)
    
    db = FAISS.from_documents(split_docs, embeddings)
    db.index = index  # Replace the index with our custom FAISS index
    
except Exception as e:
    st.error(f"An error occurred during loading the embeddings: {e}")

if 'db' in locals():
    # Calling the llm
    llm = load_llm()

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "You are a knowledgeable assistant with access to a CSV dataset. "
            "Based on the given context, answer the question accurately. "
            "Question: {question}\n\n"
            "Context:\n{context}\n"
        )
    )

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    # Function for chat
    def conversational_chat(query):
        user_embedding = embeddings.embed_query(query)
        _, indices = index.search(np.array([user_embedding]).astype(np.float32), 10)
        context_docs = [split_docs[i] for i in indices[0]]

        # Print context documents for debugging
        st.write("Context Documents:")
        for i, doc in enumerate(context_docs):
            st.write(f"Document {i + 1}:\n{doc.page_content}\n")

        context = "\n".join([doc.page_content for doc in context_docs])
        # Check the token length and truncate context if necessary
        max_context_length = 512 - len(query.split()) - 50  # Reserve 50 tokens for other parts of the prompt
        context_tokens = context.split()
        if len(context_tokens) > max_context_length:
            context = " ".join(context_tokens[:max_context_length])

        # Format the prompt with the query
        inputs = {
            "question": query,
            "context": context,
            "chat_history": st.session_state.get('history', [])
        }

        # Print the inputs for debugging
        st.write("Formatted Prompt Inputs:", inputs)

        # Run the chain with formatted inputs
        result = chain.run(**inputs)
        
        if isinstance(result, str):
            answer = result
        else:
            answer = result.get('answer', 'Sorry, I could not find an answer.')
        
        st.session_state['history'].append((query, answer))
        return answer

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about the CSV data 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
else:
    st.error("Failed to load the FAISS vector store. Embeddings not available.")
