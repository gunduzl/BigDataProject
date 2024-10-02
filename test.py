import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

file_name = "A2F23"

# Use an f-string to include the file name in the path
loader = PyPDFLoader(f'{file_name}.pdf')
 # Replace with your actual PDF file path
documents = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create a vector store from the documents
vectorstore = FAISS.from_documents(docs, embeddings)

# Initialize Chat LLM
chat_llm = ChatOpenAI()

# Initialize message history
message_history = StreamlitChatMessageHistory(key="chat_history")

# Initialize conversation memory
memory = ConversationBufferMemory(
    chat_memory=message_history,
    memory_key='chat_history',
    input_key='question',  # Set input_key to 'question'
    return_messages=True
)

# Create the Conversational Retrieval Chain
conversation = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
    # Removed input_key and output_key
)

st.title(f'{file_name} Dosya Asistanı')

# Retrieve existing messages
messages = message_history.messages if message_history.messages else []

for message in messages:
    if isinstance(message, HumanMessage):
        st.chat_message('user').markdown(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message('assistant').markdown(message.content)

prompt = st.chat_input('Enter your question here')

if prompt:
    # Display user message
    st.chat_message('user').markdown(prompt)
    message_history.add_user_message(prompt)

    # Generate AI response using the conversation chain
    response = conversation.invoke({'question': prompt})

    # Get the assistant's reply
    assistant_reply = response['answer']

    # Display AI response
    st.chat_message('assistant').markdown(assistant_reply)
    message_history.add_ai_message(assistant_reply)
