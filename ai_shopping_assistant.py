from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.llms.openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv
import os #provides ways to access the Operating System and allows us to read the environment variables

load_dotenv()

st.set_page_config(page_title="AI shopping assistant Chatbot", page_icon="")
st.title("AI shopping assistant Chatbot")

st.markdown("""
<style>
#MainMenu {
    visibility: hidden;
}
.css-h5rgaw {
    visibility: hidden;
}
.css-14xtw13 {
  display: none;
}
.css-1wbqy5l {
    display: none,
}
..css-1dp5vir {
    display: none,
}
</style>


""", unsafe_allow_html=True)

openai_api_key = os.getenv('openai_api_key')

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hola, ¿cómo puedo ayudarte?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template=
                                                                """"""
                                                                )

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

DATA_STORE_DIR = "data_store"

if os.path.exists(DATA_STORE_DIR):
  #st.write("Loading database")
  docsearch  = FAISS.load_local(
      DATA_STORE_DIR,
      OpenAIEmbeddings()
  )
else:
  st.write(f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first")

docsearch = "Abcdhjfbkn"
doc_chain = load_qa_chain(llm, chain_type="stuff")
chain_type_kwargs = {"prompt": prompt_template}  
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm)
#conversation = ConversationalRetrievalChain(retriever=docsearch.as_retriever(), memory=st.session_state.buffer_memory, chain_type_kwargs=chain_type_kwargs, combine_docs_chain=doc_chain, llm=llm)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Hello, how can I help you? ", key="input")
    if query:
        try:
            with st.spinner("typing..."):  
                #response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                response = conversation(query)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
        except Exception as e:
            st.error(f"Lo siento, parece que hay un problema. ¡Inténtalo de nuevo!\n\n ", icon="⚠️")
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          
