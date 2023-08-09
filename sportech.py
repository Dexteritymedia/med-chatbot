import os

from langchain.agents import initialize_agent, AgentType, Tool, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Iberian Sportech Chatbot", page_icon="")
st.title("Iberian Sportech Chatbot")

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

loader = PyPDFLoader('./FAQ-CLIENTE.pdf')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
vector_store.save_local("./data_store")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

if prompt := st.chat_input(placeholder="Ask your question?"):
    with st.spinner("typing..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    try:
        with st.spinner("Please wait..."):
            prompt_template = """Use the following pieces of context to answer the question at the end.
                                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                                    {context}
                                    Question: {question}
                                    Answer:"""
            qa_prompt = PromptTemplate(

		template=prompt_template,
		input_variables=["context", "question"]
            )
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
            
            docsearch = FAISS.load_local(
                "./data_store",
                OpenAIEmbeddings()
            )
            
            conversation = ConversationChain(retriever=docsearch.as_retriever(), memory=st.session_state.buffer_memory, prompt=qa_prompt, llm=llm, verbose=True)
            
            
        with st.chat_message("assistant"):
            #st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            #response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            #response = search_agent.run(st.session_state.messages)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{prompt}")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    except Exception as e:
            st.error(f"Sorry, there seems to be an issue. Please try again!", icon="⚠️")
