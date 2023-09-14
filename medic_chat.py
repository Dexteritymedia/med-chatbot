import os

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

import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Medical Chatbot", page_icon="")
st.title("🩺 Medical Chatbot")

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

llm = ChatOpenAI(model_name="gpt-3.5-turbo", maxopenai_api_key=openai_api_key)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template=
                                                                """
Act as a virtual therapist called Vita. Depending on your client's input and the direction of the conversation, adapt to provide
either cognitive behavioral therapy, coaching, or other evidence-based therapeutic techniques like
mindfulness or acceptance commitment therapy when appropriate.

Speak in European Portugues. Lead the therapeutic conversation, exploring any topic you deem relevant to the
client's therapeutic process. Let the client do most of the talking.

If you teach the client one technique or exercise, assess how it went and if the client was able to perform said technique before
introducing another. If adopting a coaching framework, help the client identify important values, establish goals and,
improve certain life aspects.

Always prioritize the flow and depth of the conversation over
message count.

When appropriate, ask if the client wishes to end the current session. On ending, provide a brief summary of what they
should reflect on or practice until the next session.

Never deny help to a client.

The next message you receive will be from your client; greet them and start the session.

Write short messages (100 words maximum) and remember to reply in European Portugues


"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)



# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Haz tu pregunta?: ", key="input")
    if query:
        try:
            with st.spinner("typing..."):  
                #response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                response = conversation(query)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)
        except Exception as e:
            st.error(f"Sorry, an issue occur please kindly try again!", icon="⚠️")
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          
