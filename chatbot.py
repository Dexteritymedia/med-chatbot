import os

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Medical Chatbot", page_icon="")
st.title("🩺 Medical Chatbot")

serper_api_key = os.getenv('serper_api_key')

st.markdown("""
<style>
#MainMenu {
    visibility: hidden;
}
.css-h5rgaw {
    visibility: hidden;
}

</style>


""", unsafe_allow_html=True)

PREFIX = """
    You are an AI medical assistant having a conversation with a human.
    The human want help on how to improve, learn about medical terms and health
    You should be has detailed and interesting as best as you can,
    a little talkative is fine and explain in length and provide information that would be useful
    to them, if possible include an example so the user can understand.
    Also, when requested modify the answers specifically to help them develop a good health habit.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""


prefix = """
    You are an AI medical assistant having a conversation with a human.
    The human want help on how to improve, learn about medical terms and health
    You should be has detailed and interesting as best as you can,
    a little talkative is fine and explain in length and provide information that would be useful
    to them, if possible include an example so the user can understand.
    Also, when requested modify the answers specifically to help them develop a good health habit.
    You have access to the following tools:"""
suffix = """Begin!
        
        Question: {input}
        {agent_scratchpad}"""


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True)
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to answer questions about a person, medical health",
        )
    ]
    search_agent = initialize_agent(
        #tools=[DuckDuckGoSearchRun(name="Search")],
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
        early_stopping_method="generate",
        agent_kwargs={
        	'prefix':PREFIX,
                'format_instructions':FORMAT_INSTRUCTIONS,
        	'suffix':SUFFIX,
                'input_variables': ['input','agent_scratchpad'],
    }
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        #response = search_agent.run(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
