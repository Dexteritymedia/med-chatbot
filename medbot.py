import os

from langchain.agents import initialize_agent, AgentType, Tool, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.tools import AIPluginTool

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
    with st.spinner("typing..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    try:
        with st.spinner("Please wait..."):
            llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key, streaming=True)
            
            #tool = AIPluginTool.from_plugin_url("https://scholar.mixerbox.com/.well-known/ai-plugin.json") #For Free and reliable academic search engine! Find research papers and get answers in an instant!
            #tool = AIPluginTool.from_plugin_url("https://scholar-ai.net/.well-known/ai-plugin.json")
            
            tool = AIPluginTool.from_plugin_url("https://nextpaperplugin--mengzhao1.repl.co/.well-known/ai-plugin.json")
            #tool = AIPluginTool.from_plugin_url("https://txyz.ai/.well-known/ai-plugin.json") #To Effortlessly decipher, compare, and answer questions about research papers using a simple Arxiv ID.
            #tool = AIPluginTool.from_plugin_url("https://scholarly.maila.ai/.well-known/ai-plugin.json") #Scholarly is an AI-powered search engine for exploring scientific literature.
            
            #tool = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
            #tool = AIPluginTool.from_plugin_url("https://api.storybird.ai/.well-known/ai-plugin.json") #Create beautiful, illustrated stories easily.
            #https://chatgpt.boolio.co.kr/.well-known/ai-plugin.json #The easiest way to analyze global stock values with the power of quantitative factor methodologies.
            #tool = AIPluginTool.from_plugin_url("https://oa.mg/.well-known/ai-plugin.json")#For Searching over 250M scientific papers and research articles. Perfect for researchers or students.
            
            tools = load_tools(["requests_all"])
            tools += [tool]
            search_agent = initialize_agent(
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
            #st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            #response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            response = search_agent.run(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    except Exception as e:
            st.error(f"Sorry, there seems to be an issue. Please try again \n\n{e}", icon="⚠️")
