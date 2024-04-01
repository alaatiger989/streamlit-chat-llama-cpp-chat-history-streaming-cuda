from deep_translator import GoogleTranslator
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import LlamaCpp

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory , ChatMessageHistory , ConversationTokenBufferMemory
from langchain_experimental.chat_models import Llama2Chat

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
import os
# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container,  initial_text="" , token_no = 0):
        self.container = container
        self.text = initial_text
        self.token_no = token_no
       

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_no = self.token_no+1
        # Add the response to the chat window
        with self.container.empty():
            self.container.header("Chat Session")
            with self.container.chat_message("ai"): 
                #print(str(self.token_no))
                if self.token_no != 0:
                    self.text += token
                    print(token , end = '',flush = True)
                    #print(self.text)
                    st.markdown(self.text)
                else:
                    self.text = ""
        


@st.cache_resource
def create_chain(system_prompt):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = LlamaCpp(
        #model_path=os.getcwd() + "\llama-2-7b-chat.Q4_K_M.gguf",
        model_path = "C:/Users/Alaa AI/Python Projects/Ai Models/alaa_ai_model_llama2_K_M_v1.4.gguf",
        #streaming=False, n_gpu_layers=30, n_ctx=3584, n_batch=521, verbose=True
         n_gpu_layers = 15000 , n_ctx = 5000 , streaming=True ,max_tokens = 5000 ,n_batch=512,verbose = False, use_mlock = True , use_mmap = True ,temperature=0
    )
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    stream_handler = StreamHandler(st.sidebar.empty())
    #stream_handler = StreamHandler(st.chat_message("ai").empty())
    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear that the LLM is typing the
    # responses in real-time.
    callback_manager = CallbackManager([stream_handler])

##    (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
##                                  "mistral-7b-instruct-v0.1.Q4_0.gguf")

##    model_path = hf_hub_download(repo_id=repo_id,
##                                 filename=model_file_name,
##                                 repo_type="model")

    # initialize LlamaCpp LLM model
    # n_gpu_layers, n_batch, and n_ctx are for GPU support.
    # When not set, CPU will be used.
    # set 1 for Mac m2, and higher numbers based on your GPU support
##    llm = LlamaCpp(
##            model_path="C:/Users/Alaa AI/Python Projects/Ai Models/llama-2-7b-chat.q4_K_M.gguf",            temperature=0,
##            max_tokens=512,
##            top_p=1,
##            # callback_manager=callback_manager,
##            # n_gpu_layers=1,
##            # n_batch=512,
##            # n_ctx=4096,
##            stop=["[INST]"],
##            verbose=False,
##            streaming=True,
##            )
    demo_ephemeral_chat_history = ChatMessageHistory()
    template_messages = [
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    
    
    prompt_template = ChatPromptTemplate.from_messages(template_messages)
    model = Llama2Chat(llm=llm , callback_manager = callback_manager)
    # Template you will use to structure your user input before converting
    # into a prompt. Here, my template first injects the personality I wish to
    # give to the LLM before in the form of system_prompt pushing the actual
    # prompt from the user. Note that this chatbot doesn't have any memory of
    # the conversation. So we will inject the system prompt for each message.
##    template = """
##    <s>[INST]{}[/INST]</s>
##
##    [INST]{}[/INST]
##    """.format(system_prompt, "{question}")
##
##    # We create a prompt from the template so we can use it with Langchain
##    prompt = PromptTemplate(template=template, input_variables=["question"])

    # We create an llm chain with our LLM and prompt
    # llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy
##    llm_chain = prompt | llm  # LCEL
    
    
    
    llm_chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)
    chain_with_message_message_history = RunnableWithMessageHistory(
        llm_chain,
        lambda session_id: demo_ephemeral_chat_history,
        input_messages_key = "text",
        history_messages_key = "chat_history"
        )
    return chain_with_message_message_history , memory , llm


# Set the webpage title
st.set_page_config(
    page_title="Alaa's Chat Robot!"
)

# Create a header element
st.header("Alaa's Chat Robot!")



# Create Select Box
lang_opts = ["ar","en" , "fr"]
lang_selected = st.selectbox("Select Target Language " , options = lang_opts)
# This sets the LLM's personality for each prompt.
# The initial personality provided is basic.
# Try something interesting and notice how the LLM responses are affected.
system_prompt = st.text_area(
    label="System Prompt",
    value="You are a helpful AI assistant who answers questions in short sentences.",
    key="system_prompt")


# Create LLM chain to use for our chatbot.
llm_chain , memory , llm= create_chain(system_prompt)

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Pass our input to the LLM chain and capture the final responses.
    # It is worth noting that the Stream Handler is already receiving the
    # streaming response as the llm is generating. We get our response
    # here once the LLM has finished generating the complete response.
    user_prompt = GoogleTranslator(source='auto', target='en').translate(user_prompt)

    #print(memory.chat_memory.messages)
    #print("\n\nCurrent buffer length : " + str(llm.get_num_tokens_from_messages(memory.chat_memory.messages)))
    #response = llm_chain.invoke(user_prompt)
    response = llm_chain.invoke(
        {"text": user_prompt},
        {"configurable": {"session_id": "unused"}},
    )
   
    buffer = memory.chat_memory.messages
    curr_buffer_length = llm.get_num_tokens_from_messages(buffer)
    print("Current buffer length " + str(curr_buffer_length))
    #print("Before pop : " + str(buffer))
    if curr_buffer_length > llm.n_ctx - 4500:
        del buffer[:2]
        #buffer.pop(0)

    #print("After pop : " + str(buffer))
    print("Current buffer length " + str(llm.get_num_tokens_from_messages(buffer)))
    response = GoogleTranslator(source='auto', target=lang_selected).translate(response["text"])
    
    # Add the response to the session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)

