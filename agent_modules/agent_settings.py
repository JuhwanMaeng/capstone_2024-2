from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os, faiss
from datetime import datetime
from agent_modules.agent_core import Agent
from agent_modules.agent_retriever import AgentRetriever
from agent_modules.agent_memory import AgentMemory
from langchain_community.docstore.in_memory import InMemoryDocstore

# Chat_Model
chat_gpt_4o = ChatOpenAI(model="gpt-4o", max_tokens= 1024, api_key="YOUR_API_KEY")

# API keys & Settings
os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY"
os.environ["PINECONE_INDEX_NAME"] = "YOUR_INDEX_NAME"
# Langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "YOUR_PROJECT_NAME"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"

# OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# FAISS
index = faiss.IndexFlatL2(len(embeddings.embed_query("size")))
faiss_db= FAISS(embedding_function=embeddings,index=index,docstore=InMemoryDocstore(),index_to_docstore_id={},)

# Agent's detail
default_status = {"Energy" : 10, "Health" : 10, "Satisfaction" : 10}
Ethan_personality = "Ethan has high conscientiousness, is highly systematic and organized in his work, pays attention to details, is responsible, and consistently delivers results. He works as a data analyst in an office."
Jack_personality = "Jack has high extraversion, is energetic, sociable, gains energy from interacting with others, has excellent leadership skills, and is popular among colleagues and students. He works as a physical education teacher at a school."
Oliver_personality = "Oliver has high openness, is open to new ideas and creative approaches, seeks change and diversity, and excels in presenting creative solutions to problems. He works as a project manager in an office."
Lily_personality = "Lily has high agreeableness, values cooperation with others, is always kind and friendly, leaves a positive impression in conversations with customers, and acts as an excellent mediator when problems arise. She works as a manager at a mart."
Emma_personality = "Emma has high neuroticism, worries a lot about situations, and always prepares for the worst-case scenario. This trait makes her very cautious and sensitive to her environment, focusing on keeping her surroundings clean and safe. She works as an environmental manager at a park."

friendship_dict = {
    "Ethan": {"Jack": 3.2, "Oliver": 3.6, "Lily": 3.2, "Emma": 3.9},
    "Jack": {"Ethan": 3.2, "Oliver": 3.8, "Lily": 3.0, "Emma": 3.1},
    "Oliver":{"Ethan": 3.6, "Jack": 3.8, "Lily": 4.0, "Emma": 3.2},
    "Lily": {"Ethan": 3.2, "Jack": 3.0, "Oliver": 4.0, "Emma": 3.3},
    "Emma": {"Ethan": 3.9, "Jack": 3.1, "Oliver": 3.2, "Lily": 3.3}}

Ethan_retriever = AgentRetriever(name = "Ethan", vectorstore = faiss_db)
Ethan_memory = AgentMemory(name = "Ethan", chat_model=chat_gpt_4o, memory_retriever=Ethan_retriever)
Ethan = Agent(name = "Ethan", age = 25, personality=Ethan_personality, status=default_status, memory=Ethan_memory, chat_model=chat_gpt_4o, friendship=friendship_dict["Ethan"])

Jack_retriever = AgentRetriever(name = "Jack", vectorstore = faiss_db)
Jack_memory = AgentMemory(name = "Jack", chat_model=chat_gpt_4o, memory_retriever= Jack_retriever)
Jack = Agent(name = "Jack", age = 32, personality=Jack_personality, status= default_status, memory=Jack_memory, chat_model=chat_gpt_4o, friendship=friendship_dict["Jack"])

Oliver_retriever = AgentRetriever(name = "Oliver", vectorstore = faiss_db)
Oliver_memory = AgentMemory(name = "Oliver",chat_model=chat_gpt_4o, memory_retriever= Oliver_retriever)
Oliver = Agent(name = "Oliver", age = 30, personality=Oliver_personality, status=default_status, memory=Oliver_memory, chat_model=chat_gpt_4o, friendship=friendship_dict["Oliver"])

Lily_retriever = AgentRetriever(name = "Lily", vectorstore = faiss_db)
Lily_memory = AgentMemory(name = "Lily",chat_model=chat_gpt_4o, memory_retriever= Lily_retriever)
Lily = Agent(name = "Lily", age = 26, personality=Lily_personality, status= default_status, memory=Lily_memory, chat_model=chat_gpt_4o, friendship=friendship_dict["Lily"])

Emma_retriever = AgentRetriever(name = "Emma", vectorstore = pinecone_db)
Emma_memory = AgentMemory(name = "Emma", chat_model=chat_gpt_4o, memory_retriever = Emma_retriever) 
Emma = Agent(name = "Emma", age = 28, personality=Emma_personality, status= default_status,memory=Emma_memory, chat_model=chat_gpt_4o, friendship=friendship_dict["Emma"])

Agents = {}
agent_names = ["Ethan", "Jack", "Lily", "Oliver", "Emma"]

for name in agent_names:
    retriever = globals()[f"{name}_retriever"]
    memory = globals()[f"{name}_memory"]
    agent = globals()[name]
    Agents[name] = agent


#parsing function
MONTHS = {
    'Jan' : "01",
    'Feb' : "02",
    'Mar' : "03",
    'Apr' : "04",
    'May' : "05",
    'Jun' : "06",
    'Jul' : "07",
    'Aug' : "08",
    'Sep' : "09",
    'Oct' : "10",
    'Nov' : "11",
    'Dec' : "12"}

def convert_time(time : str) :
    time = time.split()
    month = MONTHS[time[0]]
    date = time[1]
    time = time[2]
    return datetime.strptime(f"2024/{month}/{date} {time}", "%Y/%m/%d %H:%M")