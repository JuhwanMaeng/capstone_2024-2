from pymongo import MongoClient, errors
import os, faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4

# 디렉터리와 초기화 변수 설정
directory = 'processed_scripts/'
SPLIT_SCRIPT = []
i = 1

# 스크립트 파일 읽기
while True:
    filename = f"script_scene_{i}.txt"
    file_path = os.path.join(directory, filename)
    
    # 파일이 존재하면 읽어서 처리
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            SPLIT_SCRIPT.append(text)
        i += 1  # 다음 숫자로 넘어감
    else:
        break

# MongoDB 연결 및 예외 처리
try:
    client = MongoClient("mongodb://root:root@localhost:27017/", serverSelectionTimeoutMS=5000)
    db = client['capstone']
    character_collection = db['character']
    plot_collection = db['plot']

    # MongoDB 컬렉션 데이터 가져오기
    try:
        CHARACTER_LIST = [doc["_id"] for doc in character_collection.find({}, {"_id": 1})]
        PLOT_LIST = [doc["_id"] for doc in plot_collection.find({}, {"_id": 1})]
        if not CHARACTER_LIST:
            print("No characters found in MongoDB.")
    except errors.PyMongoError as e:
        print(f"Error accessing MongoDB collections: {e}")
        CHARACTER_LIST = []
        PLOT_LIST = []

except errors.ServerSelectionTimeoutError as e:
    print(f"MongoDB connection failed: {e}")
    CHARACTER_LIST = []
    PLOT_LIST = []

# FAISS 설정
FAISS_PATH = "db/"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_stores = {}

# 캐릭터 별로 벡터 스토어 로드 또는 초기화
for character in CHARACTER_LIST:
    character_path = os.path.join(FAISS_PATH, f"{character}_faiss")
    try:
        if os.path.exists(character_path):
            vector_store = FAISS.load_local(character_path, embeddings, allow_dangerous_deserialization=True)
        else:
            embedding_dim = len(embeddings.embed_query("test"))
            index = faiss.IndexFlatL2(embedding_dim)
            vector_store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
            os.makedirs(FAISS_PATH, exist_ok=True)
            vector_store.save_local(character_path)
        vector_stores[character] = vector_store
    except Exception as e:
        print(f"Error processing character {character}: {e}")

if not CHARACTER_LIST:
    print("No valid data to initialize FAISS vector stores.")
