import json, os, re
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import File, UploadFile, Query
from tqdm.asyncio import tqdm
from modules.process import Split_Script, Merge_Script, Extract_Characterlist, Extract_Event, Summarize_Plot, Extract_Trait
from modules.agent.agent_settings import *
from pydantic import BaseModel
import logging
import sys
from modules.database import PLOT_LIST, CHARACTER_LIST, SPLIT_SCRIPT, character_collection, plot_collection

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("server.log"), logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)

class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message != '\n': 
            self.buffer += message
        if self.buffer.endswith('\n'):
            self.level(self.buffer.strip())
            self.buffer = ''

    def flush(self):
        if self.buffer:
            self.level(self.buffer.strip())
            self.buffer = ''

    def isatty(self):
        return False

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

# 데이터 파일 경로
SCATTER_DATA_FILE = "web/data/scatter_data.json"
TREE_DATA_FILE = "web/data/tree_data.json"
GRAPH_DATA_FILE = "web/data/graph_data.json"
RADAR_DATA_FILE = "web/data/radar_data.json"
UPLOADS_DIR = "web/uploads"

# 파일에서 데이터 로드
def load_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Error reading {file_path}")

# 파일에 데이터 저장
def save_data(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files 및 Templates 설정
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# 메인 페이지 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 트리 차트 페이지
@app.get("/tree", response_class=HTMLResponse)
async def get_tree_chart(request: Request):
    return templates.TemplateResponse("tree.html", {"request": request})

# Scatter Plot 페이지
@app.get("/scatter", response_class=HTMLResponse)
async def get_scatter_plot(request: Request):
    return templates.TemplateResponse("scatter.html", {"request": request})

# Radar Chart 페이지
@app.get("/radar", response_class=HTMLResponse)
async def get_radar_chart(request: Request):
    return templates.TemplateResponse("radar.html", {"request": request})

# Vis.js 네트워크 그래프 페이지
@app.get("/graph", response_class=HTMLResponse)
async def get_graph_page(request: Request):
    return templates.TemplateResponse("graph.html", {"request": request})

# Scatter 데이터 API
@app.get("/scatter_data")
async def get_scatter_data():
    data = load_data(SCATTER_DATA_FILE)
    if not data:
        return {"message": "No scatter data found"}
    return data

@app.post("/scatter_data")
async def add_scatter_data(x: str, y: float):
    data = load_data(SCATTER_DATA_FILE) or {"data": []}
    data["data"].append({"x": x, "y": y})
    save_data(SCATTER_DATA_FILE, data)
    return {"message": "Scatter data added", "data": {"x": x, "y": y}}

# Tree 데이터 API
@app.get("/tree_data")
async def get_tree_data():
    documents = list(plot_collection.find({}, {"_id": 0}))
    if not documents:
        return {"message": "No tree data found"}
    documents.sort(key=lambda doc: int(doc["scene_number"].strip("#")))
    tree_data = {
        "name": "Story Root",
        "children": [
            {
                "name": doc["scene_number"],
                "children": [
                    {"name": "Summary", "value": doc["summary"]},
                    {"name": "Setting", "value": doc["setting"]},
                    *[
                        {"name": "Participant", "value": participant}
                        for participant in doc["participants"]
                    ]
                ]
            }
            for doc in documents
        ]
    }
    return tree_data

@app.get("/script_data")
async def get_script_data(node: str = Query(...)):
    directory = "processed_scripts"
    file_name = f"script_scene_{node}.txt"
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            # 파일 내용을 읽고 \n을 <br>로 변환
            content = file.read().replace("\n", "<br>")
            return content
    else:
        return {"error": f"No script found for node {node}"}, 404



@app.post("/tree_data")
async def add_tree_data(data: dict):
    save_data(TREE_DATA_FILE, data)
    return {"message": "Tree data saved successfully"}

# Graph 데이터 API
@app.get("/graph_data")
async def get_graph_data():
#로그 추출 후, 시뮬레이션 테스트용
    nodes = [
        {"id": 1, "label": "BENOIT"},
        {"id": 2, "label": "MARTA"},
        {"id": 3, "label": "RANSOM"},
        {"id": 4, "label": "LINDA"},
        {"id": 5, "label": "FRAN"},
    ]
    edges = [
        {"id": "1-2", "from": 1, "to": 2, "weight": -3, "label": "-3", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "1-3", "from": 1, "to": 3, "weight": -4, "label": "-4", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "1-4", "from": 1, "to": 4, "weight": 0, "label": "0", "color": {"color": "#cccccc"}, "arrows": "to"},
        {"id": "1-5", "from": 1, "to": 5, "weight": -2, "label": "-2", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "2-1", "from": 2, "to": 1, "weight": -2, "label": "-2", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "2-3", "from": 2, "to": 3, "weight": -5, "label": "-5", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "2-4", "from": 2, "to": 4, "weight": 0, "label": "0", "color": {"color": "#cccccc"}, "arrows": "to"},
        {"id": "2-5", "from": 2, "to": 5, "weight": +3, "label": "+3", "color": {"color": "#2ecc71"}, "arrows": "to"},
        {"id": "3-1", "from": 3, "to": 1, "weight": -3, "label": "-3", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "3-2", "from": 3, "to": 2, "weight": -4, "label": "-4", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "3-4", "from": 3, "to": 4, "weight": 0, "label": "0", "color": {"color": "#cccccc"}, "arrows": "to"},
        {"id": "3-5", "from": 3, "to": 5, "weight": -3, "label": "-3", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "4-1", "from": 4, "to": 1, "weight": -2, "label": "-2", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "4-2", "from": 4, "to": 2, "weight": +2, "label": "+2", "color": {"color": "#2ecc71"}, "arrows": "to"},
        {"id": "4-3", "from": 4, "to": 3, "weight": -3, "label": "-3", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "4-5", "from": 4, "to": 5, "weight": 0, "label": "0", "color": {"color": "#cccccc"}, "arrows": "to"},
        {"id": "5-1", "from": 5, "to": 1, "weight": -1, "label": "-1", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "5-2", "from": 5, "to": 2, "weight": +2, "label": "+2", "color": {"color": "#2ecc71"}, "arrows": "to"},
        {"id": "5-3", "from": 5, "to": 3, "weight": -5, "label": "-5", "color": {"color": "#e74c3c"}, "arrows": "to"},
        {"id": "5-4", "from": 5, "to": 4, "weight": 0, "label": "0", "color": {"color": "#cccccc"}, "arrows": "to"},
    ]
    
    return {"nodes": nodes, "edges": edges}

# Radar 데이터 API
@app.get("/radar_data")
async def get_radar_data():
    radar_data = []
    CHARACTER_LIST = [doc["_id"] for doc in character_collection.find({}, {"_id": 1})]
    
    for character_id in CHARACTER_LIST:
        character_doc = character_collection.find_one({"_id": character_id})
        if not character_doc or "traits" not in character_doc:
            continue

        name = character_doc.get("name", f"Character_{character_id}")
        traits = character_doc["traits"]
        r = [
            traits.get("Extraversion", 3.0),
            traits.get("Agreeableness", 3.0),
            traits.get("Conscientiousness", 3.0),
            traits.get("Neuroticism", 3.0),
            traits.get("Openness", 3.0)
        ]
        theta = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
        
        radar_data.append({
            "name": name,
            "r": r,
            "theta": theta
        })
    
    if not radar_data:
        return {"message": "No radar data found"}
    
    return radar_data

@app.get("/get_plot_radar")
async def get_plot_radar():
    """
    플롯 정보 조회 엔드포인트.
    각 씬(scene_number)과 해당 등장인물(participants)을 반환하며, scene_number 기준으로 정렬합니다.
    """
    plots = []
    
    for plot_doc in plot_collection.find({}, {"_id": 1, "scene_number": 1, "participants": 1}):
        plots.append({
            "scene_number": plot_doc.get("scene_number", "Unknown"),
            "participants": plot_doc.get("participants", [])
        })

    if not plots:
        return {"message": "No plot data found"}

    # scene_number를 숫자로 정렬
    def extract_scene_number(scene):
        try:
            # scene_number에서 숫자만 추출
            return int(scene["scene_number"].lstrip("#"))
        except ValueError:
            return float('inf')  # 숫자가 없는 경우 맨 뒤로 정렬
    
    sorted_plots = sorted(plots, key=extract_scene_number)

    return sorted_plots




@app.post("/upload_script")
async def upload_script(file: UploadFile = File(...)):
    file_content = (await file.read()).decode("utf-8")
    async def stream_and_save():
        split_texts = await Split_Script(file_content, file_content[:1500])
        directory = 'processed_scripts/'
        merged_texts = Merge_Script(SPLIT_SCRIPT, 2000)
        character_list = await Extract_Characterlist(merged_texts)
        #로그 추출 후, 시뮬레이션 테스트용
        character_list = ['MARTA', 'BLANC', 'MEG', 'ELLIOTT', 'WAGNER', 'WALT', 'HARLAN', 'LINDA', 'RICHARD', 'JONI', 'FRAN', 'RANSOM', 'JACOB']
        await Extract_Event(merged_texts, character_list)
        await Summarize_Plot(split_texts, character_list)
        await Extract_Trait(character_list)
        yield "finished"
    return StreamingResponse(stream_and_save(), media_type="text/plain")

# Pydantic Type Class
class Chat(BaseModel):
    plot_name : str

class Time(BaseModel):
    time : str

class Plot(BaseModel):
    plot_name : str

@app.get("/get_plot")
async def get_plot():
    if len(PLOT_LIST) != 0 :
         return {"result" : str(PLOT_LIST[0])}
    else :
        return {"result" : "None"}
    
#로그 추출 후, 시뮬레이션 테스트용
original_line = [
    {"time" : "20:00", "BENOIT" : "You think you can outsmart me? This isn’t just about money. This is about power, betrayal, and what you all are hiding from me."},
    {"time" : "20:01", "MARTA" : "Please, you have to believe me. I don’t know anything about what happened!"},
    {"time" : "20:02", "RANSOM" : "She’s obviously lying, detective. Why don’t you focus on the real criminals in this house?"},
    {"time" : "20:03", "LINDA" : "Detective, you don’t need to yell at everyone! This is still our family."},
    {"time" : "20:04", "FRAN" : "I’ve been keeping this secret for too long… but maybe I shouldn’t have said anything at all."},
    {"time" : "20:05", "BENOIT" : "Shut up, all of you! I’ll decide who’s guilty here. I see through your pathetic lies and your attempts to manipulate me."},
    {"time" : "20:06", "MARTA" : "I swear, I found it like that. I don’t know who changed the papers."},
    {"time" : "20:07", "RANSOM" : "Of course she’d say that. She’s been playing the innocent act since day one."},
    {"time" : "20:08", "LINDA" : "Detective, this is outrageous! Accusing without proof? That’s not justice."},
    {"time" : "20:09", "FRAN" : "I did see someone that night… but maybe I shouldn’t say anything. I’m scared."},
    {"time" : "20:10", "BENOIT" : "Enough of this back and forth! You think I don’t see what’s happening here? Fran, stop playing coy and spill it."},
    {"time" : "20:11", "MARTA" : "Please, Fran, just tell him the truth. We’re running out of time."},
    {"time" : "20:12", "RANSOM" : "Fran, don’t do this. You’ll only make things worse for yourself."},
    {"time" : "20:13", "LINDA" : "Detective, this isn’t how you solve a case. You’re bullying people into talking."},
    {"time" : "20:14", "FRAN" : "I saw Marta go into the library that night. That’s all I remember! I swear."},
    {"time" : "20:15", "BENOIT" : "Marta? So it was you all along! You played us all like fools. I should’ve known."},
    {"time" : "20:16", "MARTA" : "No, I didn’t! You’re making a mistake. Please, listen to me!"},
    {"time" : "20:17", "RANSOM" : "Finally, the detective sees the truth. Marta’s been lying from the start."},
    {"time" : "20:18", "LINDA" : "This is insane. I can’t believe you’re accusing Marta without solid evidence."},
    {"time" : "20:19", "FRAN" : "Wait… maybe I was wrong. It was dark, and I didn’t see her face clearly."},
    {"time" : "20:20", "BENOIT" : "You’re all wasting my time with excuses. Marta, unless you can prove your innocence, I’m taking you in."},
    {"time" : "20:21", "MARTA" : "This document… it proves I wasn’t there when it happened. Please, you have to look at it."},
    {"time" : "20:22", "RANSOM" : "She’s bluffing! Don’t believe her."},
    {"time" : "20:23", "LINDA" : "Detective, you need to calm down. Look at the evidence before making another mistake."},
    {"time" : "20:24", "FRAN" : "Wait, I remember now! It wasn’t Marta… it was Ransom. I saw him in the library before the accident."},
    {"time" : "20:25", "BENOIT" : "Ransom? Is this true? You better speak before I drag you out myself."},
    {"time" : "20:26", "MARTA" : "I’ve been saying it all along! I didn’t do anything!"},
    {"time" : "20:27", "RANSOM" : "Fran’s lying! She’s trying to save herself by throwing me under the bus."},
    {"time" : "20:28", "LINDA" : "Detective, this needs to stop. You’ve been wrong too many times tonight."},
    {"time" : "20:29", "FRAN" : "This is the truth. Ransom set everything up, and I saw him do it."},
    {"time" : "20:30", "BENOIT" : "I’ve had enough of your tricks, Ransom. The game is over. Your lies have caught up with you, and there’s no escaping now."},
    {"time" : "end" , "end" : "end"}]

current_line = original_line.copy()
@app.post("/chat")
async def chat(request : Request) :
    body = await request.body() 
    print("Raw Body:", body.decode("utf-8"))
    global current_line
    if not current_line:
        current_line = original_line.copy()
    result = current_line.pop(0)
    return result

@app.post("/plot_info")
async def plot_info(item : Plot):
#로그 추출 후, 시뮬레이션 테스트용
   return {
    'start_time': "20:00",
    'end_time': "20:30",
    'character_list': ['Benoit', 'Marta', 'Ransom', 'Linda', 'Fran'],
    'plan_list': [
        {
            'Benoit': [
                {'20:00': 'Library', '20:10': 'Library', '20:20': 'LivingRoom', '20:30': 'LivingRoom'}
            ]
        },
        {
            'Marta': [
                {'20:00': 'Kitchen', '20:10': 'DiningRoom', '20:20': 'LivingRoom', '20:30': 'Library'}
            ]
        },
        {
            'Ransom': [
                {'20:00': 'Exterior', '20:10': 'LivingRoom', '20:20': 'Library', '20:30': 'Library'}
            ]
        },
        {
            'Linda': [
                {'20:00': 'LivingRoom', '20:10': 'DiningRoom', '20:20': 'Library', '20:30': 'Library'}
            ]
        },
        {
            'Fran': [
                {'20:00': 'Bathroom', '20:10': 'LivingRoom', '20:20': 'Library', '20:30': 'Library'}
            ]
        }
    ]
}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)