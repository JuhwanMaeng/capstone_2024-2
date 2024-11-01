from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi import FastAPI
from pydantic import BaseModel
import logging
from agent_modules.agent_settings import *
 
app = FastAPI()

# CORS 설정: 모든 출처에서 오는 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 및 템플릿 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

# 데이터를 반환하는 API 엔드포인트
@app.get("/scatter_data")
async def get_scatter_data():
    return {"x": ["기", "승", "전", "결"], "y": [0, 3, 5, 2]}

@app.get("/radar_data")
async def get_radar_data():
    return {"r": [5, 4, 3, 2, 4, 5], "theta": ["Extroversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness", "Extroversion"]}

@app.get("/tree_data")
async def get_tree_data():
    tree_data = {
        "name": "Murder at the Manor",  # 사건의 시작
        "subname": "Case ID: M-001",  # 사건 ID
        "fill": "darkred",
        "children": [
            { 
                "name": "Discovery of the Body",  # 첫 장면: 시신 발견
                "subname": "Location: Library",  
                "fill": "grey" 
            },
            { 
                "name": "Interview with the Butler",  # 증언을 통해 단서 수집
                "subname": "Key Witness",  
                "fill": "blue" 
            },
            { 
                "name": "Secret Passage Found",  # 중요한 단서 발견
                "subname": "Hidden behind a bookshelf",  
                "fill": "blue",
                "children": [
                    { 
                        "name": "Chase through the Passage",  # 추격 장면
                        "subname": "Suspenseful pursuit",  
                        "fill": "blue",
                        "children": [
                            { 
                                "name": "Confrontation in the Basement",  # 대결 장면
                                "subname": "Showdown with the suspect",  
                                "fill": "#d281d2" 
                            }
                        ]
                    },
                    { 
                        "name": "Discovery of the Murder Weapon",  # 범행 도구 발견
                        "subname": "Weapon: Dagger",  
                        "fill": "blue",
                        "children": [
                            { 
                                "name": "Fingerprint Analysis",  # 증거 분석 장면
                                "subname": "Identifying the culprit",  
                                "fill": "#d281d2" 
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return tree_data

@app.get("/graph_data")
async def get_graph_data():
    names = ["Marta Cabrera", "Harlan Thrombey", "Ransom Drysdale", "Linda Drysdale", "Benoit Blanc"]
    nodes = [
        {"id": 1, "shape": "circularImage", "image": "/static/img/1.png", "label": names[0]},
        {"id": 2, "shape": "circularImage", "image": "/static/img/2.png", "label": names[1]},
        {"id": 3, "shape": "circularImage", "image": "/static/img/3.png", "label": names[2]},
        {"id": 4, "shape": "circularImage", "image": "/static/img/4.png", "label": names[3]},
        {"id": 5, "shape": "circularImage", "image": "/static/img/5.png", "label": names[4]},
    ]
    edges = [
        {"from": 1, "to": 2},
        {"from": 1, "to": 3},
        {"from": 1, "to": 4},
        {"from": 1, "to": 5},
        {"from": 2, "to": 3},
        {"from": 2, "to": 4},
        {"from": 2, "to": 5},
        {"from": 3, "to": 4},
        {"from": 3, "to": 5},
        {"from": 4, "to": 5},
    ]
    return {"nodes": nodes, "edges": edges}



from fastapi import FastAPI
from pydantic import BaseModel
import logging
from agent_modules.agent_settings import *

# FastAPI
app = FastAPI()

import logging
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("output.log"), logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)

# 표준 출력 리디렉션
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

# Pydantic Type Class
class NPC_Chat(BaseModel):
    participant : list[str] 
    time : str
    place : str


class User_Chat(BaseModel) : 
    npc : str
    time : str
    place : str
    UUID : str
    message : str

class User_Chat_END(BaseModel):
    UUID : str


class Time(BaseModel):
    time : str

# Endpoints
@app.post("/npc_chat")
async def npc_chat(item : NPC_Chat):
    participant = item.participant
    time = convert_time(item.time)
    place = item.place
    turns = 0
    answer = f"{participant[1]} said Hello! {participant[0]}"
    agent_1 = participant[0]
    agent_2 = participant[1]
    chat_history = [f"{agent_1} said, Hi! {agent_2} How is it going?"]
    while turns < 10 :
        break_dialogue = False
        stay_in_dialogue, answer = Agents[agent_2].npc_dialogue(agent, chat_history, now=time, place=place)
        chat_history.append(answer)
        if not stay_in_dialogue:
            break_dialogue = True
        if break_dialogue:
            break
        stay_in_dialogue, answer = Agents[agent_1].npc_dialogue(agent, chat_history, now=time, place=place)
        chat_history.append(answer)
        if not stay_in_dialogue:
            break_dialogue = True
        if break_dialogue:
            break
        turns += 1
    for agent in participant :
        Agents[agent].calc_friendship(chat_history, now=time)

    return {"status" : "OK"}

@app.post("/user_chat")
async def user_chat(item : User_Chat):
    time = convert_time(item.time)
    message = f"User : f{item.message}"
    _, return_message = Agents[item.npc].dialogue(message, time, item.place)
    return_message = return_message.replace(f"{item.npc} said", "")
    return {"message" : return_message}


@app.post("/user_chat_end")
async def chat_end(item : User_Chat_END):
    print(item)
    return {"status" : "OK"}

@app.post("/time")
async def time(item : Time):
    time = convert_time(item.time)
    hour = time.strftime("%H:%M")
    time_table = []
    for agent in Agents :
        if hour == "00:00" :
            Agents[agent].make_daily_plan(time)
        Agents[agent].make_event(time)
        Agents[agent].change_status(time)
        time_table.append({agent : Agents[agent].plan[hour]})
    return {"plan" :time_table}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
