from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI()

# CORS 설정: 모든 출처에서 오는 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안 상 실제 배포 시에는 특정 도메인만 허용하는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/network", response_class=HTMLResponse)
async def get_network_graph(request: Request):
    return templates.TemplateResponse("network.html", {"request": request})

@app.get("/scatter_data")
async def get_scatter_data():
    return {"x": ["기", "승", "전", "결"], "y": [0, 3, 5, 2]}

@app.get("/radar_data")
async def get_radar_data():
    return {"r": [5, 4, 3, 2, 4,5], "theta": ["Extroversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness", "Extroversion"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
