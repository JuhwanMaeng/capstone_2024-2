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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
