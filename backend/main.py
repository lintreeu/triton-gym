from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # ← 加這行
from pydantic import BaseModel
import triton_runner

app = FastAPI()

# 加入這段 CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或改成 ["http://127.0.0.1:5173"] 以更安全
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str
    size: int = 512

@app.post("/run")
async def run_kernel(req: CodeRequest):
    try:
        result = triton_runner.run_vector_add_kernel(req.code, req.size)
        return {"result": result.tolist()}
    except Exception as e:
        import traceback
        traceback.print_exc()  # ← 加這行印出完整錯誤
        raise HTTPException(status_code=500, detail=str(e))
