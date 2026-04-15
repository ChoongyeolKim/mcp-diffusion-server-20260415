import os
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv

# 1. .env 파일 로드 (로컬 개발 환경용)
# Azure VM에서 환경변수를 직접 설정한 경우, .env 파일이 없어도 무시됩니다.
load_dotenv()

# 2. 환경변수 읽기 (기본값 설정)
# 로컬 .env 또는 Azure 환경변수에서 'FORGE_API_URL'을 가져오고, 없으면 localhost를 기본값으로 사용합니다.
FORGE_API_URL = os.getenv("FORGE_API_URL", "http://localhost:7860/sdapi/v1/txt2img")

# MCP 서버 초기화
mcp = FastMCP("Flux-Forge-Connector")

@mcp.tool()
def generate_image(prompt: str) -> str:
    """
    Azure VM의 Flux 모델을 사용하여 이미지를 생성합니다.
    """
    payload = {
        "prompt": prompt,
        "steps": 20,
        "cfg_scale": 1.0,
        "width": 1024,
        "height": 1024,
        "sampler_name": "Euler",
        "scheduler": "Simple",
        "override_settings": {
            "sd_model_checkpoint": "flux1-dev-fp8.safetensors"
        }
    }

    try:
        # 설정된 URL로 요청 (txt2img 엔드포인트가 URL에 포함되어 있지 않을 경우를 대비해 처리)
        url = FORGE_API_URL
        if not url.endswith('/sdapi/v1/txt2img'):
            url = url.rstrip('/') + '/sdapi/v1/txt2img'
            
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        
        return "이미지 생성이 완료되었습니다."
        
    except Exception as e:
        return f"연결 실패 ({url}): {str(e)}"

if __name__ == "__main__":
    mcp.run()
