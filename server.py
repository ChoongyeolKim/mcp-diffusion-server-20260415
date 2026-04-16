import os
import requests
import base64  # 모듈 임포트
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
        
        result = response.json()
        # Forge는 images 리스트에 base64 문자열을 담아 보냅니다.
        image_b64 = result['images'][0]
        
        # 1. 로컬 저장 - 이미지 데이터 반환 직전에 추가 (디버깅용)
        # with open("test_output.png", "wb") as f:
        #     f.write(base64.b64decode(image_b64))
        # print("로컬에 test_output.png 저장 완료!")

        # 2. 클라이언트(Copilot/Inspector)에게 전달할 데이터 포맷
        # 반드시 '문자열' 하나만 깔끔하게 리턴하도록 합니다.
        # Copilot Studio가 이미지를 해석할 수 있도록 데이터 형식을 갖춰 반환합니다.
        # Markdown 형식을 사용하여 챗봇 창에서 바로 이미지를 보여주려 시도합니다.
        return f"data:image/png;base64,{image_b64}"
    
    except Exception as e:
        return f"연결 실패 ({url}): {str(e)}"

if __name__ == "__main__":
    # 환경 변수에 'SSE'라는 설정이 있으면 SSE 모드로, 아니면 기본(stdio) 모드로 실행
    transport_mode = os.getenv("MCP_TRANSPORT", "stdio")
    
    if transport_mode == "sse":
        # host="0.0.0.0"은 외부(Azure VM 밖) 접속을 허용하기 위해 필수입니다.
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        # 포트 사용을 하지 않음
        mcp.run()
