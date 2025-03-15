# ai-tutoring  

## 가상환경 생성 (Windows 기준)  
터미널에서 프로젝트 루트 디렉터리로 이동 후 아래 명령어 실행  

```bash
python -m venv web  # 가상환경 생성
web\Scripts\activate  # 가상환경 활성화
```

## 패키지 설치
```bash
pip install flask llama-index openai
pip install llama-index-vector-stores-faiss
pip install faiss-cpu
pip install python-dotenv
```

## 파일 추가
- 프로젝트 루트에 .env 파일 생성
- 아래 내용 추가
```ini
OPENAI_API_KEY=API키
```

## 실행
- 터미널에 다음 코드 실행
```bash
python app.py
```
- `script1`에는 한국사 강의의 스크립트가 포함되어 있습니다.
- 질문란에 **한국사와 관련된 질문만** 입력해주세요!
- 디버깅을 위해 터미널에도 출력값이 나옵니다.
