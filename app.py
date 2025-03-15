from flask import Flask, render_template, request, Response
import openai
import faiss
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from collections import deque
import os
from dotenv import load_dotenv

app = Flask(__name__)

# .env에서 openai 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 데이터 로드 ( 코드 병합 시 수정)
documents = SimpleDirectoryReader('./data').load_data()

faiss_index = faiss.IndexFlatL2(1536)

# llama_index에서 faiss로 벡터 DB 교체 과정
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

query_engine = index.as_query_engine()

# 단기 기억 (최근 6개 대화 저장)
conversation_history = deque(maxlen=6)

# 벡터DB에서 질문 처리
def generate_answer(question):
    # 벡터 DB 기반으로 답변 생성
    response = query_engine.query(question)
    if hasattr(response, 'text'):
        return response.text  # Response 객체에서 텍스트 추출
    return str(response)  # 응답을 문자열로 변환

# 스트리밍 응답 함수
def stream_response(prompt):
    # 벡터 DB에서 응답을 가져옴
    response = generate_answer(prompt)  # 벡터 DB에서 가져온 응답

    print("\n[API 응답 시작]")  # API 응답 시작 로그(디버깅)

    # 응답을 여러 번에 나누어 스트리밍
    chunk_size = 30  # 한번에 보낼 크기(나중에 조정)
    for i in range(0, len(response), chunk_size):
        chunk = response[i:i + chunk_size]
        yield chunk  # 각 청크를 스트리밍으로 전송

    print("\n[최종 응답] ", response)  # 전체 응답 출력(디버깅용)


# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 질문-답변 api
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')

    # 대화 기록 포함한 프롬프트 생성
    context = "\n".join([f"{role}: {content}" for role, content in conversation_history])  # 튜플을 문자열로
    full_prompt = f"이전 대화:\n{context}\n\n사용자 질문: {question}"

    # 대화 기록 추가(사용자)
    conversation_history.append(("사용자", question))

    print("\n[사용자 질문]")  # 입력한 질문 출력(디버깅)
    print(question)
    print("=" * 50)

    def generate():
        answer = ""
        for answer_chunk in stream_response(full_prompt):
            answer += answer_chunk
            yield answer_chunk  # 청크를 실시간으로 전달

        # 대화 기록 추가(ai)
        if answer.strip():
            conversation_history.append(("AI", answer))

        print("\n[답변]")  # 답변 출력(디버깅)
        print(answer)
        print("=" * 50)

        print("\n[대화기록]")  # 대화기록 출력(디버깅)
        for role, content in conversation_history:
            print(f"{role}: {content}")
        print("=" * 50)

    return Response(generate(), content_type='text/plain; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True)