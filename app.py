from flask import Flask, render_template, request, Response
import openai
import faiss
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from collections import deque
import os
from dotenv import load_dotenv

app = Flask(__name__)

# .env에서 OpenAI 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 데이터 로드
documents = SimpleDirectoryReader('./data').load_data()
faiss_index = faiss.IndexFlatL2(1536)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

query_engine = index.as_query_engine(
    system_prompt="당신은 강의 내용에 대해 답변하는 도우미입니다. 반드시 한국어로 답변해주세요."
)

# 단기 기억 (최근 6개 대화 저장)
conversation_history = deque(maxlen=6)

# 어조 선택 딕셔너리(예시)
tones = {
    "a": "공격적이고 빈정대는",
    "b": "선생님처럼 따뜻하고 정중한",
    "c": "친구 같은 편한"
}

# 벡터 DB에서 질문 처리
def generate_answer(question):
    response = query_engine.query(question)
    return response.text if hasattr(response, 'text') else str(response)


# 스트리밍 응답 함수
def stream_response(response_text):
    chunk_size = 30
    for i in range(0, len(response_text), chunk_size):
        yield response_text[i:i + chunk_size]


# 신뢰도 검증 함수
def verify_answer(question, answer, tone):

    verification_prompt = f"이 질문에 대한 답변이 정확한지 평가해 주세요. 답변은 {tones[tone]} 어조를 반영해야 하지만, 정확성만 평가해야 합니다. 질문: {question} 답변: {answer}. 답변의 정확성만 평가하고, 예/아니오로 답해주세요."

    verification_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"당신은 답변의 정확성을 평가하는 역할을 합니다. 답변은 {tones[tone]} 어조를 반영해야 하지만, 정확성만 평가해야 합니다."},
            {"role": "user", "content": verification_prompt}
        ]
    )

    # 검증 결과 추출
    content = verification_response.choices[0].message.content.strip().rstrip('.')
    print(f"[검증 결과] 질문: {question} | 검증 결과: {content}")
    return content


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    tone = data.get('tone') # 숫자로 입력
    print(f"[질문] {question}")

    # 대화 기록 포함한 프롬프트 생성
    context = "\n".join([f"{role}: {content}" for role, content in conversation_history])
    full_prompt = f"이전 대화:\n{context}\n\n사용자 질문: {question}\n\n한국어로 답변해주세요. {tones[tone]} 어조로 답변해주세요."

    # 사용자 질문 추가
    conversation_history.append(("사용자", question))

    # 벡터 DB에서 답변 생성
    answer = generate_answer(full_prompt)
    print(f"[초기 답변] {answer}")

    # 검증
    verification_result = verify_answer(question, answer, tone)

    # 검증 결과가 "아니오"이면 새로운 답변 생성
    if verification_result in ["아니오", "아니요"]:
        print("[답변 검증] 부적절한 답변으로 간주하여 새 답변을 생성합니다.")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 강의 내용에 대해 정확한 답변을 제공하는 도우미입니다. 반드시 한국어로 답변해주세요. {tones[tone]} 어조로 답변해주세요."},
                {"role": "user", "content": full_prompt}
            ]
        )
        answer = response.choices[0].message.content

    print(f"[최종 답변] {answer}")

    # AI 응답 추가
    conversation_history.append(("AI", answer))
    print(f"[대화 기록] {list(conversation_history)}")

    # 스트리밍 응답 반환
    return Response(stream_response(answer), content_type='text/plain; charset=utf-8')


if __name__ == '__main__':
    app.run(debug=True)
