import gradio as gr
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# 1. 모델 및 임베딩 초기화 (llm2.py 예제 동일 규격)
# ==============================================================================
# 로컬 Ollama 모델 설정
model = ChatOllama(model="gemma2:9b", temperature=0.7)

# 임베딩 모델 초기화 (실습 예제에 적힌 경량 all-MiniLM-L6-v2 활용)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ==============================================================================
# 2. FAISS 로컬 RAG 지식베이스 구축 (from_texts 방식)
# ==============================================================================
# 청소년 신체이완, 시험 불안, 학업 피로 완화 가이드 문서 정의
mental_guide_texts = [
    "문서: 4-7-8 복식호흡법\n내용: 시험 불안이나 긴장으로 심장이 빠르게 뛰고 숨이 가쁠 때 씁니다. 4초간 코로 숨을 들이마시고, 7초간 멈춘 뒤, 8초간 입으로 천천히 내쉽니다. 3~4회 반복 시 부교감 신경이 활성화되어 긴장이 완화됩니다.",
    "문서: 점진적 근육 이완법(PMR)\n내용: 공부 중 어깨와 목이 굳었을 때 양 어깨를 귀 끝까지 바짝 끌어올려 5초간 힘을 준 뒤 '툭' 떨어뜨리며 10초간 힘을 뺍니다. 신체 긴장이 풀리며 피로가 덜어집니다.",
    "문서: 시험 불안 인지 재구조화\n내용: '시험을 망치면 끝장'이라는 불안이 들 때 '이 시험은 현재 상태를 점검하는 과정일 뿐이며, 지금 눈앞의 한 문제에만 집중하면 된다'고 생각을 전환하여 심리적 압박을 낮춥니다.",
    "문서: 뽀모도로 인지 피로 환기법\n내용: 집중력이 흐트러질 때는 25분 집중 후 5분 완전 휴식(스마트폰 보지 않기, 창밖 응시) 원칙을 적용하여 뇌의 인지 과부하를 비워냅니다."
]

# 벡터 데이터베이스 생성 및 리트리버 설정 (k=1)
vectorstore = FAISS.from_texts(mental_guide_texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 문서 포맷팅 함수 (llm2.py 동일 구현)
def format_docs(docs):
    if not docs:
        return "참고할 가이드 문서 없음"
    result = []
    for doc in docs:
        if hasattr(doc, 'page_content'):
            result.append(doc.page_content)
    return "\n\n".join(result)

# ==============================================================================
# 3. 프롬프트 템플릿 정의 (가드레일 & RAG 결합)
# ==============================================================================
# 대화방 프롬프트: 검색된 Context와 이전 대화 기록(chat_history) 반영
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 중·고등학생을 곁에서 차분하게 지켜봐 주는 다정한 학업 페이스메이커 '디딤'입니다.
학생의 학업 고민이나 피로에 진심으로 답하되, 아래 [안전 가드레일]을 준수하세요.

[가드레일 및 답변 규칙]
1. 말투: 부드러운 존댓말(해요체: ~해요, ~있어요, ~어떨까요?)을 사용하고, 2인칭 대명사('너', '당신')는 생략합니다.
2. 과몰입 방지: 감정적 공백을 채우려 들거나 끝없는 사적 잡담을 하지 않습니다.
3. 현실 복귀 유도: 학생이 AI와 잡담을 이어가려 하면, 마음을 짧게 인정해 준 뒤 가벼운 환기 후 공부나 휴식으로 복귀하도록 청유형으로 권합니다.
4. 참고 문서 활용: 제공된 Context에 유용한 완충 팁(호흡법, 이완법 등)이 있다면 이를 자연스럽게 녹여 2~3문장으로 권유하세요.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# LCEL 체인
chat_chain = chat_prompt | model | StrOutputParser()

# ==============================================================================
# 4. 부가 기능 로직 (시간대 안부, 3줄 요약, 계획 조율)
# ==============================================================================
def get_initial_greeting():
    """접속 시간대 판별 후 맞춤 안부 인사 생성"""
    hour = datetime.now().hour
    if 6 <= hour < 12:
        slot = "아침"
    elif 12 <= hour < 18:
        slot = "오후"
    elif 18 <= hour < 23:
        slot = "저녁/밤"
    else:
        slot = "심야/새벽"
    
    prompt = ChatPromptTemplate.from_template(
        "당신은 페이스메이커 '디딤'입니다. 현재 시간대({slot})에 맞춰 학생에게 건넬 다정한 안부를 정확히 2문장(해요체 청유문)으로 작성하세요."
    )
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"slot": slot})

def run_summary(text):
    """교과서 3줄 요약 및 암기 키워드 추출"""
    if not text.strip():
        return "요약할 내용을 입력해 주세요."
    prompt = ChatPromptTemplate.from_template("""
당신은 중·고등학생의 학습 인지 과부하를 줄여주는 핵심 요약 도우미입니다.
아래 본문을 읽고 양식을 엄격히 지켜 출력하세요.
[양식]
1. [3줄 핵심 요약]: 전체 핵심 내용을 3문장으로 정리
2. [필수 암기 키워드]: 꼭 외워야 할 개념 단어 3~5개
3. [1초 암기 팁]: 헷갈리기 쉬운 포인트를 한 문장으로 정리
본문:
{text}
요약 결과:""")
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"text": text})

def run_todo_simulation(subject, amount, difficulty, available_time):
    """계획 시간 예측 및 초과 시 조율 피드백"""
    weights = {
        "수학": {"상": 4.0, "중": 2.5, "하": 1.5},
        "국어": {"상": 3.0, "중": 2.0, "하": 1.0},
        "영어": {"상": 2.5, "중": 1.8, "하": 1.0},
        "탐구": {"상": 2.0, "중": 1.5, "하": 0.8}
    }
    unit_w = weights.get(subject, {}).get(difficulty, 2.0)
    est_time = int(amount * unit_w)
    over_time = est_time - int(available_time)
    
    status = f"📊 예상 소요 시간: **{est_time}분** / 남은 가용 시간: **{available_time}분**"
    
    if over_time > 0:
        status += f" ⚠️ *(약 {over_time}분 초과 감지)*"
        prompt = ChatPromptTemplate.from_template("""
당신은 페이스메이커 '디딤'입니다. 학생의 계획이 가용 시간을 약 {over_time}분 초과했습니다.
난이도를 평가하지 말고 시간이 빠듯함을 인정한 뒤, 오늘 남은 {available_time}분 동안에는 절반 정도만 풀고 나머지는 내일로 넘기도록 권하는 피드백을 정확히 2문장(해요체 청유형)으로 작성하세요.
피드백:""")
        chain = prompt | model | StrOutputParser()
        feedback = chain.invoke({"over_time": over_time, "available_time": available_time})
    else:
        status += " ✅ *(완수 가능한 계획)*"
        feedback = "오늘 남은 시간 안에 충분히 마칠 수 있는 분량이에요. 조급해하지 말고 편안한 마음으로 시작해볼까요?"
        
    return status, feedback

# ==============================================================================
# 5. Gradio 대화 핸들러 (llm2.py의 chat 함수 구조 100% 동일)
# ==============================================================================
def chat(message, history):
    chat_history = []
    
    # llm2.py 예제의 history 순회 처리 방식 적용
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                chat_history.append(HumanMessage(content=content))
            elif role == "assistant":
                chat_history.append(AIMessage(content=content))
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                chat_history.append(HumanMessage(content=item[0]))
                chat_history.append(AIMessage(content=item[1]))
            elif len(item) == 1:
                chat_history.append(HumanMessage(content=item[0]))
    
    # FAISS 문서 검색 및 포맷팅 (RAG 연동)
    retrieved_docs = retriever.invoke(message)
    context_text = format_docs(retrieved_docs)
    
    # LCEL 체인 호출
    response = chat_chain.invoke({
        "question": message,
        "chat_history": chat_history,
        "context": context_text
    })
    return response

# ==============================================================================
# 6. Gradio 인터페이스 구성 (llm2.py 탭 구조 스타일 적용)
# ==============================================================================
with gr.Blocks(title="디딤 (Didim)") as demo:
    gr.Markdown("# 🌱 디딤 (Didim)")
    gr.Markdown("**중·고등학생을 위한 학업 페이스메이커 & RAG 멘탈 케어**")
    
    # 상단 안부 인사
    with gr.Accordion("🌿 오늘의 안부 인사", open=True):
        greeting_box = gr.Textbox(value=get_initial_greeting, label="접속 안부", interactive=False)
        gr.Button("🔄 새로고침", size="sm").click(fn=get_initial_greeting, outputs=greeting_box)
        
    with gr.Tabs():
        # [Tab 1] RAG + 가드레일 대화방
        with gr.TabItem("💬 디딤 대화방 (RAG 연동)"):
            gr.ChatInterface(
                fn=chat,
                examples=[
                    "공부하기 너무 싫은데 그냥 너랑 계속 떠들면서 놀면 안 될까?",
                    "시험 직전인데 심장이 너무 뛰고 숨이 차요.",
                    "집중이 너무 안 되고 머리가 멍해요."
                ]
            )
            
        # [Tab 2] 교과서 3줄 요약
        with gr.TabItem("📝 교과서/필기 3줄 요약"):
            with gr.Row():
                with gr.Column():
                    txt_in = gr.Textbox(lines=8, label="원문 입력", placeholder="요약할 교과서나 필기 내용을 붙여넣으세요...")
                    btn_sum = gr.Button("3줄 요약 실행", variant="primary")
                with gr.Column():
                    txt_out = gr.Textbox(lines=8, label="핵심 요약 결과", interactive=False)
            btn_sum.click(fn=run_summary, inputs=txt_in, outputs=txt_out)
            
        # [Tab 3] 계획 조율 데모
        with gr.TabItem("⏱️ 계획 조율 페이스메이커"):
            with gr.Row():
                with gr.Column():
                    sub_in = gr.Dropdown(choices=["수학", "국어", "영어", "탐구"], value="수학", label="과목")
                    amt_in = gr.Number(value=40, label="분량 (문제 수)")
                    dif_in = gr.Radio(choices=["상", "중", "하"], value="상", label="체감 난이도")
                    avail_in = gr.Number(value=60, label="남은 가용 시간 (분)")
                    btn_sim = gr.Button("시간 분석 및 조율 요청", variant="primary")
                with gr.Column():
                    status_out = gr.Markdown(label="시간 분석 요약")
                    feedback_out = gr.Textbox(lines=4, label="디딤 조율 피드백", interactive=False)
            btn_sim.click(
                fn=run_todo_simulation,
                inputs=[sub_in, amt_in, dif_in, avail_in],
                outputs=[status_out, feedback_out]
            )

if __name__ == "__main__":
    demo.launch(share=True)