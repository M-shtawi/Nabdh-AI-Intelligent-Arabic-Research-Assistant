import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from datetime import datetime
import os
import subprocess
import time
import re

# ===== التحقق من وجود النماذج =====
def check_and_install_models():
    required_models = ["qwen:14b-chat"]
    
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        installed_models = result.stdout.lower()
        
        for model in required_models:
            if model not in installed_models:
                st.warning(f"النموذج {model} غير مثبت. جاري التثبيت الآن...")
                subprocess.run(["ollama", "pull", model], check=True)
                time.sleep(3)
        
        st.success("✅ تم تثبيت النماذج بنجاح!")
    except Exception as e:
        st.error(f"❌ فشل في تثبيت النماذج: {str(e)}")
        st.stop()

# ===== إعدادات التصميم =====
st.markdown("""
<style>
:root {
    --primary: #2563EB;
    --secondary: #EFF6FF;
    --background: #FFFFFF;
    --text: #1E3A8A;
    --border: #E5E7EB;
    --accent: #FF6B6B;
}
.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'Tajawal', sans-serif;
}
.header {
    background: linear-gradient(135deg, var(--primary), #1E3A8A);
    color: white;
    padding: 1.5rem;
    border-radius: 0 0 20px 20px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.user-msg {
    background: var(--primary);
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 1rem 1.25rem;
    margin-left: 30%;
    margin-bottom: 1rem;
    max-width: 70%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    animation: fadeIn 0.3s ease;
}
.assistant-msg {
    background: var(--secondary);
    color: var(--text);
    border-radius: 18px 18px 18px 0;
    padding: 1rem 1.25rem;
    margin-right: 30%;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    max-width: 70%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    animation: fadeIn 0.4s ease;
}
.stTextInput input {
    border-radius: 24px !important;
    padding: 12px 16px !important;
    border: 1px solid var(--border) !important;
    font-size: 1rem;
}
.stButton>button {
    border-radius: 12px !important;
    padding: 10px 20px !important;
    background-color: var(--primary) !important;
    color: white !important;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    opacity: 0.9;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}
.sidebar-section {
    padding: 1.2rem;
    background: white;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border: 1px solid var(--border);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ===== نظام الذكاء الاصطناعي =====
class AISystem:
    def __init__(self, model_name="qwen:14b-chat"):
        self.model_name = model_name
        try:
            self.llm = Ollama(model=model_name)
            self.db = self._init_db()
            st.success(f"✅ تم تحميل النموذج: {model_name}")
        except Exception as e:
            st.error(f"❌ خطأ في تهيئة النظام: {str(e)}")
            st.stop()
    
    def _init_db(self):
        try:
            os.makedirs("documents", exist_ok=True)
            files = os.listdir("documents")
            if files:
                with st.spinner("جاري تحميل الوثائق..."):
                    loader = DirectoryLoader("documents", glob="**/*.*")
                    docs = loader.load()
                    if docs:
                        embeddings = OllamaEmbeddings(model=self.model_name)
                        return Chroma.from_documents(docs, embeddings, persist_directory="db")
            return None
        except Exception as e:
            st.error(f"❌ خطأ في تحميل الوثائق: {str(e)}")
            return None
    
    def get_response(self, prompt):
        try:
            if self.db:
                qa = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=self.db.as_retriever(),
                    chain_type="stuff"
                )
                response = qa.run(prompt)
            else:
                response = self.llm(prompt)
            
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            for phrase in ["Hmm", "همم", "Let me think", "جاري التحليل", "..."]:
                response = response.replace(phrase, "")
            
            return response.strip()
        except Exception as e:
            return f"❌ عذرًا، حدث خطأ: {str(e)}"

# ===== إعداد التطبيق =====
def setup_app():
    with st.spinner("جاري التحقق من النماذج المثبتة..."):
        check_and_install_models()
    
    if "ai_system" not in st.session_state:
        st.session_state.ai_system = AISystem()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        add_message("assistant", "مرحبًا بك في النظام الذكي لجهاز الموانئ. أنا المساعد القانوني المدعوم بـ qwen:14b-chat. كيف يمكنني مساعدتك؟")

# ===== وظائف المساعدة =====
def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%H:%M")
    })

def save_conversation():
    conversation = "\n\n".join(
        f"{msg['role']} ({msg['time']}):\n{msg['content']}" 
        for msg in st.session_state.messages
    )
    st.download_button("⬇️ تنزيل المحادثة", conversation, "mawani_chat_history.txt", "text/plain")

def display_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>أنت</b></span><span>{msg["time"]}</span>
                </div><div>{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-msg">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--primary);"><b>المساعد الذكي</b></span><span>{msg["time"]}</span>
                </div><div>{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# ===== الواجهة الرئيسية =====
def main():
    st.markdown("""
    <div class="header">
        <h1 style="margin: 0; font-size: 2rem;">المساعد القانوني الذكي</h1>
        <p style="margin: 0.5rem 0 0;">نظام ذكاء اصطناعي لجهاز الموانئ</p>
    </div>
    """, unsafe_allow_html=True)
    
    setup_app()
    
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚙️ إعدادات النظام")
        st.markdown("**النموذج الحالي:** `qwen:14b-chat`")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 💬 إدارة المحادثة")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("♻️ إعادة تعيين", help="مسح السجل"):
                st.session_state.messages = []
                add_message("assistant", "تمت إعادة تعيين المحادثة.")
                st.rerun()
        with col2:
            if st.button("💾 حفظ المحادثة"):
                save_conversation()
        st.markdown('</div>', unsafe_allow_html=True)
    
    display_messages()
    
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("اكتب سؤالك هنا...", label_visibility="collapsed", placeholder="اكتب سؤالك...")
        submitted = st.form_submit_button("إرسال →")
        if submitted and prompt:
            add_message("user", prompt)
            try:
                response = st.session_state.ai_system.get_response(prompt)
                add_message("assistant", response)
                st.rerun()
            except Exception as e:
                st.error(f"❌ خطأ: {str(e)}")

if __name__ == "__main__":
    main()