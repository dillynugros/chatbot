import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# KONFIGURASI TAMPILAN HALAMAN (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Asisten Layanan DJPb",
    page_icon="🏛️",
    layout="centered"
)

# Kustomisasi warna (Monokromatik Biru khas Perbendaharaan)
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f9;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-baseweb="block"]:nth-child(even) {
        background-color: #e3f2fd; /* Biru muda untuk user */
    }
    .stChatMessage[data-baseweb="block"]:nth-child(odd) {
        background-color: #ffffff; /* Putih untuk asisten */
        border: 1px solid #bbdefb;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Asisten Virtual Layanan Perbendaharaan")
st.caption("Layanan Informasi Berbasis Referensi Resmi - Kanwil DJPb")

# ==========================================
# LOGIKA RAG (Retrieval-Augmented Generation)
# ==========================================

@st.cache_resource(show_spinner="Sinkronisasi aturan (membutuhkan waktu beberapa menit untuk dokumen tebal)...")
def load_and_process_documents():
    # 1. Pastikan folder referensi ada
    if not os.path.exists("referensi"):
        os.makedirs("referensi")
        return None

    # 2. Baca semua PDF di folder referensi
    loader = PyPDFDirectoryLoader("referensi/")
    documents = loader.load()
    
    if not documents:
        return None

    # 3. Memecah dokumen menjadi potongan kecil (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # 4. Inisialisasi model Embeddings Gemini (PERBAIKAN ERROR 404 DI SINI)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 5. BATCHING LOGIC: Mengatasi limit maksimal API Google
    batch_size = 90 # Batas aman di bawah 100
    vectorstore = None
    
    progress_text = "Memproses dokumen ke dalam database..."
    my_bar = st.progress(0, text=progress_text)
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if vectorstore is None:
            # Inisialisasi awal database vektor
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            # Menambahkan dokumen berikutnya ke dalam database vektor
            vectorstore.add_documents(batch)
        
        # Update progress bar
        current_batch = (i // batch_size) + 1
        progress_percentage = current_batch / total_batches
        my_bar.progress(progress_percentage, text=f"{progress_text} ({int(progress_percentage*100)}%)")
        
        # Jeda 2 detik untuk menghindari error "429 Too Many Requests" dari Google API
        time.sleep(2) 
            
    my_bar.empty() # Hilangkan progress bar setelah selesai
    return vectorstore

def get_conversational_chain(vectorstore):
    # Menggunakan Gemini 1.5 Flash yang cepat, temperature 0 agar jawaban faktual
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) 
    
    # Prompt super ketat untuk mencegah halusinasi
    system_prompt = (
        "Anda adalah Asisten Layanan yang profesional dan responsif. "
        "Gunakan HANYA potongan konteks referensi berikut untuk menjawab pertanyaan pengguna layanan (satker). "
        "Jika jawabannya tidak ada dalam konteks, Anda WAJIB menjawab: "
        "'Mohon maaf, berdasarkan dokumen referensi yang saya miliki, informasi tersebut tidak ditemukan. Silakan konsultasikan lebih lanjut dengan petugas CSO kami.' "
        "Dilarang keras menggunakan pengetahuan dari luar konteks, berasumsi, atau mengarang jawaban.\n\n"
        "Konteks referensi:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Mengambil 5 potongan teks paling relevan
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# ==========================================
# ANTARMUKA CHATBOT STREAMLIT
# ==========================================

# Inisialisasi API Key dari Streamlit Secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ API Key belum dikonfigurasi. Silakan atur GOOGLE_API_KEY di Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Inisialisasi memori percakapan
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait aturan dan layanan perbendaharaan hari ini?"}]

# Proses dokumen di latar belakang
vectorstore = load_and_process_documents()

if vectorstore is None:
    st.warning("Belum ada dokumen referensi. Silakan unggah file PDF aturan (misal: PMK) ke folder 'referensi/' di GitHub.")
else:
    # Tampilkan riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Tangkap input dari satker/pengguna
    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        # Tambahkan ke memori dan tampilkan di UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Proses jawaban
        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi aturan..."):
                try:
                    rag_chain = get_conversational_chain(vectorstore)
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Simpan jawaban ke memori
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses jawaban: {e}")
