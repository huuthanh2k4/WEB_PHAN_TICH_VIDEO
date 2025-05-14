import os
import time
import tempfile
import shutil
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import whisper
import joblib
from transformers import pipeline
from yt_dlp import YoutubeDL
from NPL.tien_xu_ly import TienXuLy

# --- Thêm ffmpeg_static vào PATH để Whisper tìm được ffmpeg ---
ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg_static")
ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg")
ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe")
if os.path.isfile(ffmpeg_path) and os.path.isfile(ffprobe_path):
    try:
        os.chmod(ffmpeg_path, 0o755)
        os.chmod(ffprobe_path, 0o755)
    except Exception:
        pass
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_BINARY"] = ffmpeg_path


st.set_page_config(page_title="Subtitle & Emotion Analyzer", layout="wide")
st.title("🎬 Phân tích phụ đề & cảm xúc từ video")

# --- Utility: download any video URL to MP4 via yt-dlp ---
def download_video(url: str, out_dir: str = "temp_video") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

# --- Cache heavy resources to speed up reruns ---
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource(show_spinner=False)
def load_translator(src_lang: str):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-en")

@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    return joblib.load("TF-IDF6.sav")

@st.cache_resource(show_spinner=False)
def load_svc_model():
    return joblib.load("SVC6.sav")

@st.cache_resource(show_spinner=False)
def load_svc_max():
    return joblib.load("SVC7.sav")

@st.cache_resource(show_spinner=False)
def load_tfidf_model1():
    return joblib.load("TF-IDF7.sav")

# Load models & processor once
whisper_model = load_whisper_model()
processor     = TienXuLy()
tfidf_model6    = load_tfidf_model()
tfidf_model7    = load_tfidf_model1()
svc_model      = load_svc_model()
svc_max       =  load_svc_max()

# --- Sidebar: choose input method ---
st.sidebar.header("1. Chọn nguồn video")
mode = st.sidebar.radio("Chọn:", ("Tải lên file", "Nhập URL"))


st.sidebar.header("Chọn model để xử lý")
chon_model = st.sidebar.radio(
    "Model:",
    ("phân loại 1 có 7 loại", "phân loại 2 có 6 loại")
)

if chon_model == "phân loại 1 có 7 loại" :
    chon_model = svc_max
    tfidf = tfidf_model7
elif chon_model == "phân loại 2 có 6 loại" :
    chon_model = svc_model
    tfidf = tfidf_model6

video_path = None
if mode == "Tải lên file":
    uploaded = st.sidebar.file_uploader("Chọn video (.mp4/.mov/.avi)", type=["mp4", "mov", "avi"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
elif mode == "Nhập URL":
    url = st.sidebar.text_input("Nhập link video (YouTube, TikTok, Vimeo, ...):")
    if url:
        try:
            with st.spinner("⏳ Đang tải video..."):
                video_path = download_video(url)
            st.sidebar.success(f"✔️ Đã tải về: {os.path.basename(video_path)}")
        except Exception as e:
            st.sidebar.error(f"❌ Tải video thất bại: {e}")
            st.stop()

if not video_path:
    st.sidebar.warning("Vui lòng cung cấp video để bắt đầu.")
    st.stop()
else:
    st.sidebar.success(f"✔️ Đã chọn video: {os.path.basename(video_path)}")

# --- Auto-reset khi người dùng chọn video mới ---
if "last_video" not in st.session_state:
    st.session_state.last_video = None

if video_path and st.session_state.last_video != video_path:
    # Xóa mọi state cũ trừ last_video
    for key in list(st.session_state.keys()):
        if key != "last_video":
            del st.session_state[key]
    st.session_state.last_video = video_path

# --- Start processing ---
t0 = time.perf_counter()

st.header("2. Transcription")
with st.spinner("⏳ Đang chạy Whisper transcription..."):
    transcription = whisper_model.transcribe(video_path)
segments = transcription["segments"]
lang = transcription["language"]
st.write(f"🔤 Phát hiện ngôn ngữ: **{lang}**")

if lang != "en":
    st.header("3. Translation")
    with st.spinner("⏳ Đang dịch sang tiếng Anh..."):
        translator = load_translator(lang)
        for seg in segments:
            try:
                seg["phụ đề"] = translator(seg["phụ đề"])[0]["translation_text"]
            except:
                seg["phụ đề"] = "[Lỗi dịch thuật]"

st.header("4. Build DataFrame & Preprocessing")
records = []
for seg in segments:
    txt = seg["phụ đề"].strip()
    records.append({
        "start":    seg["start"],
        "end":      seg["end"],
        "phụ đề":     txt,
        "xử lý phụ đề cho model": processor.prepare_data(txt)
    })
df = pd.DataFrame(records)
st.dataframe(df, use_container_width=True)

st.header("5. Emotion Classification")
with st.spinner("⏳ Mã hoá TF-IDF và dự đoán cảm xúc..."):
    X = tfidf.transform(df["xử lý phụ đề cho model"])
    preds = chon_model.predict(X)
label_map = {
    "0": "buồn",    "1": "vui",    "2": "yêu",
    "3": "giận dữ", "4": "sợ hãi","5": "ngạc nhiên",
    "6": "ko chứa cảm xúc"
}
df["Cảm xúc"] = pd.Series(preds.astype(str)).map(label_map)
st.dataframe(df[["start", "end", "phụ đề", "Cảm xúc"]], use_container_width=True)

t1 = time.perf_counter()
elapsed = t1 - t0
if elapsed < 60:
    st.success(f"✅ Hoàn thành trong **{elapsed:.2f} giây**")
else:
    st.success(f"✅ Hoàn thành trong **{elapsed/60:.2f} phút**")

st.header("6. Thống kê Cảm xúc")
counts = df["Cảm xúc"].value_counts()
fig1, ax1 = plt.subplots()
explode = [0.1 if v/counts.sum() < 0.1 else 0 for v in counts.values]
ax1.pie(counts.values, labels=counts.index, explode=explode,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"linewidth":1, "edgecolor":"white"})
ax1.axis("equal")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(counts.index, counts.values)
ax2.set_xlabel("Cảm xúc")
ax2.set_ylabel("Số lượng")
ax2.set_title("Số lượng mỗi cảm xúc")
st.pyplot(fig2)

st.header("7. Tải về kết quả")
csv_data = df.to_csv(index=False, encoding="utf-8-sig")
st.download_button("📥 Tải CSV", data=csv_data,
                   file_name=f"{os.path.splitext(os.path.basename(video_path))[0]}_subtitles.csv",
                   mime="text/csv")

# --- Nút reset toàn bộ để phân tích video mới ---
if st.button("🔄 Phân tích video mới", key="reset_btn"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if os.path.isdir("temp_video"):
        shutil.rmtree("temp_video", ignore_errors=True)
    st.experimental_rerun()

# --- Cleanup file tạm nếu là upload ---
if mode == "Tải lên file" and video_path:
    os.remove(video_path)
elif mode == "Nhập URL" and video_path:
    shutil.rmtree("temp_video", ignore_errors=True)
