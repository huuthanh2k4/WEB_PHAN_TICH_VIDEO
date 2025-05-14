import os
import time
import tempfile
import shutil
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import whisper
import joblib
from transformers.pipelines import pipeline
from yt_dlp import YoutubeDL
from NPL.tien_xu_ly import TienXuLy
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed")

# Add ffmpeg_static to PATH if present
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

# Page config
st.set_page_config(page_title="Subtitle & Emotion Analyzer", layout="wide")
st.title("🎬 Phân tích phụ đề & cảm xúc từ video")

# Download helper
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

# Load & cache models
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource(show_spinner=False)
def load_translator(src_lang: str):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-en")

@st.cache_resource(show_spinner=False)
def load_tfidf_model(path):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_svc_model(path):
    return joblib.load(path)

# Initialize
whisper_model = load_whisper_model()
processor     = TienXuLy()

# Sidebar: choose video
st.sidebar.header("1. Chọn nguồn video")
mode = st.sidebar.radio("Chọn:", ("Tải lên file", "Nhập URL"))
video_path = None
if mode == "Tải lên file":
    uploaded = st.sidebar.file_uploader("Chọn video", type=["mp4","mov","avi"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name
elif mode == "Nhập URL":
    url = st.sidebar.text_input("Nhập link video")
    if url:
        try:
            with st.spinner("⏳ Đang tải video..."):
                video_path = download_video(url)
            st.sidebar.success("✔️ Đã tải về")
        except Exception as e:
            st.sidebar.error(f"❌ Tải video thất bại: {e}")
            st.stop()

if not video_path:
    st.sidebar.warning("Vui lòng chọn video để tiếp tục.")
    st.stop()

# Sidebar: choose classification model
st.sidebar.header("2. Chọn mô hình phân loại")
model_choice = st.sidebar.selectbox("Model", ["7 loại", "6 loại"])
if model_choice == "7 loại":
    svc = load_svc_model("SVC7.sav")
    tfidf = load_tfidf_model("TF-IDF7.sav")
else:
    svc = load_svc_model("SVC6.sav")
    tfidf = load_tfidf_model("TF-IDF6.sav")

# Button to start analysis
if not st.sidebar.button("▶️ Bắt đầu phân tích"):
    st.info("Nhấn nút Bắt đầu phân tích để chạy pipeline.")
    st.stop()

# Auto-reset session when video changes
if "last_video" not in st.session_state:
    st.session_state.last_video = None
if video_path and st.session_state.last_video != video_path:
    for key in list(st.session_state.keys()):
        if key not in ("last_video",):
            del st.session_state[key]
    st.session_state.last_video = video_path

# Run pipeline
t0 = time.perf_counter()

st.header("2. Transcription")
with st.spinner("⏳ Đang chạy Whisper..."):
    transcription = whisper_model.transcribe(video_path)
segments = transcription["segments"]
lang = transcription["language"]
st.write(f"🔤 Ngôn ngữ phát hiện: **{lang}**")

# Optional translation
if lang != "en":
    st.header("3. Translation")
    translator = load_translator(lang)
    for seg in segments:
        try:
            seg["text"] = translator(seg["text"])[0]["translation_text"]
        except:
            seg["text"] = "[Lỗi dịch thuật]"

# Build DataFrame
st.header("4. Build DataFrame & Preprocessing")
records = []
for seg in segments:
    txt = seg["text"].strip()
    records.append({
        "start": seg["start"],
        "end":   seg["end"],
        "phụ đề": txt,
        "features": processor.prepare_data(txt)
    })
df = pd.DataFrame(records)
st.dataframe(df, use_container_width=True)

# Classification
st.header("5. Emotion Classification")
with st.spinner("⏳ Phân loại cảm xúc..."):
    X = tfidf.transform(df["features"])
    preds = svc.predict(X)
label_map = {
    "0":"buồn","1":"vui","2":"yêu",
    "3":"giận dữ","4":"sợ hãi","5":"ngạc nhiên","6":"ko chứa cảm xúc"
}
df["Cảm xúc"] = [label_map.get(str(p), "?") for p in preds]
st.dataframe(df[["start","end","phụ đề","Cảm xúc"]], use_container_width=True)

# Timing
t1 = time.perf_counter()
elapsed = t1 - t0
st.success(f"✅ Hoàn thành trong **{elapsed:.2f} giây**")

# Visualization
st.header("6. Thống kê Cảm xúc")
counts = df["Cảm xúc"].value_counts()
fig1, ax1 = plt.subplots()
explode = [0.1 if v/counts.sum()<0.1 else 0 for v in counts.values]
ax1.pie(counts.values, labels=counts.index, explode=explode,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"linewidth":1,"edgecolor":"white"})
ax1.axis("equal")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.bar(counts.index, counts.values)
ax2.set_xlabel("Cảm xúc"); ax2.set_ylabel("Số lượng")
ax2.set_title("Phân bố cảm xúc")
st.pyplot(fig2)

# Download CSV
st.header("7. Tải về kết quả")
csv_data = df.to_csv(index=False, encoding="utf-8-sig")
st.download_button("📥 Tải CSV", data=csv_data,
                   file_name=f"{Path(video_path).stem}_results.csv",
                   mime="text/csv")

# Reset button
if st.button("🔄 Phân tích video mới"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if os.path.isdir("temp_video"):
        shutil.rmtree("temp_video", ignore_errors=True)
    st.experimental_rerun()

# Cleanup temp file
if mode == "Tải lên file":
    try: os.remove(video_path)
    except: pass
else:
    shutil.rmtree("temp_video", ignore_errors=True)
