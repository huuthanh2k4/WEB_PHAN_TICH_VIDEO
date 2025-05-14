import os
import time
import tempfile
import shutil
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import whisper
import joblib
from transformers.pipelines import pipeline
from yt_dlp import YoutubeDL
from NPL.tien_xu_ly import TienXuLy
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed")

# If you bundled ffmpeg_static, add to PATH
ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg_static")
ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg")
ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe")
if os.path.isfile(ffmpeg_exe) and os.path.isfile(ffprobe_exe):
    os.chmod(ffmpeg_exe, 0o755)
    os.chmod(ffprobe_exe, 0o755)
    os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH','')}"
    os.environ["FFMPEG_BINARY"] = ffmpeg_exe

st.set_page_config(page_title="Subtitle & Emotion Analyzer", layout="wide")
st.title("🎬 Phân tích phụ đề & cảm xúc từ video/audio")

# Helper to download video or audio
def download_media(url: str, out_dir: str = "temp_media") -> str:
    os.makedirs(out_dir, exist_ok=True)
    # If URL points to audio service, we'll extract MP3
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"

# Cache models & resources
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource(show_spinner=False)
def load_translator(src_lang: str):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-en")

@st.cache_resource(show_spinner=False)
def load_tfidf(path):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_svc(path):
    return joblib.load(path)

# Initialize
whisper_model = load_whisper_model()
processor     = TienXuLy()

# Sidebar: choose input
st.sidebar.header("1. Chọn nguồn media")
mode = st.sidebar.radio("Chọn:", ("Tải lên file", "Nhập URL"))
media_path = None

if mode == "Tải lên file":
    uploaded = st.sidebar.file_uploader("Chọn file (.mp4, .mov, .avi, .mp3)", type=["mp4","mov","avi","mp3"])
    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()
        out_suffix = suffix if suffix == ".mp3" else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=out_suffix) as tmp:
            tmp.write(uploaded.read())
            media_path = tmp.name

elif mode == "Nhập URL":
    url = st.sidebar.text_input("Nhập link YouTube/TikTok/Vimeo...")
    if url:
        try:
            with st.spinner("⏳ Đang tải media..."):
                media_path = download_media(url)
            st.sidebar.success(f"✔️ Đã tải về: {os.path.basename(media_path)}")
        except Exception as e:
            st.sidebar.error(f"❌ Tải media thất bại: {e}")
            st.stop()

if not media_path:
    st.sidebar.warning("Vui lòng cung cấp media để tiếp tục.")
    st.stop()
else:
    st.sidebar.success(f"✔️ Đã chọn: {Path(media_path).name}")

# Sidebar: choose classification model
st.sidebar.header("2. Chọn mô hình phân loại")
choice = st.sidebar.selectbox("Model:", ["7 loại", "6 loại"])
if choice == "7 loại":
    svc = load_svc("SVC7.sav")
    tfidf = load_tfidf("TF-IDF7.sav")
else:
    svc = load_svc("SVC6.sav")
    tfidf = load_tfidf("TF-IDF6.sav")

# Button to start
if not st.sidebar.button("▶️ Bắt đầu phân tích"):
    st.info("Nhấn 'Bắt đầu phân tích' để chạy pipeline.")
    st.stop()

# Auto-reset if media changes
if "last_media" not in st.session_state:
    st.session_state.last_media = None
if media_path and st.session_state.last_media != media_path:
    for k in list(st.session_state.keys()):
        if k != "last_media":
            del st.session_state[k]
    st.session_state.last_media = media_path

# Start timer
t0 = time.perf_counter()

# 2. Transcription
st.header("2. Transcription")
with st.spinner("⏳ Đang chạy Whisper transcription..."):
    result = whisper_model.transcribe(media_path)
segments = result["segments"]
lang     = result["language"]
st.write(f"🔤 Ngôn ngữ phát hiện: **{lang}**")

# 3. Translation if needed
if lang != "en":
    st.header("3. Translation")
    translator = load_translator(lang)
    for seg in segments:
        try:
            seg["text"] = translator(seg["text"])[0]["translation_text"]
        except:
            seg["text"] = "[Lỗi dịch thuật]"

# 4. Build DataFrame & preprocess
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

# 5. Emotion Classification
st.header("5. Emotion Classification")
with st.spinner("⏳ Đang phân loại cảm xúc..."):
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
st.success(f"✅ Hoàn thành trong **{t1-t0:.2f} giây**")

# 6. Visualization
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

# 7. Download CSV
st.header("7. Tải về kết quả")
csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("📥 Tải CSV", data=csv_bytes,
                   file_name=f"{Path(media_path).stem}_results.csv",
                   mime="text/csv")

# Reset button
if st.button("🔄 Phân tích media mới"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    if os.path.isdir("temp_media"):
        shutil.rmtree("temp_media", ignore_errors=True)
    st.experimental_rerun()

# Cleanup temp file
if mode == "Tải lên file":
    try: os.remove(media_path)
    except: pass
else:
    shutil.rmtree("temp_media", ignore_errors=True)
