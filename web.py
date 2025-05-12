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
    # Thiết lập quyền thực thi
    try:
        os.chmod(ffmpeg_path, 0o755)
        os.chmod(ffprobe_path, 0o755)
    except Exception:
        pass
    # Thêm ffmpeg vào PATH và biến môi trường
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
    return joblib.load("Model đã huấn luyện/TF-IDF.sav")

@st.cache_resource(show_spinner=False)
def load_svc_model():
    return joblib.load("Model đã huấn luyện/svc_model.pkl")

# Load once
whisper_model = load_whisper_model()
processor     = TienXuLy()
tfidf_model    = load_tfidf_model()
svc_model      = load_svc_model()

# --- Sidebar: choose input method ---
st.sidebar.header("1. Chọn nguồn video")
mode = st.sidebar.radio("Chọn:", ("Tải lên file", "Nhập URL"))

video_path = None
if mode == "Tải lên file":
    uploaded = st.sidebar.file_uploader("Chọn video (.mp4/.mov/.avi)", type=["mp4", "mov", "avi"])
    if uploaded:
        # Lưu file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(uploaded.read())
            video_path = tmpfile.name
elif mode == "Nhập URL":
    url = st.text_input("Nhập link video (YouTube, TikTok, Vimeo, ...):")
    if not url:
        st.warning("Vui lòng nhập URL để tiếp tục.")
        st.stop()

    try:
        with st.spinner("⏳ Đang tải video..."):
            video_path = download_video(url)
        st.success(f"✔️ Đã tải về: {os.path.basename(video_path)}")
    except Exception as e:
        st.error(f"❌ Tải video thất bại. Vui lòng kiểm tra lại link.\nChi tiết lỗi: {e}")
        st.stop()

if not video_path:
    st.sidebar.warning("Vui lòng cung cấp video để bắt đầu.")
    st.stop()
else:
    st.sidebar.success(f"✔️ Sẵn sàng xử lý: {os.path.basename(video_path)}")

# Start timing
t0 = time.perf_counter()

# 2) Whisper transcription
st.header("2. Transcription")
with st.spinner("⏳ Đang chạy Whisper transcription..."):
    transcription = whisper_model.transcribe(video_path)
segments = transcription["segments"]
lang = transcription["language"]
st.write(f"🔤 Phát hiện ngôn ngữ: **{lang}**")

# 3) Optional translation to English
if lang != "en":
    st.header("3. Translation")
    with st.spinner("⏳ Đang dịch sang tiếng Anh..."):
        translator = load_translator(lang)
        for seg in segments:
            try:
                seg["text"] = translator(seg["text"])[0]["translation_text"]
            except:
                seg["text"] = "[Lỗi dịch thuật]"

# 4) Build DataFrame & preprocess text
st.header("4. Build DataFrame & Preprocessing")
records = []
for seg in segments:
    txt = seg["text"].strip()
    records.append({
        "start":    seg["start"],
        "end":      seg["end"],
        "text":     txt,
        "features": processor.prepare_data(txt)
    })
df = pd.DataFrame(records)
st.dataframe(df, use_container_width=True)

# 5) TF-IDF encoding and SVC prediction
st.header("5. Emotion Classification")
with st.spinner("⏳ Mã hoá TF-IDF và dự đoán cảm xúc..."):
    X = tfidf_model.transform(df["text"])
    preds = svc_model.predict(X)
label_map = {
    "0": "buồn",    "1": "vui",    "2": "yêu",
    "3": "giận dữ", "4": "sợ hãi","5": "ngạc nhiên",
    "6": "ko chứa cảm xúc"
}
df["Cảm xúc"] = pd.Series(preds.astype(str)).map(label_map)
st.dataframe(df[["start", "end", "text", "Cảm xúc"]], use_container_width=True)

# End timing
t1 = time.perf_counter()
elapsed = t1 - t0
if elapsed < 60:
    st.success(f"✅ Hoàn thành toàn bộ pipeline trong **{elapsed:.2f} giây**")
else:
    minutes = elapsed / 60
    st.success(f"✅ Hoàn thành toàn bộ pipeline trong **{minutes:.2f} phút**")

# 6) Statistics & Visualization
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

# 7) Download results as CSV
st.header("Tải về kết quả")
csv_data = df.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="📥 Tải CSV",
    data=csv_data,
    file_name=f"{os.path.splitext(os.path.basename(video_path))[0]}_subtitles.csv",
    mime="text/csv"
)

# Xóa file tạm thời nếu là file upload
if mode == "Tải lên file" and video_path:
    os.remove(video_path)
elif mode == "Nhập URL" and video_path:
    shutil.rmtree("temp_video", ignore_errors=True)
