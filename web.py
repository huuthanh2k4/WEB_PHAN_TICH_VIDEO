import os, time, tempfile, shutil
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import whisper
import joblib
from transformers.pipelines import pipeline
from yt_dlp import YoutubeDL
from NPL.tien_xu_ly import TienXuLy

# 0) UI tối thiểu để health-check pass ngay
st.set_page_config(page_title="Subtitle & Emotion Analyzer", layout="wide")
st.title("🎬 Phân tích phụ đề & cảm xúc từ video")

# --- 1) Định nghĩa cache resources, nhưng **không load** ngay ---
@st.cache_resource(show_spinner=False)
def get_whisper_model():
    return whisper.load_model("medium")

@st.cache_resource(show_spinner=False)
def get_translator(src_lang: str):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-en")

@st.cache_resource(show_spinner=False)
def get_tfidf():
    return joblib.load("Model đã huấn luyện/TF-IDF.sav")

@st.cache_resource(show_spinner=False)
def get_svc():
    return joblib.load("Model đã huấn luyện/svc_model.pkl")

processor = TienXuLy()

# --- 2) Input from user ---
st.sidebar.header("1. Chọn nguồn video")
mode = st.sidebar.radio("Chọn:", ("Tải lên file", "Nhập URL"))

video_path = None
if mode == "Tải lên file":
    uploaded = st.sidebar.file_uploader("Upload video (.mp4/.mov/.avi)", type=["mp4","mov","avi"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read()); tmp.close()
        video_path = tmp.name
elif mode == "Nhập URL":
    url = st.sidebar.text_input("Dán URL video (YouTube, TikTok, Vimeo…)")
    if url:
        # chỉ hiển thị nút download, nhưng không thực hiện ngay
        pass

# 3) Khi đủ input, hiển thị nút xử lý
if (mode=="Tải lên file" and video_path) or (mode=="Nhập URL" and url):
    if st.button("▶️ Xử lý video"):
        # Bắt đầu đo thời gian
        t0 = time.perf_counter()

        # 3.1) Nếu từ URL thì download
        if mode == "Nhập URL":
            try:
                with st.spinner("⏳ Đang tải video..."):
                    os.makedirs("temp_video", exist_ok=True)
                    ydl_opts = {
                        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                        "outtmpl": os.path.join("temp_video","%(id)s.%(ext)s"),
                        "merge_output_format":"mp4","quiet":True
                    }
                    with YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        video_path = ydl.prepare_filename(info)
                st.success(f"✔️ Tải xong: {os.path.basename(video_path)}")
            except Exception as e:
                st.error(f"❌ Tải video thất bại: {e}")
                st.stop()

        # 3.2) Load models (lần đầu sẽ cache)
        whisper_model = get_whisper_model()
        tfidf_model    = get_tfidf()
        svc_model      = get_svc()

        # 4) Transcribe
        st.header("2. Transcription")
        with st.spinner("⏳ Whisper transcription..."):
            trans = whisper_model.transcribe(video_path)
        segments = trans["segments"]
        lang = trans["language"]
        st.write(f"🔤 Ngôn ngữ phát hiện: **{lang}**")

        # 5) Dịch nếu cần
        if lang != "en":
            st.header("3. Translation")
            translator = get_translator(lang)
            with st.spinner("⏳ Dịch sang English..."):
                for s in segments:
                    try:
                        s["text"] = translator(s["text"])[0]["translation_text"]
                    except:
                        s["text"] = "[Lỗi dịch thuật]"

        # 6) Build DataFrame & preprocess
        st.header("4. Build DataFrame & Preprocessing")
        records = []
        for s in segments:
            txt = s["text"].strip()
            records.append({
                "start": s["start"],
                "end":   s["end"],
                "text":  txt,
                "features": processor.prepare_data(txt)
            })
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)

        # 7) TF-IDF & predict
        st.header("5. Emotion Classification")
        with st.spinner("⏳ TF-IDF & SVC predict..."):
            X = tfidf_model.transform(df["text"])
            preds = svc_model.predict(X)
        label_map = {
            "0":"buồn","1":"vui","2":"yêu",
            "3":"giận dữ","4":"sợ hãi","5":"ngạc nhiên","6":"ko chứa cảm xúc"
        }
        df["Cảm xúc"] = pd.Series(preds.astype(str)).map(label_map)
        st.dataframe(df[["start","end","text","Cảm xúc"]], use_container_width=True)

        # 8) Thống kê & biểu đồ
        t1 = time.perf_counter()
        elapsed = t1 - t0
        st.success(f"✅ Hoàn thành trong **{elapsed:.1f} giây**")

        counts = df["Cảm xúc"].value_counts()
        fig1, ax1 = plt.subplots()
        explode = [0.1 if v/counts.sum()<0.1 else 0 for v in counts.values]
        ax1.pie(counts.values, labels=counts.index, explode=explode,
                autopct="%1.1f%%", startangle=140, wedgeprops={"linewidth":1,"edgecolor":"white"})
        ax1.axis("equal"); st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.bar(counts.index, counts.values)
        ax2.set_xlabel("Cảm xúc"); ax2.set_ylabel("Số lượng"); ax2.set_title("Số lượng mỗi cảm xúc")
        st.pyplot(fig2)

        # 9) Download CSV
        st.header("Tải về kết quả")
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 Tải CSV", csv,
                           file_name=f"{os.path.splitext(os.path.basename(video_path))[0]}_subtitles.csv",
                           mime="text/csv")

        # 10) Cleanup tạm
        if mode=="Tải lên file" and video_path:
            os.remove(video_path)
        if mode=="Nhập URL":
            shutil.rmtree("temp_video", ignore_errors=True)
