import importlib.metadata as metadata

# Danh sách package pip tương ứng với module bạn dùng
packages = [
    "streamlit",
    "pandas",
    "matplotlib",
    "openai-whisper",
    "joblib",
    "transformers",
    "torch",
    "yt-dlp",
    "nltk"
]

# Lấy version và ghi vào requirements.txt
with open("requirements.txt", "w", encoding="utf-8") as f:
    for pkg in packages:
        try:
            ver = metadata.version(pkg)
            f.write(f"{pkg}=={ver}\n")
        except metadata.PackageNotFoundError:
            # package chưa cài: ghi tên mà không có version
            f.write(f"{pkg}\n")

print("✅ Đã tạo/ghi đè requirements.txt với các phiên bản hiện tại.")
