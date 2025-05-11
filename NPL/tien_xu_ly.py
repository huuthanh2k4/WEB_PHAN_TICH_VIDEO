# File: NPL/tien_xu_ly.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# # # Chỉ cần chạy 1 lần để tải data NLTK
# # nltk.download('stopwords')
# # nltk.download('wordnet')

# class TienXuLy:
#     stop_words = set(stopwords.words('english'))
#     wn = WordNetLemmatizer()
#     extra_sw = {'rt', 'ht', 'fb', 'amp', 'gt'}

#     @staticmethod
#     def decontracted(text: str) -> str:
#         text = re.sub(r"won\'t", "will not", text)
#         text = re.sub(r"can\'t", "can not", text)
#         text = re.sub(r"n\'t", " not", text)
#         text = re.sub(r"\'re", " are", text)
#         text = re.sub(r"\'s", " is", text)
#         text = re.sub(r"\'d", " would", text)
#         text = re.sub(r"\'ll", " will", text)
#         text = re.sub(r"\'ve", " have", text)
#         text = re.sub(r"\'m", " am", text)
#         return text

#     @staticmethod
#     def clear_link(text: str) -> str:
#         text = re.sub(
#             r'((http|https)\:\/\/)?[A-Za-z0-9\./\?\:@\-_=#]+'
#             r'\.([A-Za-z]){2,6}([A-Za-z0-9\.\&/\?\:@\-_=#])*',
#             '', text, flags=re.MULTILINE
#         )
#         return re.sub(r'(@[^\s]+)', '', text)

#     @staticmethod
#     def clear_punctuation(text: str) -> str:
#         return re.sub(r'[^\w\s]', '', text)

#     @staticmethod
#     def clear_special(text: str) -> str:
#         return re.sub(r'[^A-Za-z]', ' ', text)

#     @classmethod
#     def clear_noise(cls, text: str) -> str:
#         t = text.lower()
#         t = cls.decontracted(t)
#         t = cls.clear_link(t)
#         t = cls.clear_punctuation(t)
#         return cls.clear_special(t)

#     @classmethod
#     def clear_stopwords(cls, text: str) -> str:
#         return " ".join(tok for tok in text.split()
#                         if tok not in cls.stop_words)

#     @classmethod
#     def black_txt(cls, tok: str) -> bool:
#         if tok == 'u':
#             tok = 'you'
#         return (tok not in cls.stop_words
#                 and tok not in string.punctuation
#                 and tok not in cls.extra_sw)

#     @classmethod
#     def fun_stemlem(cls, text: str) -> str:
#         out = []
#         for w in text.split():
#             w_low = w.lower()
#             if cls.black_txt(w_low):
#                 out.append(cls.wn.lemmatize(w, pos='v'))
#         return " ".join(out)

#     @classmethod
#     def prepare_data(cls, text: str) -> str:
#         t = cls.clear_noise(text)
#         t = cls.clear_stopwords(t)
#         return cls.fun_stemlem(t)


import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Các hàm xử lý bên ngoài (có thể gọi trực tiếp hoặc gói trong class)
def decontracted(st):
    st = re.sub(r"won\'t", "will not", st)
    st = re.sub(r"can\'t", "can not", st)
    st = re.sub(r"n\'t", " not", st)
    st = re.sub(r"\'re", " are", st)
    st = re.sub(r"\'s", " is", st)
    st = re.sub(r"\'d", " would", st)
    st = re.sub(r"\'ll", " will", st)
    st = re.sub(r"\'ve", " have", st)
    st = re.sub(r"\'m", " am", st)
    return st

def clear_link(st):
    return re.sub(r'(https?://\S+|\S+@\S+)', '', st)

def clear_punctuation(st):
    punct_to_remove = string.punctuation.replace('#', '')
    table = str.maketrans('', '', punct_to_remove)
    return st.translate(table)

def clear_stopwords(st, stop_set):
    return " ".join(tok for tok in st.split() if tok not in stop_set)

def black_txt(token, stop_set, my_sw):
    return (token not in stop_set
            and token not in my_sw
            and token.isalpha())

def fun_stemlem(st, wn, stop_set, my_sw):
    cleaned = []
    for w in st.split():
        if black_txt(w.lower(), stop_set, my_sw):
            cleaned.append(wn.lemmatize(w, pos='v'))
    return " ".join(cleaned)

class TienXuLy:
    def __init__(self):
        self.wn = WordNetLemmatizer()
        self.stop = set(stopwords.words('english'))
        self.my_sw = ['rt', 'ht', 'fb', 'amp', 'gt']

    def decontracted(self, st: str) -> str:
        return decontracted(st)

    def clear_link(self, st: str) -> str:
        return clear_link(st)

    def clear_punctuation(self, st: str) -> str:
        return clear_punctuation(st)

    def clear_stopwords(self, st: str) -> str:
        return clear_stopwords(st, self.stop)

    def fun_stemlem(self, st: str) -> str:
        return fun_stemlem(st, self.wn, self.stop, self.my_sw)

    def prepare_data(self, text: str) -> str:
        """
        Thực hiện full pipeline: lowercase, decontract, remove links,
        remove punctuation (giữ '#'), loại stopwords, và lemmatization.
        """
        text = text.lower()
        text = self.decontracted(text)
        text = self.clear_link(text)
        text = self.clear_punctuation(text)
        text = self.clear_stopwords(text)
        text = self.fun_stemlem(text)
        return text

# Ví dụ thử nghiệm
if __name__ == "__main__":
    processor = TienXuLy()
    s = "omg new track ❤️ I’m totally obsessed with this melody! #musiclove"
    print("Trước xử lý:", s)
    print("Sau xử lý :", processor.prepare_data(s))
