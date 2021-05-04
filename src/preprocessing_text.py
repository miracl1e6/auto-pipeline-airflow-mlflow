import re

import numpy as np
from pymystem3 import Mystem


def remove_emoji(string):
    """
    Удаление эмоджи из текста
    """
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_links(string):
    """
    Удаление ссылок
    """
    string = re.sub(r'http\S+', '', string)  # remove http links
    string = re.sub(r'bit.ly/\S+', '', string)  # rempve bitly links
    string = re.sub(r'www\S+', '', string)  # rempve bitly links
    string = string.strip('[link]')  # remove [links]
    return string


def preprocessing(string, stopwords, stem):
    """
    Простой препроцессинг текста, очистка, лематизация, удаление коротких слов
    """
    string = remove_emoji(string)
    string = remove_links(string)

    # удаление символов "\r\n"
    str_pattern = re.compile("\r\n")
    string = str_pattern.sub(r'', string)

    # очистка текста от символов
    string = re.sub('(((?![а-яА-Я ]).)+)', ' ', string)
    # лематизация
    string = ' '.join([
        re.sub('\\n', '', ' '.join(stem.lemmatize(s))).strip()
        for s in string.split()
    ])
    # удаляем слова короче 3 символов
    string = ' '.join([s for s in string.split() if len(s) > 3])
    # удаляем стоп-слова
    string = ' '.join([s for s in string.split() if s not in stopwords])
    return string


def get_clean_text(data, stopwords):
    """
    Получение текста в преобразованной после очистки
    матричном виде, а также модель векторизации
    """
    # Простой препроцессинг текста
    stem = Mystem()
    comments = [preprocessing(x, stopwords, stem) for x in data]
    # Удаление комментов, которые имеют меньше, чем 5 слов
    comments = [y for y in comments if len(y.split()) > 5]
    # common_texts = [i.split(' ') for i in comments]
    return comments


def vectorize_text(data, tfidf):
    """
    Получение матрицы кол-ва слов в комменариях
    Очистка от пустых строк
    """
    # Векторизация
    X_matrix = tfidf.transform(data).toarray()
    # Удаляем строки в матрице с пустыми значениями
    mask = (np.nan_to_num(X_matrix) != 0).any(axis=1)
    return X_matrix[mask]
