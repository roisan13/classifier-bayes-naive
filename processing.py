import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def init_dataframe():
    # Citirea datelor
    # Datele sunt un csv cu titlurile, genurile si rezumatul filmelor
    df = pd.read_csv('imdb_top_1000.csv', usecols=['Series_Title', 'Genre', 'Overview'])

    df['Processed_Genre'] = df['Genre'].apply(preprocess_text)
    df['Processed_Overview'] = df['Overview'].apply(preprocess_text)

    # 80% set de antrenament, 20% set de test
    train_dataframe, test_dataframe = train_test_split(df, test_size=0.2)

    # Trebuie resetat index-ul, altfel ramane shuffled
    train_dataframe = train_dataframe.reset_index(drop=True)
    test_dataframe = test_dataframe.reset_index(drop=True)

    return train_dataframe, test_dataframe


# Functia de preprocesare a textului
def preprocess_text(text):
    # Iau textul, il fac lowercase si elimin semnele de punctuatie
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Creez o lista de subsiruri din ce a ramas din text, ii fac lematizare si elimin stop_words
    # lemmatizare : (better -> good), (running -> run)  stop_words : "the", "and", "so", "him"
    tokens = word_tokenize(text)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return processed_tokens
