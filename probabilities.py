import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

def calculate_apriori_prob(df):
    all_genres = pd.Series([genre for genres in df['Processed_Genre'] for genre in genres])
    genre_counts = pd.Series(all_genres).value_counts()

    prior_probabilities = {genre: count / len(df) for genre, count in genre_counts.items()}
    return prior_probabilities


def calculate_cond_prob(df):
    # Pare ciudat ca ii dau join din nou
    # Dar acum dau join la cuvinte lemmatizate, fara stop words si punctuatie
    processed_synopses = [' '.join(overview) for overview in df['Processed_Overview']]

    X = vectorizer.fit_transform(processed_synopses)
    words = vectorizer.get_feature_names_out()

    # De cate ori apare un cuvant in toate sinopsisurile unui gen word_counts_per_genre[gen][cuvant]
    word_counts_per_genre = defaultdict(lambda: defaultdict(int))

    # Cate cuvinte apar in toate sinopsisurile unui gen
    total_words_per_genre = defaultdict(int)

    for idx, rows in df.iterrows():
        genres = rows['Processed_Genre']
        word_counts = X[idx].toarray().flatten()  # Trebuie flatten aici pt ca imi intoarce [[...]]

        for genre in genres:
            for word_idx, count in enumerate(word_counts):
                word = words[word_idx]
                word_counts_per_genre[genre][word] += count
                total_words_per_genre[genre] += count

    vocab_size = len(words)
    cond_probabilities = defaultdict(dict)

    for genre in word_counts_per_genre:
        for word in words:
            count = word_counts_per_genre[genre].get(word, 0)

            # Laplace smoothing
            cond_probabilities[genre][word] = (count + 1) / (total_words_per_genre[genre] + vocab_size)

    return cond_probabilities, vocab_size, total_words_per_genre
