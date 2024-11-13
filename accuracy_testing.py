import numpy as np

from processing import preprocess_text


def predict_top_genres(synopsis, prior_probabilities, cond_prob, total_words_per_genre,
                       vocab_size, top_n=3):
    # Preprocesare sinopsis
    tokens = preprocess_text(synopsis)

    # Inițializam dicționarul pentru scorurile de probabilitate logaritmica
    genre_scores = {}

    # Calculam probabilitățile in LOGARITM, ca sa evitam problemele de underflow
    for genre, prior_prob in prior_probabilities.items():
        # Incepem cu log P(gen)
        score = np.log(prior_prob)

        # Adaugam log P(w | gen) pentru fiecare cuv
        for word in tokens:
            # obtinem prob cond P(word | gen) cu Laplace smoothing
            word_prob = cond_prob[genre].get(word, 1 / (total_words_per_genre[genre] + vocab_size))
            score += np.log(word_prob)

        # Salvam scorul
        genre_scores[genre] = score

    # Sortam genurile in ordine desc (din cauza log), selectam primele 3
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    top_genres = sorted_genres[:top_n]

    return top_genres


def evaluate_on_test_set(df, prior_probabilities, cond_prob, total_words_per_genre, vocab_size):
    predictions = []
    actual_genres = []

    for _, row in df.iterrows():
        synopsis = row['Overview']
        true_genres = row['Processed_Genre']  # lista de genuri de la IMDb
        num_genres = len(true_genres)  # nr. de genuri acordat de IMDb

        # Predict top genuri cu cate genuri mi-a dat IMDb-ul
        top_predicted_genres = predict_top_genres(synopsis, prior_probabilities, cond_prob,
                                                  total_words_per_genre, vocab_size, top_n=num_genres)

        # Deocamdata ma intereseaza doar genurile
        predicted_genres = [genre for genre, score in top_predicted_genres]

        predictions.append(predicted_genres)
        actual_genres.append(true_genres)

    return predictions, actual_genres


def calculate_exact_match_accuracy(predictions, actual_genres):
    correct_predictions = 0

    # Exact match inseamna ca daca filmul are atribuite genurile (x, y, z)
    # modelul aceleasi genuri, chiar daca in alta ordine (y, x, z)
    for predicted, actual in zip(predictions, actual_genres):

        if set(predicted) == set(actual):
            correct_predictions += 1

    return correct_predictions / len(actual_genres)


def calculate_accuracy(predictions, actual_genres, top_n=3):
    correct_predictions = 0
    total_predictions = len(predictions)

    # Daca modelul a ghicit macar 1 din genurile date de IMDb, o sa iau in considerare ca fiind corect
    for pred, actual in zip(predictions, actual_genres):
        if any(genre in actual for genre in pred[:top_n]):
            correct_predictions += 1

    return correct_predictions / total_predictions
