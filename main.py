import nltk

from accuracy_testing import evaluate_on_test_set, calculate_accuracy, calculate_exact_match_accuracy
from predicting import predict_movie_genres
from probabilities import calculate_apriori_prob, calculate_cond_prob
from processing import init_dataframe

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


train_df, test_df = init_dataframe()

apriori_probabilities = calculate_apriori_prob(train_df)
conditional_probabilities, vocabulary_size, genre_total_words = calculate_cond_prob(train_df)

model_predictions, correct_genres = evaluate_on_test_set(test_df, apriori_probabilities, conditional_probabilities,
                                                         genre_total_words, vocabulary_size)

accuracy = calculate_accuracy(model_predictions, correct_genres, top_n=3)
print(f"Acuratetea modelului: {accuracy: .2%}")

exact_match_accuracy = calculate_exact_match_accuracy(model_predictions, correct_genres)
print(f"Acuratetea exact-match a modelului: {exact_match_accuracy: .2%}")


while True:
    movie_title = input("Introduceti numele unui film: ")
    predict_movie_genres(movie_title, apriori_probabilities, conditional_probabilities, genre_total_words,
                         vocabulary_size, top_n=3)
    more = input("One more time?: (y/n)").strip()
    if more == 'y':
        continue
    else:
        break
