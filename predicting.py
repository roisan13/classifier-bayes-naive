import requests

from accuracy_testing import predict_top_genres

# ar trebui sa pun asta in .gitignore
# dar oricum sunt gratis cheile de la OMDb
omdb_api_key = 'a3f7952'


def predict_movie_genres(movie, prior_probabilities, cond_prob, total_words_per_genre, vocab_size, top_n=3):
    synopsis, real_genres = get_movie_data(movie)

    if synopsis is None:
        print("Nu am gasit filmul :(")
        return

    print(f"Sinopsis film: {synopsis}")
    print(f"Genuri film: {real_genres}")

    predicted_genres = predict_top_genres(synopsis, prior_probabilities, cond_prob, total_words_per_genre, vocab_size,
                                          top_n=top_n)

    print("\n Modelul a prezis:")
    for genre, score in predicted_genres:
        print(f"- {genre}")


def get_movie_data(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={omdb_api_key}"

    response = requests.get(url)
    data = response.json()

    # If the movie was found, extract relevant details
    if data['Response'] == 'True':
        movie_synopsis = data['Plot']
        movie_genres = data['Genre'].split(', ')
        return movie_synopsis, movie_genres
    else:
        return None, None
