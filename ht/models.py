import numpy as np
from tqdm import tqdm
import pandas as pd


class GraphMatrixModel:
    def __init__(self, item_entry_range, adjacency_matrix):
        self.item_entry_range = item_entry_range
        self.adjacency_matrix = adjacency_matrix


class MovieModel(GraphMatrixModel):
    def __init__(self, movies, ratings):
        """
        Initialization of movie model
        :param movies: movies
        :type movies pd.DataFrame
        :param ratings: ratings
        :type ratings pd.DataFrame
        """
        # Initialize total number of movies and users
        movie_count = len(np.unique(np.array(movies['movieId'])))
        user_count = len(np.unique(np.array(ratings['userId'])))
        genre_count = len(np.unique(np.array(movies['genres'])))
        self.movie_count = movie_count
        self.user_count = user_count
        self.genre_count = genre_count

        # Initialize user-item matrix and user-genre matrix
        user_movie_matrix = np.zeros((user_count, movie_count))
        user_genre_matrix = np.zeros((user_count, genre_count))
        print('Start constructing User-Movie matrix and User-Genre matrix')

        for userId in tqdm(range(user_count)):
            genre_dict = {}
            user_ratings = ratings[ratings.userId == userId + 1].drop(columns=['userId'])
            sum_user_ratings = user_ratings['rating'].sum()
            for index, row in user_ratings.iterrows():
                movieId = int(row['movieId'])
                rating = row['rating']
                genre = int(movies[movies.movieId == movieId]['genres'].iloc[0])
                if genre in genre_dict:
                    genre_dict[genre].append(rating)
                else:
                    genre_dict[genre] = [rating]
                user_movie_matrix[userId][movieId] = rating / sum_user_ratings

            for genre in genre_dict:
                user_genre_matrix[userId][genre] = np.mean(genre_dict[genre])

            genre_avg_rating_sum = np.sum(user_genre_matrix[userId])
            if genre_avg_rating_sum != 0:
                user_genre_matrix[userId] /= genre_avg_rating_sum

        print('User-Movie matrix and User-Genre matrix construction finished.')

        # Initialize genre-item matrix
        genre_movie_matrix = np.zeros((genre_count, movie_count))
        print('Start constructing Genre-Movie matrix')

        genre_dict = {}  # average genre rating dictionary for all items in each genre
        for movieId in tqdm(range(movie_count)):
            movie_avg_ratings = np.mean(np.array(ratings[ratings.movieId == movieId]['rating']))
            genre = int(movies[movies.movieId == movieId]['genres'].iloc[0])

            if np.isnan(movie_avg_ratings):
                movie_avg_ratings = 0.0
                sum_genre_movie_avg_ratings = 1.0
            else:
                genre_movies = np.array(movies[movies.genres == genre]['movieId'])

                if genre in genre_dict:
                    sum_genre_movie_avg_ratings = genre_dict[genre]
                else:
                    sum_genre_movie_avg_ratings = 0.0
                    for genre_movieId in genre_movies:
                        genre_movie_avg_ratings = np.mean(np.array(ratings[ratings.movieId == genre_movieId]['rating']))
                        if np.isnan(genre_movie_avg_ratings):
                            genre_movie_avg_ratings = 0.0
                        sum_genre_movie_avg_ratings += genre_movie_avg_ratings
                    genre_dict[genre] = sum_genre_movie_avg_ratings

            genre_movie_matrix[genre][movieId] = movie_avg_ratings / sum_genre_movie_avg_ratings

        print('Genre-Movie matrix construction finished.')

        # Combine block matrices
        adjacency_matrix = np.concatenate(
            [np.concatenate([np.zeros((user_count, user_count)),
                             user_genre_matrix.T,
                             user_movie_matrix.T]),
             np.concatenate([user_genre_matrix,
                             np.zeros((genre_count, genre_count)),
                             genre_movie_matrix.T]),
             np.concatenate([user_movie_matrix,
                             genre_movie_matrix,
                             np.zeros((movie_count, movie_count))])
             ],
            axis=1
        )
        super().__init__((user_count + genre_count, user_count + genre_count + movie_count),
                         adjacency_matrix)


class KDDModel(GraphMatrixModel):
    def __init__(self, user_count, item_count, data: pd.DataFrame):
        self.user_count = user_count
        self.item_count = item_count

        # Initialize user-item matrix and user-genre matrix
        user_item_matrix = np.zeros((user_count, item_count))

        for user_id in range(user_count):
            user_clicks = data[data['user_id'] == user_id]
            for _, row in user_clicks.iterrows():
                item_id = int(row['item_id'])
                user_item_matrix[user_id][item_id] = 1.0

        # Combine block matrices
        adjacency_matrix = np.concatenate(
            [np.concatenate([np.zeros((user_count, user_count)),
                             user_item_matrix.T]),
             np.concatenate([user_item_matrix,
                             np.zeros((item_count, item_count))]),
             ],
            axis=1
        )
        super().__init__((user_count, user_count + item_count),
                         adjacency_matrix)


class LDAModel:
    def __init__(self, gibbs_sampling_params, user_item_weight_matrix):
        """

        :param gibbs_sampling_params: Gibbs sampling parameters
        :type gibbs_sampling_params GibbsSamplingParam
        """
        alpha = gibbs_sampling_params.alpha
        beta = gibbs_sampling_params.beta
        user_item_weight_matrix = np.array(user_item_weight_matrix, dtype=np.int)

        # Initialize gibbs sampling parameters
        self.alpha = alpha
        self.beta = beta
        self.K = gibbs_sampling_params.topic_count
        self.iter_num = gibbs_sampling_params.iter_num

        # Initialize n matrices
        self.M = len(user_item_weight_matrix)
        self.N = len(user_item_weight_matrix[0])
        self.nuz_matrix = np.zeros((self.M, self.K)) + alpha
        self.niz_matrix = np.zeros((self.N, self.K)) + beta
        self.nz_matrix = np.zeros(self.K) + self.N * beta
        self.nu_matrix = np.zeros(self.M) + np.sum(user_item_weight_matrix) * alpha

        nuz = self.nuz_matrix
        niz = self.niz_matrix
        nz = self.nz_matrix
        nu = self.nu_matrix

        # Initialize Nu
        for i in range(self.M):
            nu[i] = np.add(nu[i], np.sum(user_item_weight_matrix[i]))

        # Initialize topic list
        z_sets = []
        for i in range(self.M):
            z_set = []
            for j in range(self.N):
                z_set_size = user_item_weight_matrix[i][j]
                if z_set_size > 0:
                    z_prob = np.divide(np.multiply(nuz[i], niz[j]), np.multiply(nz, nu[i]))
                    z_list = []
                    for w in range(z_set_size):
                        z = np.random.multinomial(1, z_prob / z_prob.sum()).argmax()
                        nuz[i][z] += 1
                        niz[j][z] += 1
                        nz[z] += 1
                        z_list.append(z)
                    z_set.append(z_list)
                else:
                    z_set.append([])
            z_sets.append(z_set)

        self.nuz_matrix = nuz
        self.niz_matrix = niz
        self.nz_matrix = nz
        self.z_sets = z_sets

    def gibbs_sampling(self):
        z_sets = self.z_sets
        nuz = self.nuz_matrix
        niz = self.niz_matrix
        nz = self.nz_matrix
        nu = self.nu_matrix

        for i in range(self.M):
            for j in range(self.N):
                z_set = z_sets[i][j]
                z_set_size = len(z_set)
                if z_set_size > 0:
                    for w in range(z_set_size):
                        z = z_set[w]
                        nuz[i][z] -= 1
                        niz[j][z] -= 1
                        nz[z] -= 1
                    z_prob = np.divide(np.multiply(nuz[i], niz[j]), np.multiply(nz, nu[i]))  # update probability
                    # Re-sampling
                    for w in range(z_set_size):
                        z = np.random.multinomial(1, z_prob / z_prob.sum()).argmax()
                        nuz[i][z] += 1
                        niz[j][z] += 1
                        nz[z] += 1
                        z_sets[i][j][w] = z

        self.nuz_matrix = nuz
        self.niz_matrix = niz
        self.nz_matrix = nz
        self.z_sets = z_sets

    def train(self):
        iter_num = self.iter_num
        for _ in range(iter_num):
            self.gibbs_sampling()
        return self

    def format_user_topic_matrix(self):
        user_topic_matrix = np.zeros((self.M, self.K))
        z_sets = self.z_sets
        for i in range(self.M):
            for j in range(self.N):
                z_set = z_sets[i][j]
                z_set_size = len(z_set)
                if z_set_size > 0:
                    for w in range(z_set_size):
                        z = z_set[w]
                        user_topic_matrix[i][z] += 1
        return user_topic_matrix
