import pandas as pd
import os

dir = '/Users/xinwang/ai/dataset/ml-20m'
gScoreFile = 'genome-scores.csv'

gScoreNames = ['movieId', 'tagId', 'relevance']
gScores = pd.read_csv(os.path.join(dir, gScoreFile),
                      sep=',', header=None,  names=gScoreNames)
# print(gScores[:10])

movieFile = 'movies.csv'
movieNames = ['movieId', 'title', 'genres']
movies = pd.read_table(os.path.join(dir, movieFile),
                       sep=',', header=None, names=movieNames)
# print(movies[:10])

ratingFile = 'ratings.csv'
ratingNames = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv(os.path.join(dir, ratingFile),
                      sep=',', header=None, names=ratingNames)

# print(ratings[:10])

data = pd.merge(gScores, pd.merge(movies, ratings))
# print(data[:10])

mean_ratings = data.pivot_table('rating', rows='title', cols='')
