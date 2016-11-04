# Content Based Filtering Recommendation

import numpy as np
from pandas import read_csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize


# Compute sparsity
def density(matrix=None):
    ds = float(len(matrix.nonzero()[0]))
    ds /= (matrix.shape[0] * matrix.shape[1])
    return ds * 100


# Print Top N similar movies
def top_n_movies(raw_data_set, similarity_matrix, movie_id=1, n=5, is_desc=True):
    # FIXME: noob code
    top_n_similar_movies_index = np.argsort(similarity_matrix[:, movie_id - 1], axis=0)
    if is_desc:
        top_n_similar_movies_index = top_n_similar_movies_index[::-1]
    top_n_similar_movies_index = top_n_similar_movies_index[:n]
    # top_n_similar_movies_index = np.argsort(similarity_matrix[:, movie_id - 1], axis=0)[::-1][:n]
    print "DEBUG: (topNmovies) indexes = %s" % np.argsort(similarity_matrix[:, movie_id - 1], axis=0)[::-1][:n]
    for i in xrange(n):
        m_index = top_n_similar_movies_index[i]
        m_id = m_index + 1
        print "DEBUG: (topNmovies) #%s Movie ID: %s, Name: %s, Similarity/Distance: %s" % (i + 1, m_id,
                                                                                  raw_data_set.get_value(index=m_index, col=1, takeable=True),
                                                                                  similarity_matrix[m_index, movie_id - 1])
        print_movie_attr(raw_data_set=raw_data_set, movie_id=m_id)


# FIXME: sohai code
def print_movie_attr(raw_data_set, movie_id):
    cols = raw_dataset.columns.values
    l = []
    offset = 6
    for i in xrange(len(cols) - offset):
        if raw_dataset.get_value(movie_id - 1, i + offset, takeable=True) == 1:
            l.append(cols[i + offset])
    print '             Movie attr: ' + ', '.join(l)


names = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL',
         'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
         'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
         'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
raw_dataset = read_csv('data/u.item', sep='|', names=names)
print raw_dataset.head()
print raw_dataset.columns.values

# Remove first 5 columns (movie_id, movie_title, release_date, video_release_date, IMDb_URL)
narray = raw_dataset.values
dataset = narray[:, 5:]

# Compute similarity matrix (cosine_similarity)
similarity_dataset = cosine_similarity(dataset)
print 'DEBUG (cosine_similarity) Density: {:4.2f}%'.format(density(similarity_dataset))
print 'DEBUG (cosine_similarity) similarity_dataset shape: %i x %i' % (similarity_dataset.shape[0], similarity_dataset.shape[1])
print similarity_dataset
top_n_movies(raw_data_set=raw_dataset, similarity_matrix=similarity_dataset, movie_id=1, n=10, is_desc=True)

# Compute similarity matrix (euclidean_distances)
normed_dataset = normalize(dataset, copy=True, return_norm=False)
similarity_dataset = euclidean_distances(normed_dataset)
print 'DEBUG (euclidean_distances w/ normalization) Density: {:4.2f}%'.format(density(similarity_dataset))
print 'DEBUG (euclidean_distances w/ normalization) similarity_dataset shape: %i x %i' % (similarity_dataset.shape[0], similarity_dataset.shape[1])
print similarity_dataset
top_n_movies(raw_data_set=raw_dataset, similarity_matrix=similarity_dataset, movie_id=1, n=10, is_desc=False)

# Compute similarity matrix (euclidean_distances w/o normalized first)
# Note: Notice the results will not be approriate, due to imbalanced genre vector length (per movie)
similarity_dataset = euclidean_distances(dataset)
print 'DEBUG (euclidean_distances w/o normalization) Density: {:4.2f}%'.format(density(similarity_dataset))
print 'DEBUG (euclidean_distances w/o normalization) similarity_dataset shape: %i x %i' % (similarity_dataset.shape[0], similarity_dataset.shape[1])
print similarity_dataset
top_n_movies(raw_data_set=raw_dataset, similarity_matrix=similarity_dataset, movie_id=1, n=10, is_desc=False)
