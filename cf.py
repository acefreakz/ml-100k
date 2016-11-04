# Collaborative Filtering Recommendation
# INCOMPLETE!

import numpy as np
import pandas as pd

names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('data/u.data', sep='\t', names=names)
print df.head(n=10)

# for index, row in df.iterrows():
#     if row['user_id'] == 196:
#         print row['item_id'], row['rating']

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print str(n_users) + ' users'
print str(n_items) + ' items'

ratings = np.zeros((n_users, n_items))
end = 1
for row in df.itertuples():
    if end > 0:
        print row
        print "%s, %s" % (row[1] - 1, row[2] - 1)
        print ratings[row[1] - 1, row[2] - 1]
        print row[3]
    end -= 1
    ratings[row[1] - 1, row[2] - 1] = row[3]
print ratings
print "ratings shape: %i, %i" % (ratings.shape[0], ratings.shape[1])
print ratings[196 - 1][7]

sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print 'Sparsity: {:4.2f}%'.format(sparsity)
