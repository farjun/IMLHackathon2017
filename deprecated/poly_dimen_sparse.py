import numpy as np

import scipy


matrix = data1.word_matrix()
print(type(matrix))
print(matrix.shape)
print(matrix[1:3, 1:5])


headlines = matrix.shape[0]
words = matrix.shape[1]
new_matrix = scipy.sparse.csr_matrix((headlines, (scipy.misc.comb(words, 2))))
# for i in range(headlines):
#     col = 0
#     for j in range(words):
#         for k in range(j, words):
#             new_matrix