#!/usr/bin/env python
"""
Run k-NN classification on the Reuters text dataset using LSA.

This script leverages modules in scikit-learn for performing tf-idf and SVD.

Classification is performed using k-NN with k=5 (majority wins).

The script measures the accuracy of plain tf-idf as a baseline, then LSA to
show the improvement.
"""

import pickle
import time
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import BernoulliNB, MultinomialNB


###############################################################################
#  Load the raw text dataset.
###############################################################################

print("Loading dataset...")

data = pd.read_csv("fake_news.csv")
print(data.head())

print(data.columns.values)

X = data['text']
y = data['label']

# print(pd.value_counts(y))
# def give_rating(x):
#     if x >1:
#         return "1"
#     elif x <= 1:
#         return "0"
#
# y = y.apply(give_rating)

                            #
                            # index_Fake_News = [idx for idx,val in enumerate(y) if val == 'FAKE']
                            # index_True_News = [idx for idx,val in enumerate(y) if val == 'REAL']
                            # from random import shuffle
                            # shuffle(index_Fake_News)
                            # sh_index_Fake_News  = index_Fake_News[:]
                            #
                            # shuffle(index_True_News)
                            # sh_index_True_News  = index_True_News[:]
                            #
                            #
                            # # Apply those Shuffles index to "X" data and "y" data
                            #
                            # under_train_after_shuffle = sh_index_Fake_News+sh_index_True_News
                            # shuffle(under_train_after_shuffle)
                            #
                            # under_train_y = [y[i] for i in under_train_after_shuffle]
                            #
                            # under_train_X = [X[i] for i in under_train_after_shuffle]
                            # y = pd.Series(under_train_y)



X_train_raw = X[:5500]
y_train_labels = y[:5500]

X_test_raw = X[-500:].reset_index()['text']
y_test_labels = y[-500:].reset_index()['label']




print("print X_train_raw[0]")
print(X_train_raw[0])
print("print y_train_raw[0]")
print(y_train_labels[0])
print("print X_test_raw[0]")
print(X_test_raw[0])
print("print y_test_raw[0]")
print(y_test_labels[0])


y_train_labels.tolist()
y_train = [str(i) for i in y_train_labels]

y_test_labels.tolist()
y_test = [str(i) for i in y_test_labels]


###############################################################################
#  Use LSA to vectorize the articles.
###############################################################################

# Tfidf vectorizer:
#   - Strips out “stop words”
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
#     document length on the tf-idf values.

from nltk.corpus import stopwords
stop_words_list = stopwords.words("english")
stop_words_list += ['.','/','<','>','``','"','-','--']

vectorizer = TfidfVectorizer(min_df = 10,max_df=0.5, ngram_range=(1,3)
                             , max_features=10000,
                             stop_words=stop_words_list,
                             use_idf=True)


# from sklearn.feature_extraction.text import TfidfVectorizer
# import pandas as pd
# texts = [
#     "good movie", "not a good movie", "did not like",
#     "i like it", "good one"
# ]
# # using default tokenizer in TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
# features = vectorizer.fit_transform(X_train_raw)
# features_df = pd.DataFrame(
#     features.todense(),
#     columns=vectorizer.get_feature_names()
# )

#print(features_df.head())




# Build the tfidf vectorizer from the training data ("fit"), and apply it
# ("transform").
X_train_tfidf = vectorizer.fit_transform(X_train_raw)

print(type(X_train_tfidf))
print(X_train_tfidf)

print("show the shape of TFIDF")
print(X_train_tfidf.shape)

print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

print("\nPerforming dimensionality reduction using LSA")
t0 = time.time()

# Project the tfidf vectors onto the first N principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
svd = TruncatedSVD(1500)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
X_train_lsa = lsa.fit_transform(X_train_tfidf)
print("show one element")
print(X_train_lsa[0][0])
print("show X_train_lsa matrix")
print(X_train_lsa)


print("show the shape of X_train_lsa")
print(X_train_lsa.shape)

print("  done in %.3fsec" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(X_test_raw)
X_test_lsa = lsa.transform(X_test_tfidf)


###############################################################################
#  Run classification of the test articles
###############################################################################

print("\nClassifying tfidf vectors...")

# Time this step.
t0 = time.time()

print(X_train_tfidf.shape)
print(len(y_train))

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance,
# and brute-force calculation of distances.
model = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(X_train_tfidf, y_train)

print("see the x_train TFIDF matrix")
print(X_train_tfidf)
print("see the dimension of the x_train TFIDF matrix")
print(X_train_tfidf.shape)

# Classify the test vectors.
p = model.predict(X_test_tfidf)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("  done in %.3fsec" % elapsed)


print("\nClassifying LSA vectors...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance,
# and brute-force calculation of distances.
model_lsa = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
model_lsa.fit(X_train_lsa, y_train)


# print x_train LSA
print("X_train_lsa")
print(X_train_lsa)
# print y_train LSA
print("y_train LSA")
print(y_train)


# Classify the test vectors.
p = model_lsa.predict(X_test_lsa)

#print(p)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    #if p[i] != y_test[i]:
        #print(p[i])
    if p[i] == y_test[i]:
        #print(p[i])
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("    done in %.3fsec" % elapsed)

