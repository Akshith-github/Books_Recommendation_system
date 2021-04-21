import pandas as pd, numpy as np#, matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix
import os

# load dataset
base_loc = os.path.join(os.path.dirname(__file__), os.path.join('data'))
books = pd.read_csv(base_loc+'\\BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

users = pd.read_csv(base_loc+'\\BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

ratings = pd.read_csv(base_loc+'\BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
# filter the dataset as(To ensure statistical significance, users with less than 200 ratings, 
#          and books with less than 100 ratings are excluded.)
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

# merge dataframes
combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
# combine_book_rating.head()

#Handling NA values and feature extraction
combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

book_ratingCount = (combine_book_rating.groupby(by = ['bookTitle'])['bookRating'].count().reset_index().rename(columns = {'bookRating': 'totalRatingCount'})[['bookTitle', 'totalRatingCount']])
# book_ratingCount.head()

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
# rating_with_totalRatingCount.head()

# filter out less popular books
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
# rating_popular_book.head() 

# filter to usa and canda to mantain a consistent data set
combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')
us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating=us_canada_user_rating.drop('Age', axis=1)
# us_canada_user_rating.head()

# sparse matrix to be generated to feed on the knn model
us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
# us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

# save the  us_canada_user_rating_pivot as csv file
us_canada_user_rating_pivot.to_csv(base_loc+"\\us_canada_user_rating_pivot.csv")