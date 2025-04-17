# Cell 1: Import Libraries
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Cell 2: Load the Dataset
# Load the Book-Crossings dataset
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

# Cell 3: Data Preprocessing and Model Building
# Filter users with at least 200 ratings
user_counts = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(user_counts[user_counts >= 200].index)]

# Filter books with at least 100 ratings
book_counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 100].index)]

# Merge ratings with book titles
ratings_with_titles = ratings.merge(books[['ISBN', 'bookTitle']], on='ISBN')

# Create a pivot table: rows = books, columns = users, values = ratings
pivot_table = ratings_with_titles.pivot_table(index='bookTitle', columns='userID', values='bookRating').fillna(0)

# Convert pivot table to a sparse matrix
sparse_matrix = csr_matrix(pivot_table.values)

# Train the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(sparse_matrix)

# Function to get book recommendations
def get_recommends(book_title=""):
    try:
        # Find the index of the input book
        book_idx = pivot_table.index.get_loc(book_title)
        
        # Get the 6 nearest neighbors (including the book itself)
        distances, indices = model_knn.kneighbors(sparse_matrix[book_idx], n_neighbors=6)
        
        # Prepare the list of recommended books and their distances
        recommended_books = []
        for idx, dist in zip(indices[0][1:], distances[0][1:]):  # Skip the first (input book itself)
            recommended_books.append([pivot_table.index[idx], float(dist)])
        
        # Return the result in the required format
        return [book_title, recommended_books[:5]]  # Return top 5 recommendations
    except:
        return [book_title, []]  # Return empty list if book not found

# Cell 4: Test the Function
# Test the get_recommends function
result = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(result)

# Verify the output format
expected_books = [
    'Catch 22',
    'The Witching Hour (Lives of the Mayfair Witches)',
    'Interview with the Vampire',
    'The Tale of the Body Thief (Vampire Chronicles (Paperback))',
    'The Vampire Lestat (Vampire Chronicles, Book II)'
]
recommended_books = [book[0] for book in result[1]]
if result[0] == "The Queen of the Damned (Vampire Chronicles (Paperback))" and recommended_books == expected_books:
    print("Test passed: Correct book title and recommendations!")
else:
    print("Test failed: Check the get_recommends function.")