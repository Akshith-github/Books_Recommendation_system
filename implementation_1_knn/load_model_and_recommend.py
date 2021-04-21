#import
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd,os,pickle
try:
    from sklearn.externals import joblib
except:
    try:
        import joblib
    except:
        print("action to import joblib failed ")
from random import randint
from knn_model_build import test_model_output 

# print(__file__)
#dataset 
base_loc = os.path.join(os.path.dirname(__file__), os.path.join('data'))
us_canada_user_rating_pivot = pd.read_csv(base_loc+"\\us_canada_user_rating_pivot.csv")
us_canada_user_rating_pivot_idx = us_canada_user_rating_pivot.set_index("bookTitle")
# print(us_canada_user_rating_pivot.head())
model_loc = os.path.join(os.path.dirname(__file__), os.path.join('models')) 
pd.DataFrame(us_canada_user_rating_pivot["bookTitle"]).to_csv(base_loc+"\\popular_usa_books.csv")
titles=us_canada_user_rating_pivot["bookTitle"].str.lower()
#load model
def load_model():
    loaded_model = None
    try:
        with open(model_loc+'\\knnpickle_file', 'rb') as knnPickle:
            loaded_model = pickle.load(knnPickle)
        print("loaded model (1)")
    except:
        try:
            loaded_model = joblib.load( model_loc+'\\model_knn.pkl' , mmap_mode ='r')
            print("loaded model (2)")
        except:
            print("\nUnable to load model :(\n")
    finally:
        return loaded_model

# print(load_model())
def run_random_recommend():
    print(*test_model_output(load_model(),return_list=False,n_neighbors=3),sep="\n")

def recommend_for_book(book_name:str,return_list=True,n_neighbors=6):
    title_idx = titles[titles==book_name.lower()]
    if(not title_idx.shape[0]):
        return []
    # print(title_idx.shape)
    query_index=int(title_idx.index.values)
    # print(query_index)
    model_knn=load_model()
    # print(us_canada_user_rating_pivot_idx.iloc[query_index,1:].values.reshape(1, -1))
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot_idx.iloc[query_index,:].values.reshape(1, -1), n_neighbors = n_neighbors)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot_idx.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot_idx.index[indices.flatten()[i]], distances.flatten()[i]))
    if(return_list):
        return [
            (i,us_canada_user_rating_pivot_idx.index[indices.flatten()[i]],distances.flatten()[i])
            for i in range(0, len(distances.flatten()))]
    return []

if __name__ == "__main__":
    run_random_recommend()
    recommend_for_book(input("Enter book name :"))