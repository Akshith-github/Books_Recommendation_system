from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd,os,pickle
try:
    from sklearn.externals import joblib
except:
    import joblib
else:
    print("action to import joblib failed ")
from random import randint

#dataset 
base_loc = os.path.join(os.path.dirname(__file__), os.path.join('data'))
us_canada_user_rating_pivot = pd.read_csv(base_loc+"\\us_canada_user_rating_pivot.csv")
us_canada_user_rating_pivot.set_index("bookTitle",inplace = True)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)
#demo test_case
def test_model_output(model_knn:NearestNeighbors,return_list:bool=False,n_neighbors:int=6):# if __name__ == '__main__':
    query_index = randint(0,us_canada_user_rating_pivot.shape[0]-1)#np.random.choice(us_canada_user_rating_pivot.shape[0])
    # print(query_index,us_canada_user_rating_pivot.shape)
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = n_neighbors)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
    if(return_list):
        return [
            (i,us_canada_user_rating_pivot.index[indices.flatten()[i]],distances.flatten()[i])
            for i in range(0, len(distances.flatten()))]
    return []

def build_knn_model():
    #build model
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(us_canada_user_rating_matrix)
    test_model_output(model_knn)
    return model_knn

if __name__ == '__main__':
    model_knn = build_knn_model()
    # saving model 

    # Its important to use binary mode
    # 
    model_loc = os.path.join(os.path.dirname(__file__), os.path.join('models')) 
    knnPickle = open(model_loc+'\\knnpickle_file', 'wb') 
    # source, destination 
    pickle.dump(model_knn, knnPickle)                      
    knnPickle.close()
    # load the model from disk
    loaded_model = pickle.load(open(model_loc+'\\knnpickle_file', 'rb'))
    #test the models output
    test_model_output(loaded_model)

    ## save using joblib library
    joblib.dump(model_knn , model_loc+'\\model_knn.pkl')
    modelknn_loaded = joblib.load( model_loc+'\\model_knn.pkl' , mmap_mode ='r')
    test_model_output(modelknn_loaded)