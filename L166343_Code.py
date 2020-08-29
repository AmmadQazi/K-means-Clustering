from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Included only for elbow plot
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
#Distortions = [] 

def a():
    np.seterr('raise') 
    # Importing the dataset
    data = pd.read_csv('data_updated.csv')
    print("Input Data and Shape")
    print(data.shape)
    
    f1 = data['X1'].values
    f2 = data['X2'].values
    f3 = data['X3'].values
    f4 = data['X4'].values
    f5 = data['X5'].values
    X = np.array(list(zip(f1, f2, f3, f4, f5)))

    # Euclidean Distance Caculator
    def dist(a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

    k = input("Enter your value: ") 
    print("Number of clusters is ",k)
    k = int(k)

    C_x = np.random.randint(6, size=k)
    C_a = np.random.randint(6, size=k)
    C_b = np.random.randint(6, size=k)
    C_c = np.random.randint(6, size=k)
    C_y = np.random.randint(6, size=k)
    C = np.array(list(zip(C_x, C_y, C_a, C_b, C_c)), dtype=np.float32)
    print("Initial Centroids")
    print(C)

   
    C_old = np.zeros(C.shape)
    clusters = np.zeros(len(X))
    error = dist(C, C_old, None)
   
    while error != 0:
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        
        C_old = deepcopy(C)
        
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.nanmean(points, dtype=np.float64)
        error = dist(C, C_old, None)
    
 #   for i in range(len(X)):
  #     Distortions.append(np.min(dist(X[i],C)))

    #Uncomment this for part 4 to Display point, it's cluster number and distortion value
    #Printing data points and respective cluster number
    #for x in range(len(clusters)): 
     #   print("data point {",f1[x]," , ",f2[x]," , ",f3[x]," , ",f4[x]," , ",f5[x],"}")
      #  print("cluster number {",clusters[x],"}")
       # print("Distortion value : ",Distortions[x])  
        
    print("Final Centroid values")
    print(C)


#CODE FOR ELBOW PLOT STARTS HERE 
    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    K = range(1,10) 

    for k in K: 
        kmeanModel = KMeans(n_clusters=k).fit(X) 
        kmeanModel.fit(X)     
      
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
        inertias.append(kmeanModel.inertia_) 
  
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 


    for key,val in mapping1.items(): 
        print(str(key)+' : '+str(val))
    
    plt.plot(K, distortions, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Distortion') 
    plt.title('The Elbow Method using Distortion') 
    plt.show() 

   # K= range(1,len(distortions)+1)
    #plt.plot(K, distortions, 'bx-')
    #plt.xlabel('k')
    #plt.ylabel('Distortion')
    #plt.title('The Elbow Method showing the optimal k')
    #plt.show()
    return



def main():
    a()

if __name__ == "__main__":
    main()    
