import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


votes = pd.read_csv(r"C:\Users\Kinara\Documents\congressus.csv", engine='python')
kmeans_model = KMeans(n_clusters=2,random_state=1) 
senator_distances=kmeans_model.fit_transform(votes.iloc[:,3:]) 
labels=kmeans_model.labels_

nomer = []

for i in range(len(votes)):
    nomer.append(i+1)
    
df = pd.DataFrame({'No':nomer,
                   'party':votes['party'],
                   'Clustered':labels,
                   'distance to cluster0':senator_distances[:,0],  
                   'distance to cluster1':senator_distances[:,1]})

df.to_csv (r'C:\Users\Kinara\Documents\porter-stemmer-master\ClusteringResult.csv', index = False, header=True)


print(votes['party'].value_counts())
print(pd.crosstab(labels,votes['party']))

print(votes[(labels==0) & (votes['party']=='D')])

print(votes[(labels==1) & (votes['party']=='I')])

