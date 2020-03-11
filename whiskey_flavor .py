import numpy as np 
import pandas as pd 
  
whisky = pd.read_csv("whiskies.txt") 
whisky["Region"] = pd.read_csv("regions.txt") 
 
flavors = whisky.iloc[:, 2:14] 
  
corr_flavors = pd.DataFrame.corr(flavors) 
print(corr_flavors) 
import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 10)) 
plt.pcolor(corr_flavors) 
plt.colorbar()  

corr_whisky = pd.DataFrame.corr(flavors.transpose()) 
plt.figure(figsize=(10, 10)) 
plt.pcolor(corr_whisky) 
plt.axis("tight") 
plt.colorbar() 

plt.show() 

from sklearn.cluster.bicluster import SpectralCoclustering 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

model = SpectralCoclustering(n_clusters=6, random_state=0) 
model.fit(corr_whisky) 
model.rows_ 	 
model.row_labels_ 

whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index) 
whisky = whisky.ix[np.argsort(model.row_labels_)] 
whisky = whisky.reset_index(drop=True) 

correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose()) 
correlations = np.array(correlations) 

plt.figure(figsize = (14, 7)) 
plt.subplot(121) 
plt.pcolor(corr_whisky) 
plt.title("Original") 
plt.axis("tight") 
plt.subplot(122) 
plt.pcolor(correlations) 
plt.title("Rearranged") 
plt.axis("tight") 
plt.show() 
plt.savefig("correlations.pdf") 
