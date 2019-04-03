"""
Created on Sun Jul 08 05:03:01 2018
@Project Title: Learning and Summarizing Graphical Models using Eigen Analysis of Graph Laplacian: An Application in Analysis of Multiple Chronic Conditions 
@Project: EAGL (Simplification Based Graph Summarization)
@author: Syed Hasib AKhter Faruqui
"""
print('##############################################################################')
print('############################# Sample Example #################################')
print('##############################################################################')


## Load Necessary Library
import EAGL as gc  # Graph Compression Library
from scipy import io as sc
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

### Setting Random Seeds for reproducibility
from numpy.random import seed
seed(123)
import os
os.environ['PYTHONHASHSEED']='0'
import random as rn
rn.seed(123)

## Load Necessar Data (matlab Extraction)
# Load Random Data
# A. L. Barabási and R. Albert “Emergence of scaling in random networks”, Science 286, pp 509-512, 1999.
G=nx.barabasi_albert_graph(10,5, seed=123)
DAG=nx.to_numpy_array(G)

# Main Calculation
# 	Method						:'False' = Single edge reduction  (Default)
#								 'True'  = Multiple edge reduction
Updated_DAG,EigenValue,NumberofEdges=gc.GraphCompression(DAG,Method='False')

# Print relevent Information
print('Number of Edges on the Updated DAG:',np.count_nonzero(Updated_DAG))
Percentage = np.abs(np.count_nonzero(Updated_DAG)-np.count_nonzero(DAG))/np.count_nonzero(DAG)
print('Compression:',Percentage*100,'%')



## Plot the Tree's
pos = nx.random_layout(nx.DiGraph(DAG))
plt.figure(1)
plt.subplot(1, 2, 1)
gc.plot_Graph(DAG,pos)
plt.title('Original DAG')
plt.subplot(1, 2, 2)
gc.plot_Graph(Updated_DAG,pos)
plt.title('Summarized DAG')
plt.tight_layout() # Fitting the plot

## Plot Number of Edges Reduced
Compression=np.count_nonzero(Updated_DAG)/np.count_nonzero(DAG)
plt.figure(2)
gc.plot_Edge_Reduction(NumberofEdges,"DAG_Unsupervised_2nd_Eigen_Comp:"+str((1-Compression)*100)+'%',mark='x',Color=np.random.random(3))

## Remove Temporary Files Created
os.remove("Dummy_DAG.mat")
    
print('##############################################################################')
print('############################ Example Complete! ###############################')
print('##############################################################################')
