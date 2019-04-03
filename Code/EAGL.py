"""
Created on Sun Jul 08 05:03:01 2018
@Project Title: Learning and Summarizing Graphical Models using Eigen Analysis of Graph Laplacian: An Application in Analysis of Multiple Chronic Conditions 
@Project: EAGL (Simplification Based Graph Summarization)
@author: Syed Hasib AKhter Faruqui
"""
# Import Libraries
import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy import io as sc
import matplotlib.pyplot as plt
import warnings
import os
# Defining the functions

## Normalizing a MATRIX
# This will be used if we want to normalize the attained graph laplacian
# Input		: Laplacian Matrix
# Output 	: Normalized Laplacian Matrix

def NormalizeMatrix(Matrix):
    row_sums = Matrix.sum(axis=1)
    return Matrix / row_sums

## Function to Extract the Tree from the DAG
# Extract the Depth-first tree or Breadth-first tree. 
# Input:
# 	DAG				: Input DAG
# 	Tree_Option 	: 'dfs' or 'bfs'
# 	StartingNode 	: Starting point of tree traversing
# Output:
# 	tree_matrix 	: Extracted Tree from DAG
# !! A future improvent can be, instead of StartingNode input, we can levarage input/output degree
	
def TreeExtraction(DAG,Tree_Option,StartingNode):
    DAG = sp.csr_matrix(DAG)                                      # Compressed Sparse Row matrix
    G=nx.DiGraph(DAG)                                             # Create The Directed Graph
    
    ## Switching between DFS and BFS
    if Tree_Option=='dfs':
          tree = nx.dfs_tree(G, StartingNode)                    # Extract the DFS Tree
    else:
          tree = nx.bfs_tree(G, StartingNode)                    # Extract the BFS Tree

    tree_matrix=nx.to_numpy_matrix(tree)                          # Graph adjacency matrix as a NumPy matrix.
    # sc.savemat('TemporaryStore.mat', {'Tree_DAG':tree_matrix})  # Delete this temporary Matrix at the end of analysis
    return tree_matrix,tree

## Function for Eigenvalue Entry
# Calculates the Top_k eigen value. 
# Input:
# 	DAG 						: Input DAG
# 	DAG_Size 					: Size of the DAG (Future Improvement: Extract directly from DAG)
# 	Top_k_Eigenvalue_Number 	: Which Eigen value to extract (1st or 2nd)
# 	norm 						: Either to normalize the Laplacian or not.
# Output:
# 	EigenValue 					: Eigen value of the input DAG
# 	Top_k_Eigenvector 			: Eigen vector for the eigen value
# 	Top_k_Eigenvalue_Index 		: Index of the Top_k eigen vector
# 	Laplacian 					: The laplacian matrix for the provided 

def eigenDAG(DAG,DAG_Size,Top_k_Eigenvalue_Number,norm = False):
    # Matrix to Sparse
    DAG = sp.csr_matrix(DAG)
    # Create Graph
    G=nx.DiGraph(DAG)  # Create The Directed Graph
    # Calculate Directed laclacian
    Laplacian=nx.directed_laplacian_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95)
    # Normalize the matrix
    if norm:
        Laplacian=NormalizeMatrix(Laplacian)
    # Eigen value of Laplacian
    eigenvalues,eigenvectors = np.linalg.eig(Laplacian)
    # Sorting the eigenvalues
    np.matrix.sort(eigenvalues)
    # Top K EigenValues
    Top_k_Eigenvalue=eigenvalues[(DAG_Size-Top_k_Eigenvalue_Number):DAG_Size]
    
    ## If the test is for 2nd Eigen Value then this line will choose the 2nd one otherwise 1st one
    Top_k_Eigenvalue=Top_k_Eigenvalue[0]         
    
    # Getting the index for Max value
    Top_k_Eigenvalue_Index = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i])[-2:]
    
    # List of Top Eigen Vactors
    Top_k_Eigenvector=np.zeros
    Top_k_Eigenvector=eigenvectors[:,Top_k_Eigenvalue_Index[0]]
    for i in range(Top_k_Eigenvalue_Number-1):
          Top_k_Eigenvector=np.column_stack((Top_k_Eigenvector,eigenvectors[:,Top_k_Eigenvalue_Index[i+1]]))
    

    return Top_k_Eigenvalue,Top_k_Eigenvector,Top_k_Eigenvalue_Index,Laplacian

## Store Eigen Values for all the Test Case
# Calculate Eigen Values for Edge Deletion
# Input:
# 	DAG 						: Input DAG
# 	DAG_Size 					: Size of the DAG (Future Improvement: Extract directly from DAG)
# 	Top_k_Eigenvalue_Number 	: Which Eigen value to extract (1st or 2nd)
# Output:
# 	EigenChange 				: list of eigen changes due to edge deletion
# 	DAG_Track 					: Tracking Matrix which will keep track of eigen changes due to edge deletion
# 	OriginalEigen 				: Original Eigen value of the provided DAG

def eigenStore(DAG,DAG_Size,Top_k_Eigenvalue_Number):
    OriginalEigen,Original_Top_k_Eigenvector,Original_Top_k_Eigenvalue_Index,Original_Laplacian=eigenDAG(DAG,DAG_Size,Top_k_Eigenvalue_Number)
    EigenStoreSize=np.count_nonzero(DAG)#np.sum(DAG).astype(int)
    # Define Initials
    # Tracking the DAG Edges
    DAG_Track=np.zeros((DAG_Size,DAG_Size))
    
    # Tracking the Eigen Changes
    if Top_k_Eigenvalue_Number==1:
          EigenChange=np.zeros((Top_k_Eigenvalue_Number  , EigenStoreSize))   # 1st Eigen
    elif Top_k_Eigenvalue_Number==2:
          EigenChange=np.zeros((Top_k_Eigenvalue_Number-1, EigenStoreSize))   # 2nd Eigen
          
    # Save the DAG as a Dummy mat file
    sc.savemat('Dummy_DAG.mat', {'DAG':DAG})
    
    count=0;
    for i in range(DAG_Size):
        for j in range(DAG_Size):
            # Load DAG
            DAG=sc.loadmat('Dummy_DAG.mat')
            DAG=DAG['DAG']
            
            if DAG[i,j]>0:
                   DAG[i,j]=0
                   Top_k_Eigenvalue,Top_k_Eigenvector,Top_k_Eigenvalue_Index,Laplacian=eigenDAG(DAG,DAG_Size,Top_k_Eigenvalue_Number)
                   EigenChange[:,count]=np.absolute(OriginalEigen-Top_k_Eigenvalue)/OriginalEigen*100 #*10000
                   DAG_Track[i,j]=EigenChange[:,count]
                   count=count+1
#                   print (count)
    return EigenChange,DAG_Track,OriginalEigen

## Updated DAG from DAG Track
# Based on the tracked changes, this will create the new DAG for next iteration. This is based on Algorithm 1 of the paper.
# Input:
# 	DAG 					: Input DAG
# 	DAG_Size 				: DAG Size
# 	DAG_Track 				: Tracking Matrix which will keep track of eigen changes due to edge deletion
# 	EigenChange 			: Change in eigen value
# 	OriginalEigen 			: Eigen Value of Original DAG
# 	CompressionPercent 		: Desired compression Percentage
# Output:
# 	DAG_Updated 			: Updated DAG with removed Edge

def NewDAG_EigenBased(DAG_Size,DAG_Track,EigenChange,OriginalEigen,CompressionPercent,DAG):
    DAG_Updated=np.zeros((DAG_Size,DAG_Size))
    for i in range(DAG_Size):
        for j in range(DAG_Size):
            if DAG_Track[i,j]>(OriginalEigen*CompressionPercent):
                DAG_Updated[i,j]=DAG[i,j]
    return DAG_Updated

## Updated DAG from DAG Track
# Based on the tracked changes, this will create the new DAG for next iteration. This is based on Algorithm 2 of the paper.
# Input:
# 	DAG 					: Input DAG
# 	DAG_Size 				: DAG Size
# 	DAG_Track 				: Tracking Matrix which will keep track of eigen changes due to edge deletion
# 	EigenChange 			: Change in eigen value
# 	OriginalEigen 			: Eigen Value of Original DAG
# Output:
# 	DAG_Updated 			: Updated DAG with removed Edge

def NewDAG_IterationBased(DAG_Size,DAG_Track,EigenChange,OriginalEigen,DAG):
    DAG_Updated=np.zeros((DAG_Size,DAG_Size))
    for i in range(DAG_Size):
        for j in range(DAG_Size):
            if DAG_Track[i,j]>np.min(EigenChange):
                DAG_Updated[i,j]=DAG[i,j]
    return DAG_Updated

## Any Edge on the Tree won't be deleted
# This makes sure, No edge on the tree gets deleted (Future Improvement: Include this condition on Edge deletion test, function: eigenStore
# Input:
# 	tree_matrix 		: Extracted Tree from the original DAG
# 	Updated_DAG 		: The updated DAG with/without the tree DAG
# 	DAG_Size			: Size of the DAG
# 	DAG 				: DAG
# Output:
# 	Updated_Tree_DAG	: Updated DAG with tree DAG

def TreeConnecting(tree_matrix,Updated_DAG,DAG_Size,DAG):
    Updated_Tree_DAG = Updated_DAG
    for i in range(DAG_Size):
        for j in range (DAG_Size):
            if (tree_matrix[i,j]>1):
                if (Updated_DAG[i,j]==0):
                    Updated_Tree_DAG[i,j]=DAG[i,j]
    return Updated_Tree_DAG

#Plotting the Graph from Adjacency matrix
def plot_Graph(DAG,pos):
    G = nx.DiGraph(DAG)  # Create default Graph
    nx.draw(G,pos = pos)
    nx.draw_networkx_labels(G,pos=pos)
    return

# Plotting the reduction of Edges at each iteration
def plot_Edge_Reduction(NumberofEdges,LabelName,mark,Color):
    ## Plotting the Number of Edges Left
    plt.plot(NumberofEdges.T,'gx-',label=LabelName,marker=mark,color=Color)
    plt.grid(True)
    plt.legend(loc=1)
    plt.title('Graph  Compression for Different DAG\'s')
    plt.ylabel('Number of Edges')
    plt.xlabel('Iteration')

########################################################################################################################
##                                  			Combining Everything												  ##
########################################################################################################################
# Algorithm 2:  
# Input:
# 	DAG 						: Adjacency Matrix of the Graphs
# 	Method						:'False' = Single edge reduction  (Default)
#								 'True'  = Multiple edge reduction
def GraphCompression(DAG,Method='False'):
    # DAG Size
    DAG_Size=DAG.shape[1] 															# Retrive the DAG Size
    #User Inputs
    #warnings.warn('Extract The DFS/BFS Tree (if more than 1 source Node, Use the Dummy Node added tree)!')
    
    
    Tree_Connect  = input("Enter if tree connection to be maintained (True or False): ")				# 'dfs' or 'bfs'
    
    if Tree_Connect=='True':
        Tree_Option  = input("Enter Tree Extraction method (dfs or bfs): ")				# 'dfs' or 'bfs'
        StartingNode = int(input("Enter Traversing Start Node(0 to "+str(DAG_Size-1)+"):"))# Starting Node
        # Extract Tree Matrix
        tree_matrix,tree=TreeExtraction(DAG,Tree_Option,StartingNode)
    
    IterationNumber = int(input("Enter Number of Iterations: ")) 					# User input: Number of Iteration
    Top_k_Eigenvalue_Number = int(input("Enter for which Eigenvalue (1st or 2nd) perform the calculation: ")) 					# Eigenvalue (1st or 2nd)
    

    NumberofEdges=np.zeros((1,IterationNumber)) 									# Edge Reduction Count
    
    if Method == 'True':
        CompressionPercent = float(input("Enter Cut-off Value: ")) # User Input: Compression Percent
    for i in range(IterationNumber):
        NumberofEdges[:,i]=np.count_nonzero(DAG)
        EigenValue,DAG_Track,OriginalEigen=eigenStore(DAG,DAG_Size,Top_k_Eigenvalue_Number)
        # GS Method
        if Method == 'True':
            DAG=NewDAG_EigenBased(DAG_Size,DAG_Track,EigenValue,OriginalEigen,CompressionPercent,DAG)
        else:
            DAG=NewDAG_IterationBased(DAG_Size,DAG_Track,EigenValue,OriginalEigen,DAG)
            
        # Do we consider the tree case or not?
        if Tree_Connect=='True':
            DAG2=TreeConnecting(tree_matrix,DAG,DAG_Size,DAG)
        else:
            DAG2=DAG

    return DAG2,EigenValue,NumberofEdges