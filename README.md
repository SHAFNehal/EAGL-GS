# EAGL: Learning and Summarizing Graphical Models using Eigen Analysis of Graph Laplacian

## Summarizing Graphical Models using Eigen Analysis of Graph Laplacian (Steps)
<p align="center">
  <img src="/Images/Algorithm.png"  width="60%" height="60%">
  
  <span style="vertical-align: middle;">Figure: Proposed EAGL Algorithm for summarizing a directed probabilistic graphical model based an available dataset.</span>
</p>

## Using the code:

A Jupyter notebook named "<a href="https://github.com/SHAFNehal/EAGL-GS/blob/master/Example_GraphSummarization.ipynb">Example_GraphSummarization.ipynb</a>" is provided for step by step guide to use the "EAGL" library for summarizing graphs. For the using the code directly from python IDE please use the file "<a href="https://github.com/SHAFNehal/EAGL-GS/blob/master/Code/Example_GraphSummarization.py">Example_GraphSummarization.py</a>" provided in the "<a href="https://github.com/SHAFNehal/EAGL-GS/tree/master/Code"> Code </a>" folder. 

## Enter the following when prompt to reproduce the results:
### Setup 1: Single edge reduction (Method='False') 
<ul>
<li>Enter if tree connection to be maintained (True or False): True</li>
<li>Enter Tree Extraction method (dfs or bfs): bfs</li>
<li>Enter Traversing Start Node(0 to 9): 0</li>
<li>Enter Number of Iterations: 20</li>
<li>Enter for which Eigenvalue (1st or 2nd) perform the calculation: 2</li>
</ul>

This will result in:

<code>Number of Edges on the Updated DAG: 30</code>

<code>Compression: 40.0 %</code>

<p align="center">
  <img src="/Images/Setup_1.png"  width="100%" height="100%">
  
  <span style="vertical-align: middle;">Figure: Summarizing the Barabasi Albert Graph.</span>
</p>

### Setup 2: Multiple edge reduction (Method='True') 
<ul>
<li>Enter if tree connection to be maintained (True or False): True</li>
<li>Enter Tree Extraction method (dfs or bfs): dfs</li>
<li>Enter Traversing Start Node(0 to 9):0</li>
<li>Enter Number of Iterations: 10</li>
<li>Enter for which Eigenvalue (1st or 2nd) perform the calculation: 2</li>
<li>Enter Cut-off Value: 0.05</li>
</ul>
This will result in:

<code>Number of Edges on the Updated DAG: 38</code>

<code>Compression: 24.0 %</code>

<p align="center">
  <img src="/Images/Setup_2.png"  width="100%" height="100%">
  
  <span style="vertical-align: middle;">Figure: Summarizing the Barabasi Albert Graph.</span>
</p>

## Dependecies 
<ul>
<li>Scipy</li>
<li>Numpy</li>
<li>Networkx</li>
<li>Matplotlib</li>
</ul>

*The code was tested using Python 3.6
