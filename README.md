# EAGL: Learning and Summarizing Graphical Models using Eigen Analysis of Graph Laplacian
## Dependecies 
<ul>
<li>Scipy</li>
<li>Numpy</li>
<li>Networkx</li>
<li>Matplotlib</li>
</ul>

*The code was tested using Python 3.6

## Summarizing Graphical Models using Eigen Analysis of Graph Laplacian (Steps)
<p align="center">
  <img src="/Images/Algorithm.png"  width="60%" height="60%">
  
  <span style="vertical-align: middle;">Figure: Proposed EAGL Algorithm for summarizing a directed probabilistic graphical model based an available dataset.</span>
</p>

## Enter the following to reproduce the results:
### Setup 1: Single edge reduction (Method='False') 
<ul>
<li>Enter if tree connection to be maintained (True or False): True</li>
<li>Enter Tree Extraction method (dfs or bfs): bfs</li>
<li>Enter Traversing Start Node(0 to 9): 0</li>
<li>Enter Number of Iterations: 20</li>
<li>Enter for which Eigenvalue (1st or 2nd) perform the calculation: 2</li>
</ul>

This will result in:
<code>Number of Edges on the Updated DAG: 30

Compression: 40.0 %</code>

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
<li>Enter Cut-off Value: 0.005</li>
</ul>
This will result in:
Number of Edges on the Updated DAG: 38
Compression: 24.0 %

<p align="center">
  <img src="/Images/Setup_2.png"  width="100%" height="100%">
  
  <span style="vertical-align: middle;">Figure: Summarizing the Barabasi Albert Graph.</span>
</p>
