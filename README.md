## EAGL: Learning and Summarizing Graphical Models using Eigen Analysis of Graph Laplacian
### Dependecies 
<ul>
<li>Scipy</li>
<li>Numpy</li>
<li>Networkx</li>
<li>Matplotlib</li>
</ul>

### Enter the following to reproduce the results:
#### Setup 1: Single edge reduction (Method='True') 
<ul>
<li>Enter if tree connection to be maintained (True or False): True</li>
<li>Enter Tree Extraction method (dfs or bfs): bfs</li>
<li>Enter Traversing Start Node(0 to 24): 0</li>
<li>Enter Number of Iterations: 20</li>
<li>Enter for which Eigenvalue (1st or 2nd) perform the calculation: 2</li>
</ul>

#### Setup 2: Multiple edge reduction (Method='False') 
<ul>
<li>Enter if tree connection to be maintained (True or False): True</li>
<li>Enter Tree Extraction method (dfs or bfs): dfs</li>
<li>Enter Traversing Start Node(0 to 24):0</li>
<li>Enter Number of Iterations: 10</li>
<li>Enter for which Eigenvalue (1st or 2nd) perform the calculation: 2</li>
<li>Enter Cut-off Value: 0.005</li>
</ul>
*The code was tested using Python 3.6 
