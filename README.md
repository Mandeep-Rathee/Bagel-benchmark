<h1 style="text-align:center">
<img style="vertical-align:middle" width="300" height="120" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/luh_logo.jpg" /> 
<img style="vertical-align:middle" width="150" height="120" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/l3s_logo.jpeg" />
<img style="vertical-align:middle" width="200" height="120" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/TU_Delft-logo.png" />
</h1>



<h1>Bagel</h1>
<h2>Welcome to our Benchmark!!</h2>
<h1 style="text-align:center">
<img style="vertical-align:middle" width="900" height="380" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/bagel-v21024_1.jpg" />

 </h1>



<h2>Node Classification</h2>

For node classification, we measure **RDT-Fidelity, Sparsity and Correctness**



 
```python
### the pypi package is still in test phase, We will release it soon!!!
pip install bagel-benchmark
```

```python
from bagel_benchmark import metrics
from bagel_benchmark.node_classification import utils
from bagel_benchmark.explainers.grad_explainer_node import grad_node_explanation
```
<p>
 1. load the dataset and train the GNN model.
</p>

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_set="Cora"
dataset, data, results_path = utils.load_dataset(data_set)
data.to(device)
```
```python
model = utils.GCNNet(dataset)
model.to(device)

#### train the GNN model 
accuracy = utils.train_model(model,data)
```
<p> 2. Generate the explanation. </p>

```python
node = 10
feature_mask, node_mask = grad_node_explanation(model,node,data.x, data.edge_index)
print(feature_mask)
print(node_mask)
```
<p>3. Finally we evaluate the explanation.</p>

```python

#### Calculate Sparsity 
feature_sparsity = False
Node_sparsity = True
sparsity = metrics.sparsity(feature_sparsity, Node_sparsity, feature_mask, node_mask)
print(sparsity)

### Calculate RDT-Fidelity

feature_mask = torch.from_numpy(feature_mask).reshape(1,-1)
fidelity = metrics.fidelity(model, node, data.x,data.edge_index, feature_mask=feature_mask)
print(fidelity)



```
<h2>Graph Classification</h2>

For graph classification we measure **Faithfulness (comprehensiveness and sufficiency), Plausibility and RDT-Fidelity**

<a> We show a demo for Movie Reviews dataset.</a>
 <p> <i class="bi bi-file-earmark-pdf"></i><a href="https://arxiv.org/pdf/1911.03429.pdf" rel="permalink">Movie Reviews dataset</a> </p>
 
 Click here to to create the graphs from the text  
 
<p>1. <i class="bi bi-file-earmark-pdf"></i><a href="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/bagel_benchmark/dataset/create_movie_reviews.py" rel="permalink">Generate the graph from text.</a> </p>
for example, for the text

"? romeo and juliet ' , and ? the twelfth night ' . it is easier for me to believe that he had a wet dream and that 's how all his plays develop , but please spare me all of this unnecessary melodrama."

<h3>The following graph represents the text.</h3>

<h1 style="text-align:center">
<img style="vertical-align:middle" width="900" height="380" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/text2graph.jpg" />
 </h1>
<p>2. We train the GNN. </p2>


```python
from bagel_benchmark.graph_classification.utils_movie_reviews import load_dataset, train_gnn
from bagel_benchmark.graph_classification.models import GCN

#### the movie review dataset can be loaded 

train_loader, test_loader = load_dataset()


dataset_dim = [300,2] ### features size is 300 and there are 2 labels. 
model = GCN(dataset_dim)
train_gnn(model, train_loader, test_loader)
```
<p>3. Generate the explanation </p2>

```python
#let idx is the index on the graph in the test loader
explanation = grad_weights(model, test_loader[idx])

```
<p>4. Finally evaluate the explanation </p2>

```python
suff, comp = metrics.suff_and_comp(idx, model,explanation,test_loader)
```






