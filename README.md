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
pip install bagel-benchmark
```

```python
from bagel-benchmark import metrics
from bagel-benchmark.node_classification import utils
from bagel-benchmark import exlainers
```
<p>
 1. load the dataset and train the GNN model.
</p>

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_set="Cora"
dataset, data, results_path = utils.load_dataset(data_set)
data.to(device)
```
```python
model = utils.GCNNet(dataset)
model.to(device)

#### train the GNN model 
accuracy = utils.train(model,data)
```
<p> 2. Generate the explanation. </p>

```python
from explainers.grad_based_explainers import grad_weights
explanation = grad_weights(model,data)
```
<p>3. Finally we evaluate the explanation.</p>

```python
sparsity = metrics.sparsity(explanation)

## we need to define the node_id for which we want to calculate the fidelity

fidelity = metrics.fidelity(model, node_id, data.x,data.edge_index, feature_mask=explanation)

```
<h2>Graph Classification</h2>

For graph classification we measure **Faithfulness (comprehensiveness and sufficiency), Plausibility and RDT-Fidelity**

<a> We show a demo for Movie Reviews dataset.</a>
 <p> <i class="bi bi-file-earmark-pdf"></i><a href="https://arxiv.org/pdf/1911.03429.pdf" rel="permalink">Movie Reviews dataset</a> </p>
 
<p>1. <i class="bi bi-file-earmark-pdf"></i><a href="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/bagel_benchmark/Dataset/create_movie_reviews.py" rel="permalink">Generate the graph from text.</a> </p>
for example, for the text

"? romeo and juliet ' , and ? the twelfth night ' . it is easier for me to believe that he had a wet dream and that 's how all his plays develop , but please spare me all of this unnecessary melodrama."

<h3>The following graph represents the text.</h3>

<h1 style="text-align:center">
<img style="vertical-align:middle" width="900" height="380" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/text2graph.jpg" />
 </h1>
<p>2. We train the GNN. </p2>


```python
from bagel-benchmark.graph_classification import utils_movie_reviews
from bagel-benchmark.graph_classification import models


train_loader, test_loader = utils_movie_reviews.load_dataset


dataset_dim = [300,2] ### features size is 300 and there are 2 labels. 
model = GCN(dataset_dim)
utils_movie_reviews.train_gnn(model)
```
<p>3. Generate explanation </p2>

```python
#let idx is the index on the graph in the test loader
explanation = grad_weights(model, test_loader[idx])

```
<p>4. Finally evaluate the explanation </p2>

```python
suff, comp = metrics.suff_and_comp(idx,model,explanation,test_loader)
```






