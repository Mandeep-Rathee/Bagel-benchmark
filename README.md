



<h1>Bagel</h1>
<h2>Welcome to our Benchmark!!</h2>
<h1 style="text-align:center">
<img style="vertical-align:middle" width="300" height="120" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/luh_logo.jpg" /> 
<img style="vertical-align:middle" width="150" height="120" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/l3s_logo.jpeg" />
<img style="vertical-align:middle" width="200" height="120" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/TU_Delft-logo.png" />
</h1>
<h1 style="text-align:center">
<img style="vertical-align:middle" width="900" height="380" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/bagel-main.pdf" />

 </h1>

<h2>  
Click <a href="https://arxiv.org/abs/2206.13983" rel="permalink"> here </a> for the arxiv version of Bagel Benchmark.
</h2>

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

We run all our experiments on a servers with intel Xeon Silver 4210 CPU and an INVIDIA A100 GPU.

**Hyperparameters settings for all dataset**

 GNN layers | epochs | optimizer | lr | weight decay | 
 --- | --- | --- |--- |--- |
  2 | 200 | Adam | 0.01 | 5e-4|


```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_set="Cora"
dataset, data, results_path = utils.load_dataset(data_set)
data.to(device)
```
```python
# We train 2 layers GNN models for 200 epochs. 
# We use Adam optimizer with weight decay of 5e-4 and learnin rate of 0.01. 

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

<a> We show a demo for Movie Reviews dataset.</a>  <a>  The raw Movie Reviews text dataset is stored in this <i class="bi bi-file-earmark-pdf"></i><a href="https://github.com/Mandeep-Rathee/Bagel-benchmark/tree/main/bagel_benchmark/dataset/movie_reviews" rel="permalink">folder.</a> 
</a>

 <p> <i class="bi bi-file-earmark-pdf"></i><a href="https://arxiv.org/pdf/1911.03429.pdf" rel="permalink">Movie Reviews dataset</a> </p>
 
 
 
<p>1. <i class="bi bi-file-earmark-pdf"></i><a href="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/bagel_benchmark/dataset/create_movie_reviews.py" rel="permalink">Click here to create the graphs from the text. </a> </p>
for example, for the text

"? romeo and juliet ' , and ? the twelfth night ' . it is easier for me to believe that he had a wet dream and that 's how all his plays develop , but please spare me all of this unnecessary melodrama."

<h3>The following graph represents the text.</h3>

<h1 style="text-align:center">
<img style="vertical-align:middle" width="900" height="380" src="https://github.com/Mandeep-Rathee/Bagel-benchmark/blob/main/Images/text2graph.jpg" />
 </h1>
<p>2. We train the GNN. </p2>

We run all our experiments on a servers with intel Xeon Silver 4210 CPU and an INVIDIA A100 GPU.


**hyperparameters settings**

 Dataset |GNN layers | epochs | optimizer | lr | weight decay | pooling |
 ---|--- | --- | --- |--- |--- |---|
  MUTAG|2 | 200 | Adam | 0.01 | 0.0|mean|
  PROTEINS |2 | 200 | Adam | 0.01 |0.0 |mean|
  Movie Reviews |2 | 200 | Adam | 0.01 | 0.0 |mean|
  ENZYMES |2 | 200 | Adam | 0.001 | 0.0 |mean|

**Details of GNNs**

 GNN |hidden units | 
 ---|--- |
 GCN| 64 |
 GAT| 64 |
 GIN| 32 |
 APPNP |64|



```python
from bagel_benchmark.graph_classification import models

## The molecules dataset can be loaded as

data_set = "ENZYMES"   ### Similalry MUTAG or PROTEINS can be loaded by replacing data_set="MUTAG" or "PROTEINS"
dataset = models.load_dataset(data_set)
```
 
```python
#### load the movie review dataset and train the GNN model

from bagel_benchmark.graph_classification.utils_movie_reviews import load_dataset, train_gnn
from bagel_benchmark.metrics import suff_and_comp
from bagel_benchmark.explainers.grad_explainer_graph import grad_weights


train_loader, test_loader, test_dataset = load_dataset()


dataset_dim = [300,2] ### features size is 300 and there are 2 labels. 
model = models.GCN(dataset_dim)
train_gnn(model, train_loader, test_loader)
```
<p>3. Generate the explanation </p2>

```python
#let idx is the index on the graph in the test loader
data = test_dataset[idx]
data.batch = torch.zeros(data.x.shape[0], device=device).long()


explanation = grad_weights(model, data)

```
<p>4. Finally evaluate the explanation </p2>

```python
suff, comp = suff_and_comp(idx, model,explanation,test_dataset)
```


## Citation

If you find this benchmark useful in your research, Please consider citing our paper:

```BibTeX
@article{rathee2022bagel,
  title={BAGEL: A Benchmark for Assessing Graph Neural Network Explanations},
  author={Rathee, Mandeep and Funke, Thorben and Anand, Avishek and Khosla, Megha},
  journal={arXiv preprint arXiv:2206.13983},
  year={2022}
}
```







