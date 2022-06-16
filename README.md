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
pip istall bagel-benchmark
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
<p> 2. Generate the explanation for a given node </p>

```python
from explainers.grad_based_explainers import grad_weights
explanation = grad_weights(model,data)
```
<p>3. Finally we evaluate the explanation</p>

```python
sparsity = metrics.sparsity(explanation)

## we need to define the node_id for which we want to calculate the fidelity

fidelity = metrics.fidelity(model, node_id, data.x, feature_mask=explanation)

```
