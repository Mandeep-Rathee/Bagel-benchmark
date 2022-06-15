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


<p>
 The code for our <em>benchmark</em> will be released here !!!
</p>

<h2>Node Classification</h2>

For node classification, we measure **RDT-Fidelity, Sparsity and Correctness**



 
```python
pip istall bagel-benchmark
```

```python
from bagel-benchmark import metrics
from bagel-benchmark.node_classification import models
from bagel-benchmark import exlainers

sparsity = metrics.sparsity(node_masks,feature_mask)


```
