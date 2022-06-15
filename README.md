<h1 style="text-align:center">
<img style="vertical-align:middle" width="220" height="180" src="https://raw.githubusercontent.com/benchmarkir/beir/main/images/color_logo_transparent_cropped.png" />
</h1>

# Bagel
Welcome to our Benchmark!!


The code for our benchmark will be released here !!!
## Node Classification

For node classification, we measure **RDT-Fidelity, Sparsity and Correctness**



 
```python
pip istall bagel
```

```python
from bagel import metrics
from bagel.node_classification import models
from bagel import exlainers

sparsity = metrics.sparsity(node_masks,feature_mask)


```
