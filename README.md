# Benchmarking Graph Neural Networks on Dynamic Link Prediction
A framework for training graph neural networks on dynamic networks (temporal graphs). The framework enables the comparison of discrete and continuous models on the discrete network representation. Four types of methods are supported: Link predicion heuristics, static GNNs, discrete DGNNs and continuous DGNNs. This framework is a heavily modified extension of the [EvolveGCN framework](https://github.com/IBM/EvolveGCN).

## 1. Installation
### 1.1 Requirements
1. Install [Singularity 3.5.3](https://github.com/hpcng/singularity/blob/master/INSTALL.md).
2. Install [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive) (Optional for CPU training)

Memory requirements differ between datasets and models. 64GB RAM and 24GB vRAM is required for the largest dataset-models combinations (e.g. EGCN on Autonomous). 

### 1.2 Build container
In the root directury, run the following script to build the singularity container. Expect the building to take a couple of hours. The script builds the container and places it in the parent directory (outside of the repository). The container takes up roughly 7.5GB disk space.
```
./build_container.sh 
```

If you do not wish to use a singularity container, you can look at the installation process in the container definition file, `container.def` and install the dependencies manually. 

## 2. Download datasets
| Datasets           | Download                                                         |
| ------------------ | ---------------------------------------------------------------- |
| Enron              | [link](http://networkrepository.com/ia-enron-employees.php)      |
| UC Irvine          | [link](http://konect.cc/networks/opsahl-ucforum/)                |
| Bitcoin-OTC        | [link](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) |
| Autonomous-systems | [link](http://snap.stanford.edu/data/as-733.html)                |
| Wikipedia          | [link](https://snap.stanford.edu/jodie/)                         |
| Reddit             | [link](https://snap.stanford.edu/jodie/)                         |

Place the downloaded datasets in the `experiment/data` folder. 

## 3. Implemented GNN models
| Models      | Paper                                    | Code                                                                                                                                       |
| ----------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| GCN         | [link](https://arxiv.org/abs/1609.02907) | [link](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)                                 |
| GAT         | [link](https://arxiv.org/abs/1710.10903) | [link](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)                                 |
| EvolveGCN-H | [link](https://arxiv.org/abs/1902.10191) | [link](https://github.com/IBM/EvolveGCN)                                                                                                   |
| EvolveGCN-O | [link](https://arxiv.org/abs/1902.10191) | [link](https://github.com/IBM/EvolveGCN)                                                                                                   |
| GC-LSTM     | [link](https://arxiv.org/abs/1812.04206) | [link](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gc_lstm.GCLSTM) |
| TGAT        | [link](https://arxiv.org/abs/2002.07962) | [link](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)                                       |
| TGN         | [link](https://arxiv.org/abs/2006.10637) | [link](https://arxiv.org/abs/2006.10637)                                                                                                   |
## 4. Quick run
Code and scripts relevant to the experiment is found in the `experiment` folder. You can run a quick test run (3 min) with the `quick.yaml` found in the `config` folder. Run scripts are found in the `run_scripts` folder. How to run a quick run is shown below.

```
./run_scripts/run_cpu.sh config/quick.yaml #Change to run_gpu.sh if you want to untilize the GPU
```

If you wish to run the quick run without the helper script that can be done with the following command.
```
singularity exec ../../container.sif python run_exp.py --config_file config/quick.yaml
```

## 5. Running in the background
Helper scripts for convenient background running is also included.
```
./run_scripts/nohup_run_gpu.sh config/quick.yaml
```

## 6. Reproducing results
### 6.1 Grid search
To replicate the grid search, use the configurations in the `config/grid_searches` folder. For example to start a grid search with the Enron dataset using the GCN model.
```
./run_scripts/run_gpu.sh config/grid_searches/enron_gcn.yaml
```

For a different dataset or model, use the appropriate yaml file in the `config/grid_searches` folder.

Beware that most of the grid searches (one model on one dataset) takes days (in some cases weeks) to finish if run on a single computer with a single GPU.

It is also possible to run just one parameter setting. This is useful for running multiple parameter settings in parallel.
```
./run_scripts/run_gpu.sh config/grid_searches/enron_gcn.yaml --one_cell
```

By default the framework will check the log folders to see whether the next cell has already been run, and if it finds a log corresponding to that cell, it will skip to the next one. Thus it is simply to restart a grid search if it is interrupted.

### 6.2 Final results
The best parameters found during our grid search is in the `config/stability` folder. To replicate our results, run these configurations with 4 different seeds. Again an example with Enron and GCN.

```
./run_scripts/run_gpu.sh config/stability/enron_gcn.yaml
```

##### mAP scores of GNNs
| Models  | Enron           | UC              | Bitcoin            | Autonomous      | Wikipedia        | Reddit          |
| ------- | --------------- | --------------- | ------------------ | --------------- | ---------------- | --------------- |
| GCN     | 0.3222 ± 0.0216 | 0.0205 ± 0.0047 | 0.0019 ± 0.0018    | 0.0319 ± 0.0057 | 0.00005 0.000004 | 0.0378 ± 0.0068 |
| GAT     | 0.3471 ± 0.0202 | 0.0180 ± 0.0070 | 0.0006 ± 0.0005    | 0.0296 ± 0.0167 | 0.0004 ± 0.0006  | 0.0057 ± 0.0054 |
| EGCN-H  | 0.3290 ± 0.0477 | 0.0137 ± 0.0030 | 0.0014 ± 0.0009    | 0.1818 ± 0.0823 | 0.0035 ± 0.0008  | 0.0381 ± 0.0100 |
| EGCN-O  | 0.3376 ± 0.0311 | 0.0249 ± 0.0011 | 0.0028 ± 0.0012    | 0.1650 ± 0.0327 | 0.0047 ± 0.0014  | 0.0394 ± 0.0054 |
| GC-LSTM | 0.2700 ± 0.0227 | 0.0474 ± 0.0025 | 0.0020 ± 0.0011    | 0.4063 ± 0.0236 | 0.0073 ± 0.0012  | 0.0973 ± 0.0074 |
| TGAT    | 0.0582 ± 0.0039 | 0.0007 ± 0.0006 | 0.0002 ± 0.0001    | NA              | 0.0061 ± 0.0004  | 0.1077 ± 0.0073 |
| TGN     | 0.0545 ± 0.0085 | 0.0072 ± 0.0008 | 0.0001 ± 0.0000    | NA              | 0.0042 ± 0.0004  | 0.0417 ± 0.0091 |
