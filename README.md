# grm-faenet
A light-weight reimplementation of [FAENet](https://arxiv.org/pdf/2305.05577.pdf) in PyTorch with experiments on small sized datasets. This is a student project for the [Graphical Models: Discrete Inference and Learning course](https://thoth.inrialpes.fr/people/alahari/disinflearn/) by Ali Ramlaoui and Théo Saulus.


## Installation
The main modules of the project (FAENet) do not use `torch_geometric` and the main functions of the library are reimplemented in `src/gnn_utils.py`. However, the library is still needed to load the datasets and process batches correctly.

```bash
pip install -r requirements.txt
```

## Datasets
The datasets are automatically downloaded when running the experiments depending on the dataset specified. There are individual scripts to download the datasets in the `data` folder.

For Linux:
```bash
./data/download_data.sh is2re
```
For Windows:
```Powershell
./data/download_data.ps1 -task is2re
```

The following datasets are currently available:
- OC20 10k split for IS2RE

## Usage
The main script to run the experiments is `main.py`. The script can be run with the following arguments for a given <dataset>, <model> and <experiment> (optional):
```bash
python main.py dataset=<dataset> model=<model> +experiment=<experiment>
```
It is therefore possible to create new datasets and models and specifying them correctly in the configs folder to be able to take them into account.

For example, to run the experiments on the `oc20` dataset with the `faenet` model. By default, stochastic Frame Averaging is used for the models but can be changed with the experiment used:
```bash
python main.py dataset=oc20 model=faenet
```

It is also possible to modify the `default_config` file in the `configs` folder to change the default parameters of the experiments by creating a new config file and using the `--config-name=<config>` argument.
