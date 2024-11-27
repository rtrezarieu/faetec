# FAEtec
A geometric-GNN for structural analysis, based on a light-weight [reimplementation](https://arxiv.org/abs/2407.08313) of [FAENet][FAENet](https://arxiv.org/abs/2305.05577).

---

## Installation
```bash
pip install -r requirements.txt
```

---

## Usage  
To train the model, run the `main.py` script. You can specify a new <dataset> in one of two ways:  

1. Pass the dataset name via the command line:  
   ```bash  
   python main.py dataset=<dataset>  
   ```  

2. Modify the `default_config` file in the `configs` folder directly.

---

## Sweeper
The sweeper helps find good hyperparameters configurations. The boundaries of the hyperparameters search are defined in the `default_config` file and can be modified.

Run the sweeper with:  
```bash  
python main.py -m  
```  

---

## Visualization
Once the model is trained, keep the same <dataset> name in the `default_config` file, and run `visualize.py`:
```bash  
python visualize.py  
```  

To visualize an animation of the evolution of the displacements' predictions per epoch, run `visualize_animation.py`.
```bash  
python visualize_animation.py  
```