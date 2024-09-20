import torch
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from src.train import Trainer
import os

@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def main(config: DictConfig):
    print("Configuration used:")
    print(OmegaConf.to_yaml(config))

    debug = config.get("debug", False)
    config["run_name"] = f'{config["dataset"].get("name", "oc20")}_{config["model"].get("name", "faenet")}_{config.get("experiment_name", "")}'

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache() 
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000" # Depends on VRAM available
    else:
        device = torch.device("cpu")

    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer = Trainer(config_dict, debug=debug, device=device)
    trainer.train()

if __name__ == "__main__":
    main()