import os
import comet_ml
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src.train import Trainer
from src.predict import Predictor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000" # Depends on VRAM available

@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def main(config: DictConfig):
    # For sweeper multirun
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    print("Configuration used:")
    print(OmegaConf.to_yaml(config))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache() 
    else:
        device = torch.device("cpu")

    mode = config["dataset"].get("mode", "train")  # Add 'mode' to your config
    config["run_name"] = f'{config["dataset"].get("name", "structures")}_{config["model"].get("name", "faenet")}_{config.get("experiment_name", "")}'

    if mode == "train":
        config_dict = OmegaConf.to_container(config, resolve=True)
        trainer = Trainer(config_dict, debug=config.get("debug", False), device=device)
        return trainer.train()
    elif mode == "predict":
        config_dict = OmegaConf.to_container(config, resolve=True)
        predictor = Predictor(config_dict, debug=config.get("debug", False), device=device)
        predictor.validate()
    
if __name__ == "__main__":
    main()