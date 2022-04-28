from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import ast
from typing import Dict, Any, List, Tuple
from omegaconf import OmegaConf
import hydra
import ipdb
from clearml import Task, StorageManager, Dataset as ClearML_Dataset


from data.data import GraphDataset
from model.GraphModel import GraphEmbedding
from torch_geometric.data import DenseDataLoader, DataLoader


Task.force_requirements_env_freeze(
    force=True, requirements_file="requirements.txt")

Task.add_requirements("hydra-core")
Task.add_requirements("pytorch-lightning")


def get_clearml_params(task: Task) -> Dict[str, Any]:
    """
    returns task params as a dictionary
    the values are casted in the required Python type
    """
    string_params = task.get_parameters_as_dict()
    clean_params = {}
    for k, v in string_params["General"].items():
        try:
            # ast.literal eval cannot read empty strings + actual strings
            # i.e. ast.literal_eval("True") -> True, ast.literal_eval("i am cute") -> error
            clean_params[k] = ast.literal_eval(v)
        except:
            # if exception is triggered, it's an actual string, or empty string
            clean_params[k] = v
    return OmegaConf.create(clean_params)


def get_dataloader(split_name, cfg) -> DataLoader:
    """Get training and validation dataloaders"""
    # =============================================================================
    #     clearml_data_object = ClearML_Dataset.get(
    #         dataset_name=cfg.clearml_dataset_name,
    #         dataset_project=cfg.clearml_dataset_project_name,
    #         dataset_tags=list(cfg.clearml_dataset_tags),
    #         # only_published=True,
    #     )
    # =============================================================================

    raw_dataset = GraphDataset(cfg)
    n = (len(raw_dataset) + 9) // 10
    cfg['num_features'] = raw_dataset.num_features

    ipdb.set_trace()

    if split_name == "dev":
        return cfg, DenseDataLoader(
            raw_dataset[n: 2 * n], batch_size=cfg.batch_size, num_workers=5, collate_fn=GraphDataset.collate_fn
        )
    elif split_name == "test":
        return cfg, DenseDataLoader(
            test_dataset=raw_dataset[:n], batch_size=cfg.batch_size, num_workers=5, collate_fn=GraphDataset.collate_fn
        )
    else:
        return cfg, DenseDataLoader(
            raw_dataset[2 * n:], batch_size=cfg.batch_size, shuffle=True, num_workers=5, collate_fn=GraphDataset.collate_fn
        )


def train(cfg, task) -> GraphEmbedding:
    callbacks = []

    if cfg.checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath="./",
            filename="best_embedding_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
            every_n_epochs=cfg.every_n_epochs,
        )
        callbacks.append(checkpoint_callback)

    if cfg.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
        )
        callbacks.append(early_stop_callback)

    cfg, train_loader = get_dataloader("train", cfg)
    cfg, val_loader = get_dataloader("dev", cfg)

    model = GraphEmbedding(cfg, task)

    trainer = pl.Trainer(
        gpus=cfg.gpu,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
    return model


def test(cfg, model) -> List:
    test_loader = get_dataloader("test", cfg)
    trainer = pl.Trainer(gpus=cfg.gpu, max_epochs=cfg.num_epochs)
    results = trainer.test(model, test_loader)
    return results


@hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    pl.seed_everything(cfg.seed, workers=True)

    if cfg.train:
        task = Task.init(
            project_name="DiffPool",
            task_name="DP-train",
            output_uri="s3://experiment-logging/storage/",
        )
    else:
        task = Task.init(
            project_name="DiffPool",
            task_name="DP-predict",
            output_uri="s3://experiment-logging/storage/",
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    cfg = get_clearml_params(task)
    print("Detected config file, initiating task... {}".format(cfg))

    if cfg.remote:
        task.set_base_docker("nvidia/cuda:11.4.0-runtime-ubuntu20.04")
        task.execute_remotely(queue_name=cfg.queue, exit_process=True)

    if cfg.train:
        model = train(cfg, task)

    if cfg.test:
        if cfg.trained_model_path:
            trained_model_path = StorageManager.get_local_copy(
                cfg.trained_model_path)
            model = GraphEmbedding.load_from_checkpoint(
                trained_model_path, cfg=cfg, task=task
            )

        results = test(cfg, model)


if __name__ == "__main__":
    hydra_main()
