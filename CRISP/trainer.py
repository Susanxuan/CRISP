"""
The Trainer framework is adapted from chemCPA
"""
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat
import numpy as np
import torch
import pickle
import copy

from .data import load_dataset_splits, custom_collate
from .embedding import get_chemical_representation
from .model import PertAE
from .eval import evaluate


class Trainer:

    def __init__(self):
        pass

    def init_dataset(self, data_params: dict,seed):

        self.datasets = load_dataset_splits(
            **data_params,seed=seed
        )

    def init_drug_embedding(self, embedding: dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drug_embeddings = get_chemical_representation(
            smiles=self.datasets['training'].canon_smiles_unique_sorted,
            embedding_model=embedding["model"],
            data_dir=embedding["directory"],
            device=device,
        )
    

    def init_model(
        self,
        hparams: dict,
        seed: int,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autoencoder = PertAE(
            self.datasets["training"].num_genes,
            self.datasets["training"].num_drugs,
            self.datasets['training'].num_celltypes,
            self.datasets["training"].num_covariates,
            device=device,
            seed=seed,
            hparams=hparams,
            FM_ndim=self.datasets["training"].paired_cell_embeddings.shape[1],
            drug_embeddings=self.drug_embeddings,
        )

    def load_train(self):
        """
        Instantiates a torch DataLoader for the given batchsize
        """
        self.datasets.update(
            {
                "loader_tr": torch.utils.data.DataLoader(
                    self.datasets["training"],
                    batch_size=self.autoencoder.hparams["batch_size"],
                    collate_fn=custom_collate,
                    shuffle=True,
                    drop_last=True,
                )
            }
        )

    def train(
        self,
        num_epochs: int,
        max_minutes: int,
        checkpoint_freq: int,
        save_dir: str,
        ood_ctrl_dataset=None, # can input a custom ood control and treated dataset for ood evaluation
        ood_treat_dataset=None,
        eval_ood=True, # whether to conduct ood evaluation
    ):
        
        assert save_dir is not None
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir()

        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_training_stats = defaultdict(float)
            # if self.autoencoder.hparams['change_bs']:
            #     if epoch in [50,75]:
            #         bs = self.datasets["loader_tr"].batch_size
            #         self.datasets["loader_tr"] = torch.utils.data.DataLoader(
            #                 self.datasets["training"],
            #                 batch_size=bs*2,
            #                 collate_fn=custom_collate,
            #                 shuffle=True,)
            #         logging.info(f'Batch size change from {bs} to {self.datasets["loader_tr"].batch_size}')

            for data in self.datasets["loader_tr"]:
                genes, paired_cell_embeddings, paired_mean, paired_std, drugs_idx, dosages, degs, celltype_idx = data[:8]
                
                neg_genes, neg_paired_cell_embeddings, neg_paired_mean, neg_paired_std, neg_drugs_idx, neg_dosages, neg_degs, neg_celltype_idx = data[8:16]

                covariates,neg_covariates = data[16], data[17]

                training_stats = self.autoencoder.iter_update(
                    genes=genes,
                    cell_embeddings=paired_cell_embeddings,
                    paired_mean=paired_mean,
                    paired_std=paired_std,
                    drugs_idx=drugs_idx,
                    dosages=dosages,
                    degs=degs,
                    celltype_idx=celltype_idx,
                    covariates=covariates,
                    neg_genes=neg_genes,
                    neg_cell_embeddings=neg_paired_cell_embeddings,
                    neg_paired_mean=neg_paired_mean,
                    neg_paired_std=neg_paired_std,
                    neg_drugs_idx=neg_drugs_idx,
                    neg_dosages=neg_dosages,
                    neg_degs=neg_degs,
                    neg_celltype_idx=neg_celltype_idx,
                    neg_covariates=neg_covariates,
                )

                for key, val in training_stats.items():
                    epoch_training_stats[key] += val

            self.autoencoder.scheduler_autoencoder.step()
            self.autoencoder.scheduler_cell.step()
            if self.autoencoder.num_drugs > 0:
                self.autoencoder.scheduler_dosers.step()

            for key, val in epoch_training_stats.items():
                epoch_training_stats[key] = val / len(self.datasets["loader_tr"])
                if key not in self.autoencoder.history.keys():
                    self.autoencoder.history[key] = []
                self.autoencoder.history[key].append(val)
            self.autoencoder.history["epoch"].append(epoch)

            # print some stats for each epoch
            epoch_training_stats["epoch"] = epoch
            logging.info("\n%s", pformat(dict(epoch_training_stats), indent=4, width=1))

            ellapsed_minutes = (time.time() - start_time) / 60
            self.autoencoder.history["elapsed_time_min"] = ellapsed_minutes
            reconst_loss_is_nan = math.isnan(
                epoch_training_stats["loss_reconstruction"]
            )

            stop = (
                ellapsed_minutes > max_minutes
                or (epoch == num_epochs - 1)
                or reconst_loss_is_nan
            )

            # we always run the evaluation when training has stopped
            if ((epoch % checkpoint_freq) == 0 and epoch > 0) or stop:
                evaluation_stats = {}
                evaluation_stats_all = {}
                prediction_all = {}
                pred_all = {}
                true_all = {}

                if stop:
                    output_all = True
                else:
                    output_all = False

                with torch.no_grad():
                    self.autoencoder.eval()
                    evaluation_stats['iid'], evaluation_stats_all['iid'], prediction_all['iid'], _,_ = evaluate(
                        self.autoencoder,
                        self.datasets["test_treated"],
                        self.datasets['test_control'],
                        False,
                    )
                    if eval_ood:
                        if ood_ctrl_dataset is None:
                            # in this case, normally we include the control state gene expression of ood cell type 
                            # in training dataset, that does not requires to calculate a separate cell type FM-embedding.
                            evaluation_stats['ood'], evaluation_stats_all['ood'], prediction_all['ood'], pred_all, true_all= evaluate(
                                self.autoencoder,
                                self.datasets["ood_treated"],
                                self.datasets['ood_control'],
                                output_all,
                            )
                        else:
                            # In this case, we commonly want to predict for ood cell types that do not appears in training dataset,
                            # neither control state nor treated state. It requires users to provide custom cell type FM-embeddings that
                            # calculated from the control state of OOD dataset.
                            ood_ae = copy.deepcopy(self.autoencoder)
                            # if ood_celltype_emb is not None:
                            #     ood_ae.celltype_embeddings = ood_celltype_emb
                            evaluation_stats['ood'], evaluation_stats_all['ood'],prediction_all['ood'], pred_all, true_all = evaluate(
                                ood_ae,
                                ood_treat_dataset,
                                ood_ctrl_dataset,
                                output_all,
                            )
                            del ood_ae
                    
                    self.autoencoder.train()

                test_score = (
                    np.mean(list(evaluation_stats["iid"].values()))
                    if evaluation_stats["iid"]
                    else None
                )

                test_score_is_nan = test_score is not None and math.isnan(test_score)
                stop = stop or test_score_is_nan

                if stop:
                    file_name = f'model.pt'
                    torch.save(
                        (
                            self.autoencoder.state_dict(),
                            self.autoencoder.init_args,
                            self.autoencoder.history,
                        ),
                        os.path.join(save_dir, file_name),
                    )
                    logging.info(f"model_saved: {file_name}")

                    with open(save_dir+'/eval_stats.pkl','wb') as f:
                        pickle.dump(evaluation_stats,f)
                    with open(save_dir+'/eval_stats_all.pkl','wb') as f:
                        pickle.dump(evaluation_stats_all,f)  
                    with open(save_dir+'/pred_mean.pkl','wb') as f:
                        pickle.dump(prediction_all,f)
                    # with open(save_dir+'/pred_all.pkl','wb') as f:
                    #     pickle.dump(pred_all,f)
                    # with open(save_dir+'/true_all.pkl','wb') as f:
                    #     pickle.dump(true_all,f)
                if (
                    stop
                    and not reconst_loss_is_nan
                    and not test_score_is_nan
                ):
                    for key, val in evaluation_stats.items():
                        if key not in self.autoencoder.history:
                            self.autoencoder.history[key] = []
                        self.autoencoder.history[key].append(val)
                    self.autoencoder.history["stats_epoch"].append(epoch)

                # print some stats for the evaluation
                stats = {
                    "epoch": epoch,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                    "max_minutes_reached": ellapsed_minutes > max_minutes,
                    "max_epochs_reached": epoch == num_epochs - 1,
                }

                logging.info("\n%s", pformat(stats, indent=4, width=1))

        results = self.autoencoder.history
        results["total_epochs"] = epoch
        return results
