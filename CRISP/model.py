""" 
The drug perturbation encoding module is adapted from chemCPA
"""

import json
import logging
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .losses import MMDloss, MMD_group, AFMSELoss, dir_loss, compute_mmd,loss_adapt
from geomloss import SamplesLoss

def _move_inputs(*inputs, device="cuda"):
    def mv_input(x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return [mv_input(y) for y in x]

    return [mv_input(x) for x in inputs]

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(
        self,
        sizes,
        dropout,
        batch_norm=True,
        last_layer_act="linear",
        append_layer_position=None,
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]

        layers = [l for l in layers if l is not None][:-2]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        layers_dict = OrderedDict(
            {str(i): module for i, module in enumerate(layers)}
        )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            # dim = x.size(1) // 2
            # return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
            return self.relu(x)
        return self.network(x)


class vaencoder_FM(torch.nn.Module):
    def __init__(self,dims,n_latent,dropout):
        super(vaencoder_FM, self).__init__()
        self.encoderlayers = torch.nn.ModuleList()
        self.encodernorms = torch.nn.ModuleList()
        for nin, nout in zip(dims[:-1],dims[1:]):
            self.encoderlayers.append(torch.nn.Linear(nin,nout))
            self.encodernorms.append(torch.nn.BatchNorm1d(nout))
        self.mu = torch.nn.Linear(dims[-1],n_latent)
        self.logvar = torch.nn.Linear(dims[-1],n_latent)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.LeakyReLU()
    def forward(self,x):
        for encoderlayer, encodernorm in zip(self.encoderlayers,self.encodernorms):
            x = self.dropout(self.relu(encodernorm(encoderlayer(x))))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class PertAE(torch.nn.Module):

    def __init__(
        self,
        num_genes: int, # number of genes
        num_drugs: int, # the number of drugs in whole dataset, including test and ood
        num_celltypes: int, # the number of cell types in training dataset
        num_covariates: int, # the number of each covariate expect from cell type in whole dataset, could be [0]
        device="cpu",
        seed=0,
        hparams="",
        FM_ndim = 512,
        drug_embeddings=None,
    ):
    
        super(PertAE, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_covariates = num_covariates 
        self.num_celltypes = num_celltypes
        self.device = device
        self.FM_ndim = FM_ndim
        
        if self.num_covariates==[0]:
            self.num_latents = int(2) # drug, genes, celltype
        else:
            self.num_latents = int(len(self.num_covariates)+2)

        # set hyperparameters
        if isinstance(hparams, dict):
            self.hparams = hparams
        else:
            self.set_hparams_(hparams)

        if 'single_mmd' not in self.hparams.keys():
            self.hparams['single_mmd'] = False

        # store the variables used for initialization (allows restoring model later).
        self.init_args = {
            "num_genes": num_genes,
            "num_drugs": num_drugs,
            "num_covariates": num_covariates,
            "num_celltypes": num_celltypes,
            "FM_ndim": FM_ndim,
            "hparams": hparams,
        }

        self.encoder_FM = MLP(
            [self.FM_ndim]
            + [self.hparams["encoder_width"]] * self.hparams["encoder_depth"]
            + [self.hparams["lat_dim"]],
            dropout=self.hparams['dropout'],
        )

        self.decoder = MLP(
            [self.hparams["lat_dim"] * self.num_latents]
            + [self.hparams["decoder_width"]] * self.hparams["decoder_depth"]
            + [num_genes],
            dropout=self.hparams['dropout'],
            last_layer_act='ReLU',
            append_layer_position="last",
        )

        # if we initialize cell type embedding with FM-encoded embs
        if self.num_celltypes>0:
            # self.celltype_embeddings = celltype_embeddings

            # add cell type encoder and classifier
            # self.ct_emb_encoder = MLP(
            #     [self.celltype_embeddings.embedding_dim]
            #     + [self.hparams["embedding_encoder_width"]]
            #     * self.hparams["embedding_encoder_depth"]
            #     + [self.hparams["lat_dim"]],
            #     dropout=self.hparams['dropout'],
            #     last_layer_act="linear",
            # ).to(self.device)

            self.ct_predictor = MLP(
                [self.hparams["lat_dim"] * self.num_latents]
                + [128] 
                + [self.num_celltypes],
                dropout=self.hparams['dropout'],
            )
        # if we use one hot encoder and randomly initialize cell type embedding
        else: 
            # self.celltype_embeddings = torch.nn.Embedding(
            #         self.num_celltypes, self.hparams["lat_dim"]
            #     )
            # self.ct_emb_encoder = None
            self.ct_predictor = None
    
        # initialize drug embeddings from rdkit model        
        if self.num_drugs > 0:
            if drug_embeddings is None:
                self.drug_embeddings = torch.nn.Embedding(
                    self.num_drugs, self.hparams["lat_dim"]
                )
                embedding_requires_grad = True
            else:
                self.drug_embeddings = drug_embeddings
                embedding_requires_grad = False

            self.drug_embedding_encoder = MLP(
                [self.drug_embeddings.embedding_dim]
                + [self.hparams["embedding_encoder_width"]]
                * self.hparams["embedding_encoder_depth"]
                + [self.hparams["lat_dim"]],
                dropout=self.hparams['dropout'],
                last_layer_act="linear",
            )

            self.dosers = MLP(
                [self.drug_embeddings.embedding_dim + 1]
                + [self.hparams["dosers_width"]] * self.hparams["dosers_depth"]
                + [1],
                dropout=self.hparams['dropout'],
            )

        # randomly initialize other covariates except from cell type
        if self.num_covariates == [0]:
            pass
        else:
            assert 0 not in self.num_covariates

            self.covariates_embeddings = (
                []
            ) 
            for num_covariate in self.num_covariates:
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams["dim"])
                )
            cov_emb_grad = True

        # self.loss_autoencoder = torch.nn.GaussianNLLLoss()
        self.loss_afmse = AFMSELoss()
        self.loss_mse = torch.nn.MSELoss()
        self.loss_cell_pred = torch.nn.CrossEntropyLoss()
        self.contrloss = nn.CosineEmbeddingLoss()
        self.loss_mmd = SamplesLoss(loss='gaussian',blur=1).to(self.device)
        self.iteration = 0
        self.to(self.device)

        # optimizers
        has_drugs = self.num_drugs > 0
        has_covariates = self.num_covariates[0] > 0
        get_params = lambda model, cond: list(model.parameters()) if cond else []
        _parameters = (
            get_params(self.encoder_FM, True)
            + get_params(self.decoder, True)
            + get_params(self.drug_embeddings, has_drugs and embedding_requires_grad)
            + get_params(self.drug_embedding_encoder, has_drugs)
        )

        if self.num_covariates != [0]:
            for emb in self.covariates_embeddings:
                _parameters.extend(get_params(emb, has_covariates and cov_emb_grad))
    
    
        if self.ct_predictor is not None:
            cell_parameters=(get_params(self.ct_predictor,True))
            
        # else:
        #     cell_parameters = (get_params(self.celltype_embeddings,True))
                
        self.optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"],
        )

        self.optimizer_cell = torch.optim.Adam(
            cell_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["cell_wd"],
        )

        if has_drugs:
            self.optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["wd"],
            )

        # learning rate schedulers
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder,
            step_size=self.hparams["step_size_lr"],
            gamma=0.5,
        )
        self.scheduler_cell = torch.optim.lr_scheduler.StepLR(
            self.optimizer_cell,
            step_size=self.hparams["step_size_lr"],
            gamma=0.5,
        )

        if has_drugs:
            self.scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                self.optimizer_dosers,
                step_size=self.hparams["step_size_lr"],
                gamma=0.5,
            )

        self.history = {"epoch": [], "stats_epoch": []}
        
    def set_hparams_(self, hparams):
        self.hparams = {
            "lat_dim": 128,
            "dosers_width": 64,
            "dosers_depth": 3,
            "encoder_width": 256,
            "encoder_depth": 4,
            "decoder_width": 1028,
            "decoder_depth": 4,    
            "embedding_encoder_width": 128,
            "embedding_encoder_depth": 4,        
            "lr": 1e-3,
            "wd": 1e-7,
            "batch_size": 128,
            "step_size_lr": 50,
            "dropout": 0.2,
            "change_bs": 0,
            "alpha": 0.5,
            "cell_pred": 1.5,
            "cell_wd": 0.001,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def compute_drug_embeddings_(self, drugs_idx=None, dosages=None):
        """
        Compute sum of drug embeddings, each of them multiplied by its dose-response curve.
        @param drugs_idx: A vector of dim [batch_size]. Each entry contains the index of the applied drug. The
            index is ∈ [0, num_drugs).
        @param dosages: A vector of dim [batch_size]. Each entry contains the dose of the applied drug.
        @return: a tensor of shape [batch_size, drug_embedding_dimension]
        """
        assert (drugs_idx is not None and dosages is not None)

        drugs_idx, dosages = _move_inputs(
            drugs_idx, dosages, device=self.device
        )

        latent_drugs = self.drug_embeddings.weight

        if len(drugs_idx.size()) == 0:
            drugs_idx = drugs_idx.unsqueeze(0)

        if len(dosages.size()) == 0:
            dosages = dosages.unsqueeze(0)

        assert drugs_idx.shape == dosages.shape and len(drugs_idx.shape) == 1
        # results in a tensor of shape [batchsize, drug_embedding_dimension]
        latent_drugs = latent_drugs[drugs_idx]

        scaled_dosages = self.dosers(
            torch.concat([latent_drugs, torch.unsqueeze(dosages, dim=-1)], dim=1)
        ).squeeze()

        # unsqueeze if batch_size is 1
        if len(scaled_dosages.size()) == 0:
            scaled_dosages = scaled_dosages.unsqueeze(0)

        # Transform and adjust dimension to latent dims
        latent_drugs = self.drug_embedding_encoder(latent_drugs)

        # scale latent vector by scalar scaled_dosage
        return torch.einsum("b,be->be", [scaled_dosages, latent_drugs])

    def predict(
        self,
        cell_embeddings, # FM-encoded paired control embedding 
        drugs_idx=None,
        dosages=None,
        covariates=None,
    ):
        """
        Predict the post-perturbation gene expression profile 
        given paired control embedding, drugs, and cell type, etc
        """
        assert drugs_idx is not None and dosages is not None
        cell_embeddings, drugs_idx, dosages, covariates = _move_inputs(
            cell_embeddings, drugs_idx, dosages, covariates, device=self.device
        )

        latent_basal = self.encoder_FM(cell_embeddings)
        latent_treated = [latent_basal]
        
        if self.num_drugs > 0:
            drug_embedding = self.compute_drug_embeddings_(
                drugs_idx=drugs_idx, dosages=dosages
            )
            latent_treated.append(drug_embedding)

        if self.num_covariates[0] > 0:
            for cov_type, emb_cov in enumerate(self.covariates_embeddings):
                emb_cov = emb_cov.to(self.device)
                cov_idx = covariates[cov_type].argmax(1)
                cell_emb = emb_cov(cov_idx)
                latent_treated.append(cell_emb)

        latent_treated = torch.cat(latent_treated,dim=1)
        gene_reconstructions = self.decoder(latent_treated)

        return gene_reconstructions, latent_treated


    def iter_update(
        self,
        genes,
        cell_embeddings,
        paired_mean=None,
        paired_std=None,
        drugs_idx=None,
        dosages=None,
        degs=None,
        celltype_idx=None,
        covariates=None,
        neg_genes=None,
        neg_cell_embeddings=None,
        neg_paired_mean=None,
        neg_paired_std=None,
        neg_drugs_idx=None,
        neg_dosages=None,
        neg_degs=None,
        neg_celltype_idx=None,
        neg_covariates=None,
    ):
        """
        Calculate loss and update parameters of model
        """
        assert drugs_idx is not None and dosages is not None

        gene_reconstructions, latent_treated = self.predict(
            cell_embeddings=cell_embeddings,
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=covariates,
        )
        neg_gene_reconstructions, neg_latent_treated = self.predict(
            cell_embeddings=neg_cell_embeddings,
            drugs_idx=neg_drugs_idx,
            dosages=neg_dosages,
            covariates=neg_covariates,
        )

        both_latent = torch.concatenate((latent_treated,neg_latent_treated))
        both_celltype = torch.concatenate((celltype_idx,neg_celltype_idx))
        both_genes = torch.concatenate((genes,neg_genes))
        both_recon = torch.concatenate((gene_reconstructions,neg_gene_reconstructions))
        both_degs = torch.concatenate((degs,neg_degs))
        both_paired_mean = torch.concatenate((paired_mean,neg_paired_mean))
        both_paired_std = torch.concatenate((paired_std,neg_paired_std))

        afloss = self.loss_afmse(y=both_genes,pred=both_recon,degs=both_degs)
        mseloss = self.loss_mse(both_genes, both_recon)
        if self.hparams['single_mmd']:
            mmdloss = self.loss_mmd(both_genes, both_recon)
        else:
            mmdloss = MMDloss(both_genes, both_recon)

        negative_labels = -torch.ones(neg_latent_treated.size(0),device=self.device)
        neg_loss = self.contrloss(neg_latent_treated, latent_treated,negative_labels)
        adapt = loss_adapt(pred=both_recon,true=both_genes,mean_ctrl=both_paired_mean,std_ctrl=both_paired_std)

        if self.num_celltypes > 0:
            self.ct_predictor = self.ct_predictor.to(self.device)
            ct_predictions = self.ct_predictor(both_latent)
            ct_pred_loss = self.loss_cell_pred(
                ct_predictions, both_celltype
            )

        alpha = self.hparams['alpha']
        reconstruction_loss = mseloss * alpha + afloss * (1-alpha)
        loss = reconstruction_loss + adapt*self.hparams['adapt'] + ct_pred_loss * self.hparams['cell_pred'] + mmdloss*self.hparams['mmd'] + neg_loss*self.hparams['neg']

        self.optimizer_autoencoder.zero_grad()
        self.optimizer_cell.zero_grad()
        if self.num_drugs > 0:
            self.optimizer_dosers.zero_grad()

        loss.backward()
        self.optimizer_autoencoder.step()
        self.optimizer_cell.step()
        if self.num_drugs > 0:
            self.optimizer_dosers.step()
        self.iteration += 1

        return {
            "loss": loss.item(),
            "autofocus_loss": afloss.item(),
            "mmd_loss": mmdloss.item(),         
            "mse_loss": mseloss.item(),
            "neg_loss": neg_loss.item(),
            "cell_pred": ct_pred_loss.item(),
            "adapt": adapt.item(),
            "loss_reconstruction": reconstruction_loss.item(),
        }
