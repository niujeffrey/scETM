from typing import Any, Mapping, Sequence, Tuple, Union, Iterable
import anndata
import numpy as np
import torch
from torch import nn, optim
from scETM.batch_sampler import CellSampler

class BaseCellModel(nn.Module):
    """Base class for single cell models.

    If you wish to modify scETM or implement other single cell models, consider
    extending this class.

    Attributes:
        clustering_input: name of the embedding used for clustering.
        emb_names: name of embeddings returned by self.forward.
        device: device to store the model parameters.
        n_trainable_genes: number of trainable genes.
        n_fixed_genes: number of fixed_genes. Parameters in the input and
            output layer related to these genes should be fixed. Useful for the
            fine-tuning stage in transfer learning.
        n_batches: number of batches in the dataset.
        need_batch: whether the model need batch infomation.
    """

    clustering_input: str
    emb_names: Sequence[str]

    def __init__(self,
        n_trainable_genes: int,
        n_batches: int,
        n_fixed_genes: int = 0,
        need_batch: bool = False,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        """Initializes the BaseCellModel object.

        Args:
            n_trainable_genes: number of trainable genes.
            n_batches: number of batches in the dataset.
            n_fixed_genes: number of fixed_genes. Parameters in the input and
                output layer related to these genes should be fixed. Useful for the
                fine-tuning stage in transfer learning.
            need_batch: whether the model need batch infomation.
            device: device to store the model parameters.
        """

        super().__init__()
        self.device: torch.device = device
        self.n_trainable_genes: int = n_trainable_genes
        self.n_fixed_genes: int = n_fixed_genes
        self.n_batches: int = n_batches
        self.need_batch: bool = need_batch

    def train_step(self,
        optimizer: optim.Optimizer,
        data_dict: Mapping[str, torch.Tensor],
        hyper_param_dict: Mapping[str, Any]
    ) -> Mapping[str, torch.Tensor]:
        """Execute a training step given a minibatch of data.

        Set the model to train mode, run the forward pass, back propagate the
        gradients, step the optimizer, return the record for this step.

        Args:
            optimizer: optimizer of the model parameters.
            data_dict: a dict containing the current minibatch for training.
            hyper_param_dict: a dict containing hyperparameters for the current
                batch.
        
        Returns:
            A dict storing the record for this training step, which typically
            includes decomposed loss terms, gradient norm, and other values we
            care about.
        """

        self.train()
        optimizer.zero_grad()
        loss, _, new_record = self(data_dict, hyper_param_dict)
        loss.backward()
        norms = torch.nn.utils.clip_grad_norm_(self.parameters(), 500)
        new_record['max_norm'] = norms.cpu().numpy()
        optimizer.step()
        return new_record

    def get_embeddings_and_nll(self,
        adata: anndata.AnnData,
        batch_size: int = 2000,
        emb_names: Union[str, Iterable[str], None] = None,
        batch_col: str = 'batch_indices',
        inplace: bool = True
    ) -> Union[float, Tuple[Mapping[str, np.ndarray], float]]:
        """Calculate embeddings and nll for the given dataset.

        Args:
            adata: the test dataset. adata.n_vars must equal to #genes of this
                model.
            batch_size: batch size for test data input.
            emb_names: names of the embeddings to be returned or stored to
                adata.obsm.
            batch_col: a key in adata.obs to the batch column. Only used when
                self.need_batch is True.
            inplace: whether embeddings will be stored to adata or returned.

        Returns:
            If inplace, only the test nll. Otherwise, return the embeddings as
            a dict and also the test nll.
        """

        assert adata.n_vars == self.n_fixed_genes + self.n_trainable_genes
        assert not self.need_batch or adata.obs[batch_col].nunique() == self.n_batches
        if emb_names is None:
            emb_names = self.emb_names
        self.eval()
        if isinstance(emb_names, str):
            emb_names = [emb_names]

        sampler = CellSampler(adata, batch_size=batch_size, sample_batch_id=self.need_batch, n_epochs=1, batch_col=batch_col, shuffle=False)
        embs = {name: [] for name in emb_names}
        nll = 0.
        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self(data_dict, dict(val=True))
            for name in emb_names:
                embs[name].append(fwd_dict[name].detach().cpu())
            nll += fwd_dict['nll'].detach().item()
        embs = {name: torch.cat(embs[name], dim=0).numpy() for name in emb_names}
        if inplace:
            for emb_name, emb in embs.items():
                adata.obsm[emb_name] = emb
        nll /= adata.n_obs

        if inplace:
            return nll
        else:
            return embs, nll
