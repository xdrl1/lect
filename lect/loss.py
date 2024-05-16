"""
loss function

Jiapan Wang
jiapan.wang@tum.de

InfoNCE and SNNL are modified from

- InfoNCE: https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
- SNNL: https://gitlab.com/afagarap/pt-snnl/-/blob/master/snnl/__init__.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.latlon import calculate_great_circle_distance
from embedding.models import CustomViT, CustomResNet

# Define a triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, mode="embedding", margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor_emb, positive_emb, negative_emb, anchor_loc, positive_loc, negative_loc):

        # image embedding distance
        distance_positive_emb = self.calc_euclidean(anchor_emb, positive_emb)
        distance_negative_emb = self.calc_euclidean(anchor_emb, negative_emb)
        # print(f"positive_emb: {distance_positive_emb}\nnegative_emb: {distance_negative_emb}\n")

        if self.mode == "embedding":
            # distance = image embedding distance
            distance_positive = distance_positive_emb
            distance_negative = distance_negative_emb

        elif self.mode == "geo":
            # # location distance
            # print(f'anchor_loc: {anchor_loc}\npositive_loc: {positive_loc}\nnegative_loc: {negative_loc}')
            try:
                distance_positive_loc = calc_geo_distance(anchor_loc, positive_loc)
                distance_negative_loc = calc_geo_distance(anchor_loc, negative_loc)
                distance_within_loc = calc_geo_distance(positive_loc, negative_loc)
                norm_dis_loc = F.normalize(torch.stack([distance_positive_loc, distance_negative_loc, distance_within_loc]), p=3, dim=1) # [[positive], [negative]]
                # print(f"positive_dis: {distance_positive_loc}\nnegative_dis: {distance_negative_loc}\n")
                # print(f'normalized dis: {norm_dis_loc}\n')
            except:
                print(f"distance embedding error: \n{distance_positive_loc}\n{distance_negative_loc}\n{norm_dis_loc}")

            # distance = image embedding distance + location distance
            distance_positive = distance_positive_emb + norm_dis_loc[0]
            distance_negative = distance_negative_emb + norm_dis_loc[1]
            distance_within = norm_dis_loc[2]

        # loss
        losses = torch.relu(distance_positive - distance_negative + distance_within + self.margin)
        # print(f"loss: {losses}\n")

        return torch.mean(losses)


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
        penalty_mode: 'embedding' or 'geo'

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

    Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='paired', penalty_mode='embedding'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.penalty_mode = penalty_mode

    def forward(self, query, positive_key, negative_keys=None, ancLoc=None, posLoc=None, negLoc=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        penalty_mode=self.penalty_mode,
                        ancLoc=ancLoc, posLoc=posLoc, negLoc=negLoc)


def info_nce(query, positive_key, negative_keys=None,
    temperature=0.1,
    reduction='mean',
    negative_mode='unpaired',
    penalty_mode='embedding',
    ancLoc=None, posLoc=None, negLoc=None):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys
        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)
        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)
        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.
        # Cosine between all combinations
        logits = query @ transpose(positive_key)
        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    if penalty_mode == 'embedding':
        return F.cross_entropy(logits / temperature, labels, reduction=reduction)
    elif penalty_mode == 'geo':
        distPosLoc = calc_geo_distance(ancLoc, posLoc).unsqueeze(0)
        distNegLoc = torch.stack([calc_geo_distance(ancLoc, thisNegLoc) for thisNegLoc in negLoc])
        distWinLoc = torch.stack([calc_geo_distance(posLoc, thisNegLoc) for thisNegLoc in negLoc])
        normDistLoc = F.normalize(torch.cat((distPosLoc, distNegLoc, distWinLoc), 0), p=3, dim=1)
        distPos = normDistLoc[0]
        distNeg = torch.mean(normDistLoc[1:1+negative_keys.size(1)], dim=0)
        distWin = torch.mean(normDistLoc[1+negative_keys.size(1): ], dim=0)
        return F.cross_entropy(logits / temperature, labels, reduction=reduction) + torch.mean(torch.relu(distPos - distNeg + distWin))  # scalar


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def calc_geo_distance(x1, x2):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = []
        for idx, (lon1, lat1, lon2, lat2) in enumerate(zip(x1[0], x1[1], x2[0], x2[1])):
            if lat1 == lat2 and lon1 == lon2:
                d.append(0.0000001)
            else:
                dist = calculate_great_circle_distance(lat1, lon1, lat2, lon2)
                d.append(dist)
        return torch.tensor(d).to(device)
    except:
        print(f"calculating distance error! {dist}, {lat1}, {lon1}, {lat2}, {lon2}")


from typing import Dict, Tuple

class SNNLoss(torch.nn.Module):
    """
    A composite loss of the Soft Nearest Neighbor Loss
    computed at each hidden layer, and a softmax
    cross entropy (for classification) loss or binary
    cross entropy (for reconstruction) loss.

    Presented in
    "Improving k-Means Clustering Performance with Disentangled Internal
    Representations" by Abien Fred Agarap and Arnulfo P. Azcarraga (2020),
    and in
    "Analyzing and Improving Representations with the Soft Nearest Neighbor
    Loss" by Nicholas Frosst, Nicolas Papernot, and Geoffrey Hinton (2019).

    https://arxiv.org/abs/2006.04535/
    https://arxiv.org/abs/1902.01889/
    """

    def __init__(
        self,
        temperature: float = None,
        use_annealing: bool = True,
        code_units: int = 30,
        stability_epsilon: float = 1e-5,
        penalty_mode='embedding'
    ):
        """
        Constructs the Soft Nearest Neighbor Loss.

        Parameters
        ----------
        temperature: float
            The SNNL temperature.
        use_annealing: bool
            Whether to use annealing temperature or not.
        use_sum: bool
            If true, the sum of SNNL across all hidden layers are used.
            Otherwise, the minimum SNNL will be obtained.
        code_units: int
            The number of units in which the SNNL will be applied.
        stability_epsilon: float
            A constant for helping SNNL computation stability.
        """
        super().__init__()
        assert isinstance(
            code_units, int
        ), f"Expected dtype for [code_units]: int, but {code_units} is {type(code_units)}"
        self.temperature = temperature
        self.use_annealing = use_annealing
        self.code_units = code_units
        self.stability_epsilon = stability_epsilon
        self.mode = penalty_mode

    def pairwise_geo_distance(self,
        loc,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )):

        res = torch.zeros([len(loc[0]), len(loc[0])], dtype=torch.float32)
        for i in range(res.size(0)):
            for j in range(res.size(1)):
                res[i][j] = calc_geo_distance(
                    [[loc[0][i]], [loc[1][i]]],
                    [[loc[0][j]], [loc[1][j]]])
        return res.to(device)

    def scale_matrix(self, iM, tarMin, tarMax):
        iM = iM - torch.min(iM)
        iM = iM / (torch.max(iM) - torch.min(iM))
        iM = iM * (tarMax - tarMin)
        return iM

    def forward(
        self,
        model: torch.nn.Module,
        img: torch.Tensor,
        label: torch.Tensor,
        epoch: int,
        loc: list,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> Tuple:
        """
        Defines the forward pass for the Soft Nearest Neighbor Loss.
        """

        if self.use_annealing:
            self.temperature = 1.0 / ((1.0 + epoch) ** 0.55)

        activations = self.compute_activations(model, iFeat=img)

        layers_snnl = []
        for idx, value in activations.items():
            if idx != len(activations.items()) - 1:
                # only compute last-layer embeddings
                continue
            if len(value.shape) > 2:
                # flatten
                value = value.view(value.shape[0], -1)
            distance_matrix = self.pairwise_cosine_distance(features=value)
            if self.mode == "geo":
                geo_dist_matrix = self.pairwise_geo_distance(loc)
                geo_dist_matrix = self.scale_matrix(geo_dist_matrix.clone(), torch.min(distance_matrix), torch.max(distance_matrix))
                distance_matrix = torch.add(distance_matrix, geo_dist_matrix) / 2

            ## old approach of afterwards nornalization
            pairwise_distance_matrix = self.normalize_distance_matrix(
                features=value, distance_matrix=distance_matrix, device=device
            )
            pick_probability = self.compute_sampling_probability(
                pairwise_distance_matrix
            )
            summed_masked_pick_probability = self.mask_sampling_probability(
                label, pick_probability
            )
            snnl = torch.mean(
                -torch.log(self.stability_epsilon + summed_masked_pick_probability)
            )
            layers_snnl.append(snnl)
        snn_loss = torch.stack(layers_snnl)
        snn_loss = snn_loss[-1]
        return snn_loss

    def compute_activations(
        self, model: torch.nn.Module, iFeat: torch.Tensor
    ) -> Dict:
        """
        Returns the hidden layer activations of a model.

        Parameters
        ----------
        model: torch.nn.Module
            The model whose hidden layer representations shall be computed.
        iFeat: torch.Tensor
            The input features.

        Returns
        -------
        activations: Dict
            The hidden layer activations of the model.
        """
        activations = dict()
        if isinstance(model, CustomResNet):
            for index, (name, layer) in enumerate(list(model.model.named_children())):
                if index == 0:
                    activations[index] = layer(iFeat)
                elif index == 9:
                    value = activations[index - 1].view(
                        activations[index - 1].shape[0], -1
                    )
                    activations[index] = layer(value)
                else:
                    activations[index] = layer(activations[index - 1])
        elif isinstance(model, CustomViT):
            for index, (name, layer) in enumerate(list(model.model.named_children())):
                if index == 0:
                    activations[index] = layer(iFeat)
                elif index == 1:
                    x = activations[index - 1]
                    x = x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3))
                    x = x.permute(0, 2, 1)
                    # Expand the class token to the full batch
                    batch_class_token = model.model.class_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    activations[index] = layer(x)
                else:
                    activations[index] = layer(activations[index - 1])
        return activations

    def pairwise_cosine_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns the pairwise cosine distance between two copies
        of the features matrix.

        Parameter
        ---------
        features: torch.Tensor
            The input features.

        Returns
        -------
        distance_matrix: torch.Tensor
            The pairwise cosine distance matrix.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> snnl.pairwise_cosine_distance(a)
        tensor([[1.1921e-07, 7.4125e-02, 1.8179e-02, 1.0152e-01],
                [7.4125e-02, 1.1921e-07, 1.9241e-02, 2.2473e-03],
                [1.8179e-02, 1.9241e-02, 1.1921e-07, 3.4526e-02],
                [1.0152e-01, 2.2473e-03, 3.4526e-02, 0.0000e+00]])
        """
        a, b = features.clone(), features.clone()
        normalized_a = torch.nn.functional.normalize(a, dim=1, p=2)
        normalized_b = torch.nn.functional.normalize(b, dim=1, p=2)
        normalized_b = torch.conj(normalized_b).T
        product = torch.matmul(normalized_a, normalized_b)
        distance_matrix = torch.sub(torch.tensor(1.0), product)
        return distance_matrix

    def normalize_distance_matrix(
        self,
        features: torch.Tensor,
        distance_matrix: torch.Tensor,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> torch.Tensor:
        """
        Normalizes the pairwise distance matrix.

        Parameters
        ----------
        features: torch.Tensor
            The input features.
        distance_matrix: torch.Tensor
            The pairwise distance matrix to normalize.
        device: torch.device
            The device to use for computation.

        Returns
        -------
        pairwise_distance_matrix: torch.Tensor
            The normalized pairwise distance matrix.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> snnl.normalize_distance_matrix(a, distance_matrix, device=torch.device("cpu"))
        tensor([[-1.1921e-07,  9.2856e-01,  9.8199e-01,  9.0346e-01],
                [ 9.2856e-01, -1.1921e-07,  9.8094e-01,  9.9776e-01 ],
                [ 9.8199e-01,  9.8094e-01, -1.1921e-07,  9.6606e-01 ],
                [ 9.0346e-01,  9.9776e-01,  9.6606e-01,  0.0000e+00 ]])
        """
        pairwise_distance_matrix = torch.exp(
            -(distance_matrix / self.temperature)
        ) - torch.eye(features.shape[0]).to(device)
        return pairwise_distance_matrix

    def compute_sampling_probability(
        self, pairwise_distance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the probability of sampling `j` based
        on distance between points `i` and `j`.

        Parameter
        ---------
        pairwise_distance_matrix: torch.Tensor
            The normalized pairwise distance matrix.

        Returns
        -------
        pick_probability: torch.Tensor
            The probability matrix for selecting neighbors.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> distance_matrix = snnl.normalize_distance_matrix(a, distance_matrix)
        >>> snnl.compute_sampling_probability(distance_matrix)
        tensor([[-4.2363e-08,  3.2998e-01,  3.4896e-01,  3.2106e-01],
                [ 3.1939e-01, -4.1004e-08,  3.3741e-01,  3.4319e-01 ],
                [ 3.3526e-01,  3.3491e-01, -4.0700e-08,  3.2983e-01 ],
                [ 3.1509e-01,  3.4798e-01,  3.3693e-01,  0.0000e+00 ]])
        """
        pick_probability = pairwise_distance_matrix / (
            self.stability_epsilon + torch.sum(pairwise_distance_matrix, 1).view(-1, 1)
        )
        return pick_probability

    def mask_sampling_probability(
        self, labels: torch.Tensor, sampling_probability: torch.Tensor
    ) -> torch.Tensor:
        """
        Masks the sampling probability, to zero out diagonal
        of sampling probability, and returns the sum per row.

        Parameters
        ----------
        labels: torch.Tensor
            The labels of the input features.
        sampling_probability: torch.Tensor
            The probability matrix of picking neighboring points.

        Returns
        -------
        summed_masked_pick_probability: torch.Tensor
            The probability matrix of selecting a
            class-similar data points.

        Example
        -------
        >>> import torch
        >>> from snnl import SNNLoss
        >>> _ = torch.manual_seed(42)
        >>> a = torch.rand((4, 2))
        >>> snnl = SNNLoss(temperature=1.0)
        >>> distance_matrix = snnl.pairwise_cosine_distance(a)
        >>> distance_matrix = snnl.normalize_distance_matrix(a, distance_matrix)
        >>> pick_probability = snnl.compute_sampling_probability(distance_matrix)
        >>> snnl.mask_sampling_probability(labels, pick_probability)
        tensor([0.3490, 0.3432, 0.3353, 0.3480])
        """
        masking_matrix = torch.squeeze(torch.eq(labels, labels.unsqueeze(1)).float())
        masked_pick_probability = sampling_probability * masking_matrix
        summed_masked_pick_probability = torch.sum(masked_pick_probability, dim=1)
        return summed_masked_pick_probability

