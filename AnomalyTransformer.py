
import torch
import torch.nn as nn
import torch.nn.functional as F
from AnomalyAttention import AnomalyAttention

class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, hidden_dim, lambda_=3, heads=4):
        super(AnomalyTransformer, self).__init__()
        self.N = N
        self.d_model = d_model
        self.heads = heads
        self.lambda_ = lambda_
        self.hidden_dim = hidden_dim
        self.attention_layers = AnomalyAttention(N, d_model, heads)
        self.attention_norm = nn.LayerNorm(d_model)
        self.hiddenlayer = nn.Linear(d_model, hidden_dim)
        self.outputLayer = nn.Linear(hidden_dim, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        Z, P_list, S_list = self.attention_layers(x)
        Z = self.attention_norm(Z+x)
        hidden = torch.relu(self.hiddenlayer(Z))
        x_hat = self.outputLayer(hidden)
        x_hat = self.output_norm(x_hat+Z)
        return x_hat, P_list, S_list

    def layer_association_discrepancy(self, Pl, Sl):
        epsilon = 1e-10
        kl_div_sum = F.kl_div(Pl + epsilon, Sl + epsilon, reduction="sum") + F.kl_div(Sl + epsilon, Pl + epsilon, reduction="sum")
        return kl_div_sum

    def association_discrepancy(self, P_list, S_list):
        ass_diss = (1 / len(P_list)) * torch.tensor([
            self.layer_association_discrepancy(P, S) for P, S in zip(P_list, S_list)
        ])
        return ass_diss

    def average_association_discrepency(self, P_list, S_list):
        asso_dis = [self.association_discrepancy(P, S) for P, S in zip(P_list, S_list)]
        assoc_disc = torch.mean(torch.stack(asso_dis), dim=0)
        return assoc_disc

    def loss_function(self, x_hat, x, P_list, S_list, lambda_):
        frob_norm = torch.linalg.norm(x_hat - x, ord="fro")**2
        assoc_disc = self.average_association_discrepency(P_list, S_list)
        kl_div = torch.norm(assoc_disc, p=1)
        return frob_norm - lambda_ * kl_div

    def min_loss(self, x_hat, x, P_list, S_list):
        p_list_detach = [P.detach() for P in P_list]
        return self.loss_function(x_hat, x, p_list_detach, S_list, -self.lambda_)

    def max_loss(self, x_hat, x, P_list, S_list):
        s_list_detach = [S.detach() for S in S_list]
        return self.loss_function(x_hat, x, P_list, s_list_detach, self.lambda_)

    def anomaly_score(self, x):
        x_hat, P_list, S_list = self(x)
        anomaly_score = torch.linalg.norm(x_hat - x, ord="fro")
        assoc_dis = self.average_association_discrepency(P_list, S_list)
        ad = F.softmax(-assoc_dis, dim=0)
        reconstruction_error = torch.linalg.norm((x - x_hat) ** 2, dim=1)
        return ad * reconstruction_error

