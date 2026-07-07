import torch
import torch.nn as nn


class RelationMixer(nn.Module):
    def __init__(self, pred_len, emb_dim=None, hidden_dim=128, input_mode='retrieved_plus_query'):
        super().__init__()
        if input_mode not in ('retrieved', 'retrieved_plus_query'):
            raise ValueError(f'Unsupported relation_mixer_input: {input_mode}')
        self.input_mode = input_mode
        feat_dim = pred_len if input_mode == 'retrieved' else pred_len + int(emb_dim)
        self.score_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, relation_outputs, relation_query_embs=None):
        if self.input_mode == 'retrieved_plus_query':
            if relation_query_embs is None:
                raise ValueError('relation_query_embs is required for retrieved_plus_query')
            feat = torch.cat([relation_outputs, relation_query_embs], dim=-1)
        else:
            feat = relation_outputs

        scores = self.score_net(feat).squeeze(-1)
        beta = torch.softmax(scores, dim=1)
        y_ret = (beta.unsqueeze(-1) * relation_outputs).sum(dim=1)
        return y_ret, beta, scores
