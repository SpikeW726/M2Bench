"""GRU and LSTM actor, critic, and Q-network modules.

External hidden states always use shape ``(recurrent_N, batch, hidden_size)``.
LSTMs split this tensor into internal ``(h, c)`` states while preserving the
common external interface. ``recurrent_N`` is the layer count for GRUs and twice
the layer count for LSTMs.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from networks.mlp import layer_init

class _BaseRNN(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 1,
        rnn_type: str = "gru",
        output_std: float = 0.01,
        fc_hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.fc_hidden: List[int] = list(fc_hidden) if fc_hidden else []

        fc_sizes = list(fc_hidden) if fc_hidden else [hidden_size]
        layers = []
        prev = input_dim
        for sz in fc_sizes:
            layers.append(layer_init(nn.Linear(prev, sz)))
            layers.append(nn.Tanh())
            prev = sz
        self.fc_in = nn.Sequential(*layers)
        rnn_input_size = fc_sizes[-1]

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.fc_out = layer_init(nn.Linear(hidden_size, output_dim), std=output_std)

    def _apply(self, fn):
        ret = super()._apply(fn)
        if hasattr(self.rnn, "flatten_parameters"):
            p0 = next(self.rnn.parameters(), None)
            if p0 is not None and p0.is_cuda:
                self.rnn.flatten_parameters()
        return ret

    # hidden state helpers.

    @property
    def recurrent_N(self) -> int:
        return self.num_layers * (2 if self.rnn_type == "lstm" else 1)

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.recurrent_N, batch_size, self.hidden_size, device=device)

    def _to_rnn_state(self, hidden: torch.Tensor):
        if self.rnn_type == "lstm":
            h, c = hidden.chunk(2, dim=0)
            return (h.contiguous(), c.contiguous())
        return hidden.contiguous()

    def _from_rnn_state(self, state) -> torch.Tensor:
        if self.rnn_type == "lstm":
            return torch.cat(state, dim=0)
        return state

    # forward / forward_sequence.

    def _rnn_forward(self, x_features: torch.Tensor, hidden_state: torch.Tensor):
        """x_features: (seq_len, batch, hidden_size)"""

        if hasattr(self.rnn, "flatten_parameters"):
            p0 = next(self.rnn.parameters(), None)
            if p0 is not None and p0.is_cuda:
                self.rnn.flatten_parameters()
        rnn_state = self._to_rnn_state(hidden_state)
        rnn_out, new_state = self.rnn(x_features, rnn_state)
        return rnn_out, self._from_rnn_state(new_state)

    def _head(self, rnn_out: torch.Tensor) -> torch.Tensor:
        return self.fc_out(rnn_out)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor = None):
        if hidden_state is None:
            hidden_state = self.get_initial_hidden(obs.shape[0], obs.device)
        x = self.fc_in(obs).unsqueeze(0)               # (1, batch, H).
        rnn_out, new_hidden = self._rnn_forward(x, hidden_state)
        output = self._head(rnn_out.squeeze(0))         # (batch, output_dim).
        return output, new_hidden

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        hidden_state: torch.Tensor,
        done_seq: Optional[torch.Tensor] = None,
    ):
        seq_len, batch, _ = obs_seq.shape
        if done_seq is None:
            x = self.fc_in(obs_seq.reshape(seq_len * batch, -1))
            x = x.view(seq_len, batch, -1)
            rnn_out, final_hidden = self._rnn_forward(x, hidden_state)
            output = self._head(rnn_out.reshape(seq_len * batch, -1))
            output = output.view(seq_len, batch, -1)
            return output, final_hidden

        if done_seq.shape != (seq_len, batch):
            raise ValueError(
                f"done_seq must be (seq_len, batch)={(seq_len, batch)}, got {tuple(done_seq.shape)}"
            )

        h = hidden_state.clone()
        out_list: List[torch.Tensor] = []
        for t in range(seq_len):
            if t > 0:
                d = done_seq[t - 1]
                if d.dtype != torch.bool:
                    d = d > 0.5
                if d.any():
                    h[:, d, :] = 0
            xf = self.fc_in(obs_seq[t]).unsqueeze(0)  # (1, batch, enc_dim).
            rnn_out, h = self._rnn_forward(xf, h)
            out_t = self._head(rnn_out.squeeze(0))
            out_list.append(out_t)
        output = torch.stack(out_list, dim=0)
        return output, h

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        return {
            "type": type(self).__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "fc_hidden": self.fc_hidden,
        }

class ActorRNN(_BaseRNN):
    def __init__(self, input_dim, hidden_size, output_dim,
                 num_layers=1, rnn_type="gru", fc_hidden=None):
        super().__init__(input_dim, hidden_size, output_dim,
                         num_layers, rnn_type, output_std=0.01,
                         fc_hidden=fc_hidden)

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "ActorRNN":
        return cls(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            output_dim=cfg["output_dim"],
            num_layers=cfg.get("num_layers", 1),
            rnn_type=cfg.get("rnn_type", "gru"),
            fc_hidden=cfg.get("fc_hidden") or None,
        )

class CriticRNN(_BaseRNN):
    """Recurrent critic with scalar output and unit initialization gain."""

    def __init__(self, input_dim, hidden_size, output_dim=1,
                 num_layers=1, rnn_type="gru", fc_hidden=None):
        super().__init__(input_dim, hidden_size, output_dim,
                         num_layers, rnn_type, output_std=1.0,
                         fc_hidden=fc_hidden)

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "CriticRNN":
        return cls(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            output_dim=cfg.get("output_dim", 1),
            num_layers=cfg.get("num_layers", 1),
            rnn_type=cfg.get("rnn_type", "gru"),
            fc_hidden=cfg.get("fc_hidden") or None,
        )

class QRNN(_BaseRNN):
    def __init__(self, input_dim, hidden_size, output_dim,
                 num_layers=1, rnn_type="gru", dueling=False, fc_hidden=None):
        super().__init__(input_dim, hidden_size, output_dim,
                         num_layers, rnn_type, output_std=1.0,
                         fc_hidden=fc_hidden)
        self.dueling = dueling
        if dueling:
            self.v_stream = layer_init(nn.Linear(hidden_size, 1), std=1.0)
            self.a_stream = layer_init(nn.Linear(hidden_size, output_dim), std=1.0)

            del self.fc_out

    def _head(self, rnn_out: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            v = self.v_stream(rnn_out)
            a = self.a_stream(rnn_out)
            return v + a - a.mean(dim=-1, keepdim=True)
        return self.fc_out(rnn_out)

    def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
        cfg = super().get_config_dict(input_dim, output_dim)
        cfg["dueling"] = self.dueling
        return cfg

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "QRNN":
        return cls(
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            output_dim=cfg["output_dim"],
            num_layers=cfg.get("num_layers", 1),
            rnn_type=cfg.get("rnn_type", "gru"),
            dueling=cfg.get("dueling", False),
            fc_hidden=cfg.get("fc_hidden") or None,
        )
