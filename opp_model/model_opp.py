"""
model_opp.py

Opponent hand prediction model.

Architecture:
    Same transformer backbone as WallPredictionModel.
    Shared per-opponent MLP head applied independently to each of the 3
    non-observer seats.  Weights are shared across seats (seat-agnostic).

    For opponent i:
        per_opp_input  = [CLS_out | opp_discards[i] | opp_melds[i] | opp_hand_size[i]]
        output logits[i] = shared_mlp(per_opp_input)   # raw, no baseline residual

    No residual baseline is used.  Unlike the wall model (where the uniform prior
    is correct at game start), opponents always have at least one discard visible,
    so per-opponent features already encode the strongest signal.  A uniform
    baseline would require large negative corrections for the obvious case of
    "opponent discarded X → unlikely to still hold X".

Wall reconstruction (post-prediction):
    hidden[t]   = max(0, full_counts[t] - seen[t])
    wall_pred[t] = max(0, hidden[t] - sum_p(opp_counts[p][t]))
    Because each opponent is predicted independently, their sum can exceed
    hidden[t] for some tile; the clamp handles this gracefully.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import torch.nn as nn

from tenhou_db import EVENT_TYPES

_N_EVENT_TYPES  = len(EVENT_TYPES)
_TILE_VOCAB     = 37
_TILE_PAD       = 37
_FULL_COUNTS = [4] * 34 + [1, 1, 1]  # max copies per tile type


def _make_sinusoidal_pe(max_len: int, d_model: int, base: float = 90.0) -> torch.Tensor:
    pe       = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(base) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class OppHandModel(nn.Module):
    """
    Args:
        d_model, nhead, num_layers, tile_emb_dim, event_type_emb_dim,
        player_emb_dim, dropout, max_turn:  same semantics as WallPredictionModel.
    """

    def __init__(
        self,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_layers:         int   = 6,
        tile_emb_dim:       int   = 32,
        event_type_emb_dim: int   = 32,
        player_emb_dim:     int   = 16,
        dropout:            float = 0.1,
        max_turn:           int   = 90,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Embeddings (identical to WallPredictionModel) ---
        self.tile_emb        = nn.Embedding(_TILE_VOCAB + 1, tile_emb_dim, padding_idx=_TILE_PAD)
        self.event_type_emb  = nn.Embedding(_N_EVENT_TYPES, event_type_emb_dim)
        self.player_emb      = nn.Embedding(4, player_emb_dim)
        self.register_buffer("turn_pe", _make_sinusoidal_pe(max_turn + 2, d_model, base=float(max_turn)))

        raw_event_dim = tile_emb_dim + event_type_emb_dim + player_emb_dim + 3
        self.event_proj = nn.Linear(raw_event_dim, d_model)
        self.cls_token  = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )

        # --- Shared per-opponent MLP ---
        # Input: CLS [d_model] + opp_discards [37] + opp_melds [37] + opp_hand_size [1]
        # Output: [37 * 5] logits over counts 0-4 per tile type
        opp_mlp_in = d_model + 37 + 37 + 1
        self.opp_mlp = nn.Sequential(
            nn.Linear(opp_mlp_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, _TILE_VOCAB * 5),
        )

    # ------------------------------------------------------------------ #

    def _encode(
        self,
        event_types:    torch.Tensor,  # [B, S]
        tile_ids:       torch.Tensor,  # [B, S, 4]
        scalars:        torch.Tensor,  # [B, S, 3]
        player_ids:     torch.Tensor,  # [B, S]
        padding_mask:   torch.Tensor,  # [B, S] True=padding
        turn_positions: torch.Tensor,  # [B, S]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_out [B, d_model], event_tokens [B, S, d_model])."""
        B, S = event_types.shape

        tile_embs   = self.tile_emb(tile_ids).sum(dim=2)
        etype_embs  = self.event_type_emb(event_types)
        player_embs = self.player_emb(player_ids)
        raw_event   = torch.cat([tile_embs, etype_embs, player_embs, scalars], dim=-1)
        tokens      = self.event_proj(raw_event)

        pos     = turn_positions.clamp(0, self.turn_pe.size(0) - 1)
        tokens  = tokens + self.turn_pe[pos]

        cls     = self.cls_token.expand(B, -1, -1)
        tokens  = torch.cat([cls, tokens], dim=1)
        mask    = torch.cat([padding_mask.new_zeros(B, 1), padding_mask], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=mask)
        return encoded[:, 0, :], encoded[:, 1:, :]  # CLS, event tokens

    @staticmethod
    def _player_pool(
        event_tokens: torch.Tensor,  # [B, S, d_model]
        player_ids:   torch.Tensor,  # [B, S]
        padding_mask: torch.Tensor,  # [B, S] True=padding
        seat:         int,
    ) -> torch.Tensor:               # [B, d_model]
        """Mean-pool transformer outputs over a specific player's non-padded events."""
        valid = (player_ids == seat) & ~padding_mask          # [B, S]
        n     = valid.float().sum(dim=1, keepdim=True).clamp(min=1)
        return (event_tokens * valid.unsqueeze(-1).float()).sum(dim=1) / n

    # ------------------------------------------------------------------ #

    def forward(
        self,
        event_types:    torch.Tensor,  # [B, S]
        tile_ids:       torch.Tensor,  # [B, S, 4]
        scalars:        torch.Tensor,  # [B, S, 3]
        player_ids:     torch.Tensor,  # [B, S]
        padding_mask:   torch.Tensor,  # [B, S]
        seen:           torch.Tensor,  # [B, 37]
        tiles_remaining: torch.Tensor, # [B]
        opp_discards:   torch.Tensor,  # [B, 3, 37]
        opp_melds:      torch.Tensor,  # [B, 3, 37]
        opp_hand_sizes: torch.Tensor,  # [B, 3]  (normalised by 13)
        turn_positions: torch.Tensor,  # [B, S]
        observer_seat:  int = 0,       # which seat is the observer (0–3)
    ) -> torch.Tensor:                 # [B, 3, 37] logits
        _, event_tokens = self._encode(event_types, tile_ids, scalars, player_ids,
                                       padding_mask, turn_positions)

        # Opponent seats in ascending order, excluding the observer.
        # Must match the ordering used by dataset_opp.py when building opp_discards etc.
        opp_seats = [s for s in range(4) if s != observer_seat]

        # Per-player pool: mean-pool transformer outputs over each opponent's own
        # events only.  This gives a distinct [d_model] representation per seat,
        # directly encoding that player's discard / meld history.
        logits_list = []
        for i in range(3):
            seat       = opp_seats[i]
            player_rep = self._player_pool(event_tokens, player_ids, padding_mask, seat)
            opp_feat   = torch.cat([
                player_rep,
                opp_discards[:, i, :],
                opp_melds[:, i, :],
                opp_hand_sizes[:, i:i+1],
            ], dim=-1)                                              # [B, opp_mlp_in]
            logits_list.append(
                self.opp_mlp(opp_feat).view(-1, _TILE_VOCAB, 5)   # [B, 37, 5]
            )

        return torch.stack(logits_list, dim=1)  # [B, 3, 37, 5]

    # ------------------------------------------------------------------ #

    @staticmethod
    def mask_logits(
        logits: torch.Tensor,  # [B, 3, 37, 5]
        seen:   torch.Tensor,  # [B, 37]
    ) -> torch.Tensor:         # [B, 3, 37, 5] with impossible k masked to -inf
        """Zero out logits for counts that exceed available tiles."""
        full  = torch.tensor(_FULL_COUNTS, dtype=torch.float32, device=seen.device)
        max_k = (full - seen).clamp(0, 4).long()                    # [B, 37]
        k_idx = torch.arange(5, device=logits.device).view(1, 1, 1, 5)
        invalid = k_idx > max_k.view(max_k.size(0), 1, 37, 1)       # [B, 1, 37, 5]
        return logits.masked_fill(invalid, float('-inf'))

    @torch.no_grad()
    def predict_counts(
        self,
        logits: torch.Tensor,  # [B, 3, 37, 5]
        seen:   torch.Tensor,  # [B, 37]
    ) -> torch.Tensor:         # [B, 3, 37] expected count per tile per opponent
        """E[k] = sum_k k * softmax(masked_logits)[k] for each tile."""
        masked = self.mask_logits(logits, seen)
        k = torch.arange(5, dtype=torch.float32, device=logits.device)
        return (torch.softmax(masked, dim=-1) * k).sum(dim=-1)

    @torch.no_grad()
    def reconstruct_wall(
        self,
        opp_counts: torch.Tensor,  # [B, 3, 37]
        seen:       torch.Tensor,  # [B, 37]
    ) -> torch.Tensor:             # [B, 37] wall count predictions
        """
        Derive wall predictions from opponent hand predictions.

        wall[t] = max(0, (full_counts[t] - seen[t]) - sum_p(opp_counts[p][t]))

        Each opponent is predicted independently so their sum can occasionally
        exceed the available hidden tiles for a given type; the clamp handles this.
        """
        full = torch.tensor(_FULL_COUNTS, dtype=torch.float32, device=seen.device)
        hidden    = (full - seen).clamp(min=0.0)                    # [B, 37]
        opp_total = opp_counts.sum(dim=1)                           # [B, 37]
        return (hidden - opp_total).clamp(min=0.0)
