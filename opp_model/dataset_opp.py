"""
dataset_opp.py

Dataset for the opponent hand prediction model.

For each snapshot the target is the closed hand composition of each of the
3 non-observer players, represented as a [3, 37] float tensor where each
row is a probability distribution over tile types (counts / hand_size).

Per-opponent observable features (discard counts, meld tile counts, hand size)
are derived from the event stream and stored alongside the standard event
sequence features reused from the wall model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

from tenhou_db import BoardSnapshot, EVENT_TYPES

_DISCARD   = EVENT_TYPES["DISCARD"]
_CHI       = EVENT_TYPES["CHI"]
_PON       = EVENT_TYPES["PON"]
_OPEN_KAN  = EVENT_TYPES["OPEN_KAN"]
_CLOSED_KAN = EVENT_TYPES["CLOSED_KAN"]
_ADDED_KAN = EVENT_TYPES["ADDED_KAN"]
_MELD_TYPES = {_CHI, _PON, _OPEN_KAN, _CLOSED_KAN, _ADDED_KAN}


def snapshot_to_opp_tensors(snap: BoardSnapshot) -> Dict[str, Any]:
    """
    Convert a BoardSnapshot to tensors for opponent hand prediction.

    Returns the standard event-sequence features (identical to dataset.py)
    plus opponent-specific features and targets.

    Opponent ordering: the 3 seats that are not the observer, in ascending
    seat order.  With observer=0 this is always [1, 2, 3].
    """
    assert snap.player_hand_counts is not None, \
        "BoardSnapshot.player_hand_counts must be populated"

    observer = snap.observer_seat
    opp_seats = [s for s in range(4) if s != observer]  # 3 opponents in seat order

    # ------------------------------------------------------------------
    # Standard event-sequence features (same layout as dataset.py)
    # ------------------------------------------------------------------
    events = snap.events
    S = len(events)

    if S == 0:
        event_types     = torch.zeros(0, dtype=torch.long)
        tile_ids_padded = torch.zeros((0, 4), dtype=torch.long)
        scalars         = torch.zeros((0, 3), dtype=torch.float32)
        player_ids      = torch.zeros(0, dtype=torch.long)
        turn_positions  = torch.zeros(0, dtype=torch.long)
    else:
        event_types     = torch.zeros(S, dtype=torch.long)
        tile_ids_padded = torch.full((S, 4), 37, dtype=torch.long)
        tsumogiri       = torch.zeros(S, dtype=torch.float32)
        is_post_call    = torch.zeros(S, dtype=torch.float32)
        closed_sizes    = torch.zeros(S, dtype=torch.float32)
        player_ids      = torch.zeros(S, dtype=torch.long)
        turn_positions  = torch.zeros(S, dtype=torch.long)

        for i, ev in enumerate(events):
            event_types[i]    = ev.event_type
            player_ids[i]     = ev.player
            turn_positions[i] = ev.turn
            tsumogiri[i]      = float(ev.tsumogiri)
            is_post_call[i]   = float(ev.is_post_call)
            closed_sizes[i]   = ev.closed_hand_size / 13.0
            if ev.meld_tiles:
                for j, t in enumerate(ev.meld_tiles[:4]):
                    if 0 <= t < 37:
                        tile_ids_padded[i, j] = t
            elif ev.tile >= 0:
                tile_ids_padded[i, 0] = ev.tile

        scalars = torch.stack([tsumogiri, is_post_call, closed_sizes], dim=1)

    # ------------------------------------------------------------------
    # Global features (observer's hand, seen counts, tiles_remaining)
    # ------------------------------------------------------------------
    own_hand = torch.zeros(37, dtype=torch.float32)
    for t in snap.own_hand:
        if 0 <= t < 37:
            own_hand[t] += 1.0

    seen           = torch.tensor(snap.seen_counts, dtype=torch.float32)
    tiles_remaining = float(snap.tiles_remaining)

    # ------------------------------------------------------------------
    # Per-opponent observable features derived from the event stream
    # ------------------------------------------------------------------
    # Accumulate per-player discard counts and meld tile counts from events.
    player_discards = [[0] * 37 for _ in range(4)]
    player_melds    = [[0] * 37 for _ in range(4)]

    for ev in events:
        if ev.event_type == _DISCARD:
            if 0 <= ev.tile < 37:
                player_discards[ev.player][ev.tile] += 1
        elif ev.event_type in _MELD_TYPES:
            for t in ev.meld_tiles:
                if 0 <= t < 37:
                    player_melds[ev.player][t] += 1

    opp_discards   = torch.zeros(3, 37, dtype=torch.float32)
    opp_melds      = torch.zeros(3, 37, dtype=torch.float32)
    opp_hand_sizes = torch.zeros(3, dtype=torch.float32)

    for i, seat in enumerate(opp_seats):
        opp_discards[i]   = torch.tensor(player_discards[seat], dtype=torch.float32)
        opp_melds[i]      = torch.tensor(player_melds[seat],    dtype=torch.float32)
        hand_size         = sum(snap.player_hand_counts[seat])
        opp_hand_sizes[i] = hand_size / 13.0

    # ------------------------------------------------------------------
    # Targets: each opponent's actual closed hand as a distribution [37]
    # (counts / hand_size, or zero vector if hand is empty)
    # ------------------------------------------------------------------
    opp_targets     = torch.zeros(3, 37, dtype=torch.float32)
    opp_hand_counts = torch.zeros(3, 37, dtype=torch.float32)  # raw counts for MAE

    for i, seat in enumerate(opp_seats):
        counts    = torch.tensor(snap.player_hand_counts[seat], dtype=torch.float32)
        hand_size = counts.sum().item()
        opp_hand_counts[i] = counts
        if hand_size > 0:
            opp_targets[i] = counts / hand_size

    return {
        # Event sequence (identical layout to dataset.py)
        "event_types":     event_types,       # [S] long
        "tile_ids_padded": tile_ids_padded,   # [S, 4] long
        "scalars":         scalars,           # [S, 3] float
        "player_ids":      player_ids,        # [S] long
        "turn_positions":  turn_positions,    # [S] long
        # Global features
        "own_hand":        own_hand,          # [37] float
        "seen":            seen,              # [37] float
        "tiles_remaining": torch.tensor(tiles_remaining, dtype=torch.float32),
        # Per-opponent features (3 opponents, shared-weight head)
        "opp_discards":    opp_discards,      # [3, 37] float
        "opp_melds":       opp_melds,         # [3, 37] float
        "opp_hand_sizes":  opp_hand_sizes,    # [3] float, normalised by 13
        # Targets
        "opp_targets":     opp_targets,       # [3, 37] float, distribution per opponent
        "opp_hand_counts": opp_hand_counts,   # [3, 37] float, raw counts for MAE
    }


class OppDataset(Dataset):
    def __init__(self, snapshots: List[BoardSnapshot]):
        self.snapshots = snapshots

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        return snapshot_to_opp_tensors(self.snapshots[idx])


def collate_opp(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_seq = max(item["event_types"].size(0) for item in batch)
    max_seq = max(max_seq, 1)
    B = len(batch)

    event_types    = torch.zeros(B, max_seq, dtype=torch.long)
    tile_ids       = torch.full((B, max_seq, 4), 37, dtype=torch.long)
    scalars        = torch.zeros(B, max_seq, 3, dtype=torch.float32)
    player_ids     = torch.zeros(B, max_seq, dtype=torch.long)
    turn_positions = torch.zeros(B, max_seq, dtype=torch.long)
    padding_mask   = torch.ones(B, max_seq, dtype=torch.bool)

    for i, item in enumerate(batch):
        S = item["event_types"].size(0)
        if S > 0:
            event_types[i,    :S] = item["event_types"]
            tile_ids[i,       :S] = item["tile_ids_padded"]
            scalars[i,        :S] = item["scalars"]
            player_ids[i,     :S] = item["player_ids"]
            turn_positions[i, :S] = item["turn_positions"]
            padding_mask[i,   :S] = False

    return {
        "event_types":     event_types,
        "tile_ids":        tile_ids,
        "scalars":         scalars,
        "player_ids":      player_ids,
        "turn_positions":  turn_positions,
        "padding_mask":    padding_mask,
        "own_hand":        torch.stack([x["own_hand"]        for x in batch]),
        "seen":            torch.stack([x["seen"]            for x in batch]),
        "tiles_remaining": torch.stack([x["tiles_remaining"] for x in batch]),
        "opp_discards":    torch.stack([x["opp_discards"]    for x in batch]),
        "opp_melds":       torch.stack([x["opp_melds"]       for x in batch]),
        "opp_hand_sizes":  torch.stack([x["opp_hand_sizes"]  for x in batch]),
        "opp_targets":     torch.stack([x["opp_targets"]     for x in batch]),
        "opp_hand_counts": torch.stack([x["opp_hand_counts"] for x in batch]),
    }
