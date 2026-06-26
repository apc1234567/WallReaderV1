"""
app.py  --  Flask web GUI for WallReader opponent inference.

Run:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
from urllib.parse import urlparse, parse_qs

import requests
import torch
from flask import Flask, request, jsonify, render_template
from copy import deepcopy
from lxml import etree

from tenhou_db import (
    EVENT_TYPES, GameEvent, PlayerState, BoardSnapshot,
    TILE_NAMES, tenhou_tile_to_vocab, decode_meld,
)
from opp_model.dataset_opp import snapshot_to_opp_tensors, collate_opp
from opp_model.model_opp import OppHandModel

app = Flask(__name__)

_OPP_HIDDEN     = 39.0
_LOG_URL        = "https://tenhou.net/0/log/?{log_id}"
_DRAW_TAG_RE    = re.compile(r"^([TUVW])(\d+)$")
_DISCARD_TAG_RE = re.compile(r"^([DEFG])(\d+)$")
_DRAW_PLAYER    = {"T": 0, "U": 1, "V": 2, "W": 3}
_DISCARD_PLAYER = {"D": 0, "E": 1, "F": 2, "G": 3}
_SEAT_LABELS    = {1: "Right", 2: "Across", 3: "Left"}
_xml_cache: dict  = {}
_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_log_id(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "log" in qs:
        return qs["log"][0]
    if parsed.query:
        return parsed.query
    raise ValueError(f"Cannot extract log ID from {url!r}")


def fetch_xml(url: str):
    if url in _xml_cache:
        return _xml_cache[url]
    log_id = extract_log_id(url)
    resp = requests.get(_LOG_URL.format(log_id=log_id),
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    root = etree.fromstring(resp.content)
    _xml_cache[url] = (root, log_id)
    return root, log_id


def kyoku_to_name(kyoku_idx: int, honba: int) -> str:
    winds = ["East", "South", "West", "North"]
    wind  = winds[kyoku_idx // 4] if kyoku_idx // 4 < 4 else f"Wind{kyoku_idx // 4}"
    return f"{wind} {(kyoku_idx % 4) + 1}, Honba {honba}"


def relative_label(observer: int, seat: int) -> str:
    return _SEAT_LABELS.get((seat - observer) % 4, f"Seat {seat}")


def get_model(checkpoint: str, device):
    if checkpoint not in _model_cache:
        ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
        saved = ckpt.get("args", {})
        model = OppHandModel(
            d_model=saved.get("d_model", 256),
            nhead=saved.get("nhead", 8),
            num_layers=saved.get("num_layers", 6),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[checkpoint] = (model, ckpt.get("epoch"))
    return _model_cache[checkpoint]


def get_rounds(root) -> list:
    rounds, seen = [], set()
    for child in root:
        if child.tag == "INIT":
            seed = [int(x) for x in child.get("seed", "0,0,0,0,0,0").split(",")]
            key  = (seed[0], seed[1])
            if key not in seen:
                seen.add(key)
                rounds.append({"kyoku_idx": seed[0], "honba": seed[1],
                                "name": kyoku_to_name(seed[0], seed[1])})
    return rounds


def find_hand_element(root, kyoku_idx: int, honba: int):
    # Collect groups of elements between INIT tags, then find the matching one.
    # We must NOT append original elements into a new parent — lxml moves them
    # out of root, corrupting the cached tree for future calls.
    groups, current = [], None
    for child in list(root):
        if child.tag == "INIT":
            if current is not None:
                groups.append(current)
            current = [child]
        elif current is not None:
            current.append(child)
    if current is not None:
        groups.append(current)

    for group in groups:
        sp = [int(x) for x in group[0].get("seed", "0,0").split(",")]
        if sp[0] == kyoku_idx and sp[1] == honba:
            hr = etree.Element("hand")
            for e in group:
                hr.append(deepcopy(e))
            return hr
    return None


def compute_seen(players, observer):
    counts = [0] * 37
    for p in players:
        for t in p.discards: counts[t] += 1
        for m in p.melds:
            tiles = [m.tiles[0]] if m.type == "added_kan" else m.tiles
            for t in tiles: counts[t] += 1
    for t in players[observer].hand: counts[t] += 1
    return counts


def parse_all_snapshots(hand_root, observer: int) -> list:
    init = hand_root.find("INIT")
    if init is None:
        return []
    seed = [int(x) for x in init.get("seed", "0,0,0,0,0,0").split(",")]
    dora = tenhou_tile_to_vocab(seed[5])

    players = []
    for seat in range(4):
        hai = init.get(f"hai{seat}", "")
        if not hai:
            return []
        players.append(PlayerState(seat=seat,
                                   hand=[tenhou_tile_to_vocab(int(t)) for t in hai.split(",")]))

    wall = [4] * 34 + [1, 1, 1]
    wall[4] = wall[13] = wall[22] = 3
    for p in players:
        for t in p.hand: wall[t] -= 1
    wall[dora] -= 1

    live = 70
    pending_rinshan = False
    events: list = []
    turn = 0
    last_draw = [None, None, None, None]
    last_discard_player = None
    snapshots = []

    for elem in list(hand_root):
        tag = elem.tag

        dm = _DRAW_TAG_RE.match(tag)
        if dm:
            pidx = _DRAW_PLAYER[dm.group(1)]
            tv   = tenhou_tile_to_vocab(int(dm.group(2)))
            players[pidx].hand.append(tv)
            last_draw[pidx] = tv
            wall[tv] -= 1
            if pending_rinshan:
                pending_rinshan = False
            else:
                live -= 1
            turn += 1
            continue

        ddm = _DISCARD_TAG_RE.match(tag)
        if ddm:
            pidx = _DISCARD_PLAYER[ddm.group(1)]
            tv   = tenhou_tile_to_vocab(int(ddm.group(2)))
            is_tsumogiri = (last_draw[pidx] == tv)
            is_post_call = players[pidx].just_called
            players[pidx].just_called = False
            if tv in players[pidx].hand:
                players[pidx].hand.remove(tv)
            players[pidx].discards.append(tv)
            last_discard_player = pidx
            meld_sets  = sum(1 for m in players[pidx].melds if m.type != "added_kan")
            closed_size = 13 - 3 * meld_sets
            events.append(GameEvent(
                event_type=EVENT_TYPES["DISCARD"], player=pidx, turn=turn, tile=tv,
                meld_tiles=[], tsumogiri=is_tsumogiri, is_post_call=is_post_call,
                closed_hand_size=closed_size,
            ))
            seen = compute_seen(players, observer)
            phc  = []
            for p in players:
                cnts = [0] * 37
                for t in p.hand:
                    if 0 <= t < 37: cnts[t] += 1
                phc.append(cnts)
            snapshots.append(BoardSnapshot(
                observer_seat=observer, turn=turn, events=list(events),
                own_hand=list(players[observer].hand), seen_counts=seen,
                tiles_remaining=live + 13, true_wall_counts=list(wall),
                player_hand_counts=phc,
            ))
            continue

        if tag == "N":
            who  = int(elem.get("who", 0))
            meld = decode_meld(int(elem.get("m", 0)))
            players[who].melds.append(meld)
            to_remove = list(meld.tiles)
            if meld.type != "closed_kan":
                to_remove.remove(meld.called_tile)
            for t in to_remove:
                if t in players[who].hand: players[who].hand.remove(t)
            if meld.type in ("chi", "pon", "open_kan") and last_discard_player is not None:
                dp = players[last_discard_player].discards
                if dp and dp[-1] == meld.called_tile: dp.pop()
            if meld.type in ("open_kan", "closed_kan", "added_kan"):
                pending_rinshan = True
            players[who].just_called = True
            type_map = {"chi": "CHI", "pon": "PON", "open_kan": "OPEN_KAN",
                        "closed_kan": "CLOSED_KAN", "added_kan": "ADDED_KAN"}
            meld_sets = sum(1 for m in players[who].melds if m.type != "added_kan")
            _closed   = 13 - 3 * meld_sets + (1 if meld.type in ("chi", "pon") else 0)
            events.append(GameEvent(
                event_type=EVENT_TYPES[type_map[meld.type]], player=who, turn=turn,
                tile=meld.called_tile, meld_tiles=meld.tiles, tsumogiri=False,
                is_post_call=False, closed_hand_size=_closed,
            ))
            continue

        if tag == "REACH":
            who = int(elem.get("who", 0))
            if int(elem.get("step", 1)) == 1:
                players[who].in_riichi = True
                events.append(GameEvent(
                    event_type=EVENT_TYPES["RIICHI"], player=who, turn=turn,
                    tile=-1, meld_tiles=[], tsumogiri=False,
                    is_post_call=False, closed_hand_size=len(players[who].hand),
                ))
            continue

        if tag == "DORA":
            wall[tenhou_tile_to_vocab(int(elem.get("hai", 0)))] -= 1
            continue

        if tag in ("AGARI", "RYUUKYOKU"):
            break

    return snapshots


def _hand_sort_key(v):
    if v < 27:   return (v // 9, v % 9)
    elif v < 34: return (3, v - 27)
    else:        return (v - 34, 4.5)


def run_inference(snapshots: list, model, device, observer: int) -> list:
    opp_seats = [s for s in range(4) if s != observer]
    turns = []

    for snap in snapshots:
        item  = snapshot_to_opp_tensors(snap)
        batch = collate_opp([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            logits     = model(
                batch["event_types"], batch["tile_ids"], batch["scalars"],
                batch["player_ids"],  batch["padding_mask"],
                batch["seen"],        batch["tiles_remaining"],
                batch["opp_discards"], batch["opp_melds"], batch["opp_hand_sizes"],
                batch["turn_positions"],
                observer_seat=observer,
            )
            masked     = OppHandModel.mask_logits(logits, batch["seen"])
            probs      = torch.softmax(masked, dim=-1)[0]               # [3, 37, 5]
            opp_counts = model.predict_counts(logits, batch["seen"])[0] # [3, 37]
            seen_t     = batch["seen"][0]
            wall_pred  = model.reconstruct_wall(
                opp_counts.unsqueeze(0), seen_t.unsqueeze(0)
            )[0]  # [37]

        seen = snap.seen_counts
        N    = snap.tiles_remaining
        total_hidden = N + _OPP_HIDDEN
        base_wall = [(4 - seen[t]) * N / total_hidden for t in range(37)]

        opponents = []
        for i, seat in enumerate(opp_seats):
            true_hand = snap.player_hand_counts[seat]
            hand_size = sum(true_hand)
            expected  = opp_counts[i].tolist()
            base_hand = [(4 - seen[t]) * hand_size / total_hidden for t in range(37)]
            mae       = sum(abs(expected[t] - true_hand[t]) for t in range(37)) / 37
            base_mae  = sum(abs(base_hand[t] - true_hand[t]) for t in range(37)) / 37
            opponents.append({
                "seat":      seat,
                "label":     relative_label(observer, seat),
                "hand_size": hand_size,
                "probs":     [probs[i, t].tolist() for t in range(37)],  # [37][5]
                "expected":  [round(x, 4) for x in expected],
                "true_hand": true_hand,
                "mae":       round(mae, 4),
                "base_mae":  round(base_mae, 4),
            })

        wall_p  = wall_pred.tolist()
        wall_t  = snap.true_wall_counts
        wall_mae      = sum(abs(wall_p[t] - wall_t[t]) for t in range(37)) / 37
        base_wall_mae = sum(abs(base_wall[t] - wall_t[t]) for t in range(37)) / 37

        turns.append({
            "turn":          snap.turn,
            "live_tiles":    snap.tiles_remaining - 13,
            "own_hand":      [TILE_NAMES[t] for t in sorted(snap.own_hand, key=_hand_sort_key)],
            "seen":          seen,
            "opponents":     opponents,
            "wall_pred":     [round(x, 3) for x in wall_p],
            "wall_true":     wall_t,
            "wall_base":     [round(x, 3) for x in base_wall],
            "wall_mae":      round(wall_mae, 4),
            "wall_base_mae": round(base_wall_mae, 4),
        })

    return turns


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/rounds", methods=["POST"])
def api_rounds():
    data = request.json or {}
    try:
        root, log_id = fetch_xml(data.get("url", "").strip())
        return jsonify({"rounds": get_rounds(root), "log_id": log_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/infer", methods=["POST"])
def api_infer():
    data       = request.json or {}
    url        = data.get("url", "").strip()
    kyoku_idx  = int(data.get("kyoku_idx", 0))
    honba      = int(data.get("honba", 0))
    observer   = int(data.get("player", 0))
    checkpoint = data.get("checkpoint", "checkpoints_new_architecture/best_model.pt")

    try:
        device           = torch.device("cpu")
        model, epoch     = get_model(checkpoint, device)
        root, log_id     = fetch_xml(url)
        hand_root        = find_hand_element(root, kyoku_idx, honba)
        if hand_root is None:
            return jsonify({"error": "Round not found in log"}), 400
        snapshots = parse_all_snapshots(hand_root, observer)
        if not snapshots:
            return jsonify({"error": "No discard events found in this round"}), 400
        turns = run_inference(snapshots, model, device, observer)
        return jsonify({
            "turns":      turns,
            "log_id":     log_id,
            "kyoku_name": kyoku_to_name(kyoku_idx, honba),
            "observer":   observer,
            "epoch":      epoch,
            "n_turns":    len(turns),
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
