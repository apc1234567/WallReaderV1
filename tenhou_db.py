"""
tenhou_db.py

Utilities for reading the Houou phoenix-logs SQLite database and converting
raw Tenhou XML replays into structured game events suitable for the wall
prediction model.

Database schema (es4p.db):
    logs(log_id TEXT, year INTEGER, log_content BLOB)
    log_content is zlib-compressed XML.

Tile encoding (standard Tenhou integers):
    0-35:   1m-9m (×4 each, indices 0,4,8,...,32 = 1m...9m)
    36-71:  1p-9p
    72-107: 1s-9s
    108-135: honors (East/South/West/North/Haku/Hatsu/Chun)
    16, 52, 88: red 5m, red 5p, red 5s (one of the four 5-tile slots)

We normalize to a 37-tile vocabulary:
    0-8:   1m-9m
    9-17:  1p-9p
    18-26: 1s-9s
    27-33: East/South/West/North/Haku/Hatsu/Chun
    34:    red 5m
    35:    red 5p
    36:    red 5s
"""

import bz2
import sqlite3
import zlib
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple
from lxml import etree

# ---------------------------------------------------------------------------
# Tile encoding helpers
# ---------------------------------------------------------------------------

# Tenhou tile integer → (suit_index, value_1indexed)
# Suits: 0=man, 1=pin, 2=sou, 3=honors
# Red fives have Tenhou IDs 16, 52, 88 (the third copy of 5 in each suit)
RED_FIVE_IDS = {16, 52, 88}

def tenhou_tile_to_vocab(tile_id: int) -> int:
    """
    Convert a raw Tenhou tile integer to our 37-tile vocabulary index.

    Returns an integer in [0, 36]:
        0-8   = 1m-9m
        9-17  = 1p-9p
        18-26 = 1s-9s
        27-33 = East/South/West/North/Haku/Hatsu/Chun
        34    = red 5m
        35    = red 5p
        36    = red 5s
    """
    if tile_id in RED_FIVE_IDS:
        suit = tile_id // 36  # 0, 1, or 2
        return 34 + suit

    suit = tile_id // 36
    value = (tile_id % 36) // 4  # 0-indexed (0=1, 8=9)

    if suit < 3:
        return suit * 9 + value
    else:
        # honors: tile_id 108-135, value 0-6
        return 27 + value


def vocab_to_suit_value(vocab_id: int) -> Tuple[int, int]:
    """Return (suit, value_1indexed) for display. Suit: 0=m,1=p,2=s,3=z."""
    if vocab_id < 27:
        return vocab_id // 9, (vocab_id % 9) + 1
    elif vocab_id < 34:
        return 3, vocab_id - 27 + 1
    else:
        # red fives
        suit = vocab_id - 34
        return suit, 5


TILE_NAMES = (
    [f"{v}m" for v in range(1, 10)] +
    [f"{v}p" for v in range(1, 10)] +
    [f"{v}s" for v in range(1, 10)] +
    ["East", "South", "West", "North", "Haku", "Hatsu", "Chun"] +
    ["r5m", "r5p", "r5s"]
)

# ---------------------------------------------------------------------------
# Meld decoding
# ---------------------------------------------------------------------------

@dataclass
class Meld:
    type: str          # "chi", "pon", "open_kan", "closed_kan", "added_kan"
    tiles: List[int]   # vocab tile IDs of tiles in the meld
    called_tile: int   # vocab tile ID of the tile that was called (for open melds)
    called_from: int   # seat offset of the player called from (0=self for closed kan)


def decode_meld(meld_int: int) -> Meld:
    """
    Decode a Tenhou meld integer (the 'm' attribute of <N> tags).

    Bit layout documented at:
    https://github.com/ApplySci/tenhou-log#meld-format
    """
    if meld_int & 0x4:
        return _decode_chi(meld_int)
    elif meld_int & 0x18:
        return _decode_pon_or_added_kan(meld_int)
    else:
        return _decode_kan(meld_int)


def _decode_chi(m: int) -> Meld:
    who = m & 0x3
    t0 = (m >> 3) & 0x3
    t1 = (m >> 5) & 0x3
    t2 = (m >> 7) & 0x3
    base_and_called = (m >> 10)
    base = (base_and_called // 3)
    called = base_and_called % 3
    base_tile = (base // 7) * 9 * 4 + (base % 7) * 4

    tiles_raw = [
        base_tile + 4 * 0 + t0,
        base_tile + 4 * 1 + t1,
        base_tile + 4 * 2 + t2,
    ]
    tiles_vocab = [tenhou_tile_to_vocab(t) for t in tiles_raw]
    called_tile_vocab = tiles_vocab[called]

    return Meld(
        type="chi",
        tiles=tiles_vocab,
        called_tile=called_tile_vocab,
        called_from=who,
    )


def _decode_pon_or_added_kan(m: int) -> Meld:
    who = m & 0x3
    t4 = (m >> 5) & 0x3
    is_added_kan = bool(m & 0x10)
    base_and_called = m >> 9
    base = base_and_called // 3
    called = base_and_called % 3
    base_tile = base * 4

    tiles_raw = [base_tile + i for i in range(4) if i != t4]
    tiles_vocab = [tenhou_tile_to_vocab(t) for t in tiles_raw]
    called_tile_vocab = tiles_vocab[called]

    return Meld(
        type="added_kan" if is_added_kan else "pon",
        tiles=tiles_vocab,
        called_tile=called_tile_vocab,
        called_from=who,
    )


def _decode_kan(m: int) -> Meld:
    who = m & 0x3
    base_and_called = m >> 8
    base = base_and_called // 4
    called = base_and_called % 4  # which of the 4 copies was called / drawn
    base_tile = base * 4

    tiles_vocab = [tenhou_tile_to_vocab(base_tile + i) for i in range(4)]
    is_closed = (who == 0)

    return Meld(
        type="closed_kan" if is_closed else "open_kan",
        tiles=tiles_vocab,
        called_tile=tiles_vocab[called],
        called_from=who,
    )

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_TYPES = {
    "DISCARD": 0,
    "CHI": 1,
    "PON": 2,
    "OPEN_KAN": 3,
    "CLOSED_KAN": 4,
    "ADDED_KAN": 5,
    "RIICHI": 6,
}

@dataclass
class GameEvent:
    event_type: int          # one of EVENT_TYPES values
    player: int              # seat 0-3
    turn: int                # global turn counter within this hand
    tile: int                # primary tile vocab ID (discard tile, or called tile for melds)
    meld_tiles: List[int]    # all 3 (chi/pon) or 4 (kan) vocab tile IDs; empty for discard
    tsumogiri: bool          # True if drawn-and-discarded immediately
    is_post_call: bool       # True if this discard follows a call by the same player
    closed_hand_size: int    # tiles remaining in player's closed hand after this event


# ---------------------------------------------------------------------------
# Hand state tracker
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    seat: int
    hand: List[int] = field(default_factory=list)   # vocab tile IDs in closed hand
    discards: List[int] = field(default_factory=list)
    melds: List[Meld] = field(default_factory=list)
    in_riichi: bool = False
    just_called: bool = False  # flag: next discard is post-call


@dataclass
class BoardSnapshot:
    """
    Complete observable board state at a given moment, from the perspective
    of `observer_seat`. Used as a training sample.
    """
    observer_seat: int
    turn: int
    events: List[GameEvent]                  # all events so far this hand
    own_hand: List[int]                      # observer's closed hand (vocab IDs)
    seen_counts: List[int]                   # [37] counts of visible tiles
    tiles_remaining: int                     # tiles left in total wall (live + dead)
    true_wall_counts: List[int]              # [37] ground truth for training
    player_hand_counts: List[List[int]] = None  # [4][37] each player's closed hand counts


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

# Draw tags: T=player0, U=player1, V=player2, W=player3
# Discard tags: D=player0, E=player1, F=player2, G=player3
_DRAW_TAG_RE = re.compile(r'^([TUVW])(\d+)$')
_DISCARD_TAG_RE = re.compile(r'^([DEFG])(\d+)$')
_DRAW_PLAYER = {'T': 0, 'U': 1, 'V': 2, 'W': 3}
_DISCARD_PLAYER = {'D': 0, 'E': 1, 'F': 2, 'G': 3}


def parse_hand_xml(hand_xml: etree._Element) -> List[BoardSnapshot]:
    """
    Parse a single hand (round) XML element and yield BoardSnapshot objects,
    one per discard event (the most informative moments for wall prediction).

    Returns an empty list if the hand XML is malformed.
    """
    snapshots = []

    init = hand_xml.find('INIT')
    if init is None:
        return snapshots

    seed_str = init.get('seed', '0,0,0,0,0,0')
    seed_parts = [int(x) for x in seed_str.split(',')]
    dora_indicator_raw = seed_parts[5]
    dora_indicator = tenhou_tile_to_vocab(dora_indicator_raw)

    # Parse starting hands
    players = []
    for seat in range(4):
        hai_str = init.get(f'hai{seat}', '')
        if not hai_str:
            return snapshots  # sanma or malformed; skip
        hand_raw = [int(x) for x in hai_str.split(',')]
        hand_vocab = [tenhou_tile_to_vocab(t) for t in hand_raw]
        players.append(PlayerState(seat=seat, hand=hand_vocab))

    # Full tile pool: 136 tiles total.
    # Red fives (vocab 34/35/36) replace one normal copy of 5m/5p/5s,
    # so vocab 4/13/22 (normal 5-tiles) have only 3 non-red copies.
    wall_counts = [4] * 34 + [1, 1, 1]  # vocab counts, 37 total
    wall_counts[4] = 3   # 5m: 3 normal copies (red 5m is vocab 34)
    wall_counts[13] = 3  # 5p: 3 normal copies (red 5p is vocab 35)
    wall_counts[22] = 3  # 5s: 3 normal copies (red 5s is vocab 36)

    # Remove dealt tiles from wall
    for p in players:
        for t in p.hand:
            wall_counts[t] -= 1
    # Remove dora indicator (from dead wall — does not affect live wall count)
    wall_counts[dora_indicator] -= 1

    # Live wall starts at 70 in standard 4-player Tenhou (136 - 52 dealt - 14 dead).
    # Only live wall draws decrement this; rinshan draws (after kan) and dora reveals
    # come from the dead wall and leave the live count unchanged.
    live_tiles_remaining = 70
    pending_rinshan = False  # set True after any kan declaration

    events_so_far: List[GameEvent] = []
    turn_counter = 0
    last_draw: List[Optional[int]] = [None, None, None, None]  # last drawn tile per player
    last_discard_player: Optional[int] = None  # player who made the most recent discard

    def seen_counts_snapshot() -> List[int]:
        # From observer seat 0's perspective (in training we'll rotate).
        # Seen = own hand + all discards + all melds.
        counts = [0] * 37
        for p in players:
            for t in p.discards:
                counts[t] += 1
            for meld in p.melds:
                if meld.type == "added_kan":
                    # Pon tiles already counted via the original pon meld entry.
                    # Only the 1 newly added tile is new information.
                    counts[meld.tiles[0]] += 1
                else:
                    for t in meld.tiles:
                        counts[t] += 1
        # Own hand
        for t in players[0].hand:
            counts[t] += 1
        return counts

    for elem in hand_xml:
        tag = elem.tag

        draw_match = _DRAW_TAG_RE.match(tag)
        if draw_match:
            player_idx = _DRAW_PLAYER[draw_match.group(1)]
            tile_raw = int(draw_match.group(2))
            tile_vocab = tenhou_tile_to_vocab(tile_raw)
            players[player_idx].hand.append(tile_vocab)
            last_draw[player_idx] = tile_vocab
            wall_counts[tile_vocab] -= 1
            if pending_rinshan:
                pending_rinshan = False  # rinshan draw from dead wall
            else:
                live_tiles_remaining -= 1
            turn_counter += 1
            continue

        discard_match = _DISCARD_TAG_RE.match(tag)
        if discard_match:
            player_idx = _DISCARD_PLAYER[discard_match.group(1)]
            tile_raw = int(discard_match.group(2))
            tile_vocab = tenhou_tile_to_vocab(tile_raw)

            is_tsumogiri = (last_draw[player_idx] == tile_vocab)
            is_post_call = players[player_idx].just_called
            players[player_idx].just_called = False

            # Remove from player hand, add to discards
            if tile_vocab in players[player_idx].hand:
                players[player_idx].hand.remove(tile_vocab)
            players[player_idx].discards.append(tile_vocab)
            last_discard_player = player_idx

            meld_sets = sum(1 for m in players[player_idx].melds if m.type != "added_kan")
            closed_size = 13 - 3 * meld_sets

            event = GameEvent(
                event_type=EVENT_TYPES["DISCARD"],
                player=player_idx,
                turn=turn_counter,
                tile=tile_vocab,
                meld_tiles=[],
                tsumogiri=is_tsumogiri,
                is_post_call=is_post_call,
                closed_hand_size=closed_size,
            )
            events_so_far.append(event)

            # Take a snapshot after each discard (from player 0's perspective)
            phc = []
            for p in players:
                counts = [0] * 37
                for t in p.hand:
                    if 0 <= t < 37:
                        counts[t] += 1
                phc.append(counts)
            snapshot = BoardSnapshot(
                observer_seat=0,
                turn=turn_counter,
                events=list(events_so_far),
                own_hand=list(players[0].hand),
                seen_counts=seen_counts_snapshot(),
                tiles_remaining=live_tiles_remaining + 13,
                true_wall_counts=list(wall_counts),
                player_hand_counts=phc,
            )
            snapshots.append(snapshot)
            continue

        if tag == 'N':
            who = int(elem.get('who', 0))
            meld_int = int(elem.get('m', 0))
            meld = decode_meld(meld_int)

            # Add meld tiles to player's meld list
            players[who].melds.append(meld)

            # Remove tiles the player contributed from their closed hand.
            # called_tile came from the discard pile (external) for all types
            # except closed_kan, where every tile was already in hand.
            tiles_to_remove = list(meld.tiles)
            if meld.type != "closed_kan":
                tiles_to_remove.remove(meld.called_tile)
            for t in tiles_to_remove:
                if t in players[who].hand:
                    players[who].hand.remove(t)

            # For chi/pon/open_kan the called tile came from the last discard.
            # Remove it from that player's discard pile so seen_counts doesn't
            # double-count it (once in discards, once in melds).
            if meld.type in ("chi", "pon", "open_kan") and last_discard_player is not None:
                dp = players[last_discard_player].discards
                if dp and dp[-1] == meld.called_tile:
                    dp.pop()

            if meld.type in ("open_kan", "closed_kan", "added_kan"):
                pending_rinshan = True
            players[who].just_called = True

            type_map = {
                "chi": "CHI", "pon": "PON",
                "open_kan": "OPEN_KAN",
                "closed_kan": "CLOSED_KAN",
                "added_kan": "ADDED_KAN",
            }
            etype = EVENT_TYPES[type_map[meld.type]]
            meld_sets = sum(1 for m in players[who].melds if m.type != "added_kan")
            if meld.type in ("chi", "pon"):
                closed_size = 13 - 3 * meld_sets + 1  # 1 tile still to discard
            else:
                closed_size = 13 - 3 * meld_sets

            event = GameEvent(
                event_type=etype,
                player=who,
                turn=turn_counter,
                tile=meld.called_tile,
                meld_tiles=meld.tiles,
                tsumogiri=False,
                is_post_call=False,
                closed_hand_size=closed_size,
            )
            events_so_far.append(event)
            continue

        if tag == 'REACH':
            who = int(elem.get('who', 0))
            step = int(elem.get('step', 1))
            if step == 1:
                players[who].in_riichi = True
                event = GameEvent(
                    event_type=EVENT_TYPES["RIICHI"],
                    player=who,
                    turn=turn_counter,
                    tile=-1,
                    meld_tiles=[],
                    tsumogiri=False,
                    is_post_call=False,
                    closed_hand_size=len(players[who].hand),
                )
                events_so_far.append(event)
            continue

        if tag == 'DORA':
            # New dora indicator revealed (after a kan) — from dead wall, no live change
            hai_raw = int(elem.get('hai', 0))
            new_dora = tenhou_tile_to_vocab(hai_raw)
            wall_counts[new_dora] -= 1
            continue

        # AGARI or RYUUKYOKU — hand over, stop collecting
        if tag in ('AGARI', 'RYUUKYOKU'):
            break

    return snapshots


# ---------------------------------------------------------------------------
# Database interface
# ---------------------------------------------------------------------------

def iter_games(
    db_path: str,
    limit: Optional[int] = None,
    offset: int = 0,
    year: Optional[int] = None,
) -> Iterator[Tuple[str, etree._Element]]:
    """
    Yield (log_id, game_root_element) for games in the database.

    Args:
        db_path:  Path to the merged SQLite database (e.g. 'data/es4p.db').
        limit:    Maximum number of games to yield (None = all).
        offset:   Skip this many rows first (for resuming).
        year:     If given, only yield games from this year.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where_clauses = []
    params: list = []
    if year is not None:
        where_clauses.append("year = ?")
        params.append(year)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    limit_sql = f"LIMIT {limit}" if limit is not None else ""
    offset_sql = f"OFFSET {offset}" if offset > 0 else ""

    sql = f"SELECT log_id, log_content FROM logs {where_sql} {limit_sql} {offset_sql}"

    try:
        for row in conn.execute(sql, params):
            log_id = row['log_id']
            raw = row['log_content']
            try:
                xml_bytes = bz2.decompress(raw)
            except OSError:
                try:
                    xml_bytes = zlib.decompress(raw)
                except zlib.error:
                    continue
            try:
                root = etree.fromstring(xml_bytes)
                yield log_id, root
            except etree.XMLSyntaxError:
                continue
    finally:
        conn.close()


def iter_snapshots(
    db_path: str,
    limit: Optional[int] = None,
    offset: int = 0,
    year: Optional[int] = None,
    snapshots_per_hand: int = 4,
    seed: int = 42,
) -> Iterator[BoardSnapshot]:
    """
    Yield BoardSnapshot objects for model training.

    Samples `snapshots_per_hand` turns per hand, stratified across
    early/mid/late thirds to avoid overrepresenting late-game states.

    Args:
        db_path:            Path to es4p.db.
        limit:              Max games to read.
        offset:             Row offset.
        year:               Filter by year.
        snapshots_per_hand: How many snapshots to sample per hand.
        seed:               RNG seed for reproducible sampling.
    """
    import random
    rng = random.Random(seed)

    for log_id, root in iter_games(db_path, limit=limit, offset=offset, year=year):
        # A game XML contains multiple hands (rounds)
        for hand_elem in root.iter('mjloggm'):
            # Iterate <INIT>-delimited hand blocks
            # The root itself is the game; hands are separated by INIT tags.
            pass

        # The root element *is* the game log; hands are sequential INIT blocks.
        # Collect all INIT elements and slice the XML between them.
        all_snapshots = _parse_game(root)

        for snap_list in all_snapshots:
            if not snap_list:
                continue
            n = len(snap_list)
            # Stratified sampling: pick from early/mid/late thirds
            thirds = max(1, n // 3)
            buckets = [
                snap_list[:thirds],
                snap_list[thirds:2*thirds],
                snap_list[2*thirds:],
            ]
            chosen = []
            per_bucket = max(1, snapshots_per_hand // 3)
            for bucket in buckets:
                if bucket:
                    chosen.extend(rng.sample(bucket, min(per_bucket, len(bucket))))
            # Fill remainder from any bucket
            remaining = snapshots_per_hand - len(chosen)
            if remaining > 0 and n > len(chosen):
                pool = [s for s in snap_list if s not in chosen]
                chosen.extend(rng.sample(pool, min(remaining, len(pool))))

            for snap in chosen:
                yield snap


def _parse_game(root: etree._Element) -> List[List[BoardSnapshot]]:
    """
    Parse a full game XML element into per-hand snapshot lists.

    Tenhou game XML looks like:
        <mjloggm ver="2.3">
          <SHUFFLE ... />
          <GO ... />
          <UN ... />
          <TAIKYOKU ... />
          <INIT seed="..." ten="..." oya="..." hai0="..." ... />
          <T12/><D15/>... (draws/discards/calls for hand 1)
          <AGARI .../>
          <INIT .../>   (next hand)
          ...
        </mjloggm>

    We split the children into hand blocks at each INIT tag.
    """
    results = []
    current_hand_elems: List[etree._Element] = []
    in_hand = False

    for child in list(root):  # snapshot children list before any tree mutations below
        if child.tag == 'INIT':
            if in_hand and current_hand_elems:
                # Build a fake parent for the previous hand
                hand_root = etree.Element("hand")
                hand_root.append(current_hand_elems[0])  # INIT
                for e in current_hand_elems[1:]:
                    hand_root.append(e)
                results.append(parse_hand_xml(hand_root))
            current_hand_elems = [child]
            in_hand = True
        elif in_hand:
            current_hand_elems.append(child)

    # Last hand
    if in_hand and current_hand_elems:
        hand_root = etree.Element("hand")
        for e in current_hand_elems:
            hand_root.append(e)
        results.append(parse_hand_xml(hand_root))

    return results


# ---------------------------------------------------------------------------
# Train/val/test split by game
# ---------------------------------------------------------------------------

def get_game_ids(db_path: str, year: Optional[int] = None) -> List[str]:
    """Return all log_ids in the database (for game-level splitting)."""
    conn = sqlite3.connect(db_path)
    where = f"WHERE year = {year}" if year else ""
    rows = conn.execute(f"SELECT log_id FROM logs {where}").fetchall()
    conn.close()
    return [r[0] for r in rows]


def split_game_ids(
    game_ids: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Shuffle and split game IDs into train/val/test sets.
    IMPORTANT: always split at the game level, never at the turn level,
    to prevent data leakage from correlated turns within the same game.
    """
    import random
    rng = random.Random(seed)
    ids = list(game_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os

    db = sys.argv[1] if len(sys.argv) > 1 else 'data/es4p.db'
    if not os.path.exists(db):
        print(f"Database not found: {db}")
        sys.exit(1)

    print("Reading first 100 games...")
    count = 0
    snap_count = 0
    for snap in iter_snapshots(db, limit=100, snapshots_per_hand=4):
        snap_count += 1
        count += 1

    print(f"Games processed (sample): 100")
    print(f"Total snapshots yielded: {snap_count}")
    print(f"\nExample snapshot:")
    for snap in iter_snapshots(db, limit=1, snapshots_per_hand=1):
        print(f"  Turn: {snap.turn}")
        print(f"  Tiles remaining: {snap.tiles_remaining}")
        print(f"  Own hand: {[TILE_NAMES[t] for t in snap.own_hand]}")
        print(f"  Events so far: {len(snap.events)}")
        print(f"  Wall counts (nonzero): {[(TILE_NAMES[i], c) for i, c in enumerate(snap.true_wall_counts) if c > 0][:10]}...")
        break
