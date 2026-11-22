"""E04 - Recursive grammars and description length (Case 8).

Re-Pair builds a straight-line grammar. Description length (bits):
  final sequence: n_seq * entropy(final sequence)   [order-0, empirical]
  dictionary:     2 * log2(256 + R) bits per rule    [uniform, generous to the grammar]
"""
import math
from collections import Counter
from experiments.common import get_data, save, order0_bits

N = 60_000


def repair(data, max_rules):
    seq = list(data); nxt = 256; rules = []
    snaps = {}
    for r in range(max_rules + 1):
        snaps[r] = seq
        if r == max_rules: break
        pairs = Counter(zip(seq, seq[1:]))
        if not pairs: break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 2: break
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                out.append(nxt); i += 2
            else:
                out.append(seq[i]); i += 1
        rules.append((a, b)); seq = out; nxt += 1
    return rules, snaps


def dl_bits(seq, R):
    c = Counter(seq); n = len(seq)
    body = -sum(v * math.log2(v / n) for v in c.values())
    return body + 2 * R * math.log2(256 + R)


def expand(sym, rules):
    st, out = [sym], bytearray()
    while st:
        s = st.pop()
        if s < 256: out.append(s)
        else:
            a, b = rules[s - 256]; st.append(b); st.append(a)
    return bytes(out)


def run():
    d = get_data(N)
    rules, snaps = repair(d, 2500)
    curve = {}
    for R in sorted(snaps):
        if R in (0, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500) or R == len(rules):
            curve[R] = round(dl_bits(snaps[R], R) / 8)
    best_R = min(snaps, key=lambda R: dl_bits(snaps[R], R))
    # exact reconstruction check
    final = snaps[best_R]
    assert b"".join(expand(s, rules) for s in final) == d
    # recursive fixture: a repeated phrase, grammar vs bytes
    ph = b"{{cite web |url=http://example.org/2026 |title=Annual Report}}\n" * 128
    r2, s2 = repair(ph, 80)
    fx = {R: round(dl_bits(s2[R], R) / 8) for R in s2 if R in (0, 4, 8, 16, 40, 80) and R in s2}
    return save("e04_grammar", dict(
        bytes=len(d), order0_bytes=round(order0_bits(d) / 8), rules_available=len(rules),
        description_length_bytes_by_rules=curve, best_rule_count=best_R,
        best_bytes=round(dl_bits(snaps[best_R], best_R) / 8),
        recursive_rules_in_best=sum(1 for a, b in rules[:best_R] if a >= 256 or b >= 256),
        fixture_repeated_phrase=dict(bytes=len(ph), description_length_bytes_by_rules=fx)))


if __name__ == "__main__":
    print(run())
