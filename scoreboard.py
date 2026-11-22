#!/usr/bin/env python3
"""Build the net-of-control scoreboard from results/all_results.json (single source of truth for the book)."""
import json, os
R = json.load(open(os.path.join(os.path.dirname(__file__), "results", "all_results.json")))


def pct(new, old): return 100.0 * (new - old) / old


def rows():
    out = []
    e = R["e06_euclid"]["b_row_translation"]
    out.append(("E06 above/left lattice context", "bpb", e["base_plus_mixer_only_bpb"], e["plus_above_left_bpb"], e["plus_noise_control_bpb"]))
    e = R["e08_topology"]["a_planar_markup"]
    out.append(("E08 planar stack context", "bpb", e["mixer_only_bpb"], e["plus_stack_bpb"], e["plus_noise_control_bpb"]))
    e = R["e10_harmonic"]["a_period_character"]
    out.append(("E10 period character (fixed-width data)", "bpb", e["fixed_width_mixer_only_bpb"], e["fixed_width_plus_period_bpb"], e["fixed_width_plus_noise_control_bpb"]))
    e = R["e12_physics"]["b_renormalisation"]
    out.append(("E12 word-scale (RG) context", "bpb", e["mixer_only_bpb"], e["plus_word_scale_bpb"], e["plus_noise_control_bpb"]))
    out.append(("E12 character-class block context", "bpb", e["mixer_only_bpb"], e["plus_class_block_scale_bpb"], e["plus_noise_control_bpb"]))
    out.append(("E12 both scales", "bpb", e["mixer_only_bpb"], e["plus_both_scales_bpb"], e["plus_noise_control_bpb"]))
    return out


def md():
    lines = ["| Approach | Baseline | Candidate | Noise control | Candidate vs baseline | Control vs baseline | Net of control |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, unit, base, cand, ctl in rows():
        lines.append("| %s | %.4f | %.4f | %.4f | %+.1f%% | %+.1f%% | %+.1f%% |" % (
            name, base, cand, ctl, pct(cand, base), pct(ctl, base), pct(cand, base) - pct(ctl, base)))
    return "\n".join(lines)


if __name__ == "__main__":
    t = md(); print(t)
    open(os.path.join(os.path.dirname(__file__), "results", "SCOREBOARD.md"), "w").write(t + "\n")
