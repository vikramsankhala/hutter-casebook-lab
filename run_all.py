#!/usr/bin/env python3
"""Run every experiment, write results/*.json and results/RESULTS.md.

    python run_all.py                 # synthetic River Archive II (default)
    HCL_ENWIK=/path/to/enwik8 python run_all.py   # same experiments on a real slice (first 128 KiB by default)
"""
import importlib, json, os, sys, time

EXPS = ["e01_baselines", "e02_multisymbol", "e03_copy_edits", "e04_grammar", "e05_energy", "e06_euclid",
        "e07_infogeom", "e08_topology", "e09_quantum_topology", "e10_harmonic", "e11_lens_functor", "e12_physics"]


def main(only=None):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    summary, t0 = {}, time.time()
    for name in EXPS:
        if only and name not in only:
            continue
        t = time.time()
        mod = importlib.import_module("experiments." + name)
        summary[name] = mod.run()
        print("%-22s %5.1fs" % (name, time.time() - t), flush=True)
    out = os.path.join(here, "results")
    with open(os.path.join(out, "all_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("total %.1fs -> results/all_results.json" % (time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
