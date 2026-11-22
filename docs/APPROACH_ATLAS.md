# Approach Atlas

One line per approach: the assumption it makes about the data, the measured result on the synthetic River Archive II, and the verdict. Numbers come from `results/all_results.json`. Bits per byte unless stated.

| Approach | Assumes | Result | Verdict |
|:--|:--|:--|:--|
| Multi-symbol coding | Byte decisions can be one step | 256-ary 2.1174; binary 2.0523; Huffman tree 2.0813 | Binary wins here; cause untested |
| Copy-with-edits | Near-duplicates with small changes | Optimal parse −8.1% vs greedy; edits −1.7% | Parsing robust; edit source is digits in templates |
| Recursive grammars | Phrases nest and repeat | Best 975 rules, 13,591 B (U-shaped MDL) | Works; better as a context source |
| Energy (Boltzmann) | Log-linear scores learn contexts | 2.0897 vs 1.9378 counts | Does not beat counts |
| Born amplitudes | Squared amplitudes as probabilities | 2.3040 vs 2.2624 Boltzmann | No gain |
| Euclidean: repeat offsets | Offsets recur | −512 B | Loses |
| Euclidean: lattice | Aligned records | net −0.9% | Small gain |
| Information geometry | Geometric mixing is the right family | 1.9834 vs 2.0628 linear; natural gradient 2.0173 | Geometric wins; natural gradient loses |
| Topology: planar stack | Markup is nested | net −2.4% (depth ≤ 1) | Gain mostly a flag |
| Topology: link graph | Clusters predict links | no gain on corpus; 17% of planted gain | No evidence on corpus |
| Quantum topology | State sums, amplitudes | Stack-aware 1.565 vs 1.767 (constructed source) | Conceptual |
| Harmonic / period | Fixed record width | net −26.1% (fixture) | Strong where structure exists; none on corpus |
| Langlands-flavoured generators | Tables follow a rule | 115 B vs 5,397 B table | Potential ≤ ~1% of corpus |
| Functors / lenses | Reversible simplification helps | Loses by 60 B (xz), 947 B (model) | Complement costs the gain |
| Temperature | Models are over-confident | −1.0% held-out | Cheap, real |
| Renormalisation scales | Word and class structure | net −16.9% | Robust on this corpus |
| Landauer / entropy rate | Physical calibration | floor 2.3e-11 J vs 1.8e7 J | Not binding; estimator biased |

Untested ideas are in Chapter 17 of the casebook.
