# test55 — test50's model, unchanged, on a NEW SiO4-tetrahedron-centroid CG mapping + CGMD

Two things, both new relative to test50-54:

1. **A genuinely weighted coarse-graining**, not "keep Si / drop O" (test50/52) or "keep
   everything" (test53/54): each of the 64 Si atoms anchors one CG bead at the **mass-weighted
   centroid of that Si and its 4 nearest O** — the real SiO4 tetrahedron.
2. **CGMD**: `export`/`md` subcommands (new — test50-54 never had them), ported from
   `test37.py`'s own `export`/`md`, which run actual Langevin dynamics using the trained
   denoiser's output as an approximate force, instead of only one-shot reverse-diffusion
   `generate`.

The model, training objective, sampler, `prepare`/`train`/`generate` CLI structure are **exactly
test50.py, unchanged**.

## The coarse-graining, and a geometric subtlety it does NOT solve

`prepare()` finds, once from the reference data's first frame, each Si's 4 nearest O (PBC-aware
minimum-image distances) — verified against the bundled reference data: every Si has a clean
nearest-4 shell at 1.53-1.73 Å, and this topology is then fixed and reused for every frame and for
generation/MD (same "solid crystal, fixed topology" assumption every file in this project makes).
Each frame's CG bead position is the mass-weighted centroid of that Si plus its 4 O.

**The subtlety test39.py's own docstring already named** ("does not combine overlapping SiO4 and
AlO6 groups or silently redistribute their shared oxygen masses"): in a corner-sharing SiO2
network, every O bridges exactly **two** Si tetrahedra (verified: all 128 O atoms are each a
nearest-4 neighbor of exactly 2 different Si) — so the 64 tetrahedra are **not disjoint atom
groups**, unlike what test37.py's general weighted `prepare()` (which requires disjoint groups)
assumes. test55 does not split each bridging O's weight 50/50 between its two tetrahedra; it uses
the simpler, more common convention of letting each O contribute its **full** mass/position to
*every* tetrahedron centroid it belongs to. This makes each bead a faithful **local-environment**
centroid (and, in practice, very close to the Si position itself — mean deviation ~0.03 Å, up to
~0.07 Å, reflecting real thermal tetrahedral distortion, since a perfectly regular tetrahedron's
centroid coincides exactly with its central atom) but **not** a mass-conserving partition of the
system (an O's mass effectively gets counted twice, once per tetrahedron it bridges).

## CGMD: `export` and `md`

Ported from `test37.py`'s own `export()`/`md()`, unchanged in spirit. The one real adaptation:
test37's model is sigma-**agnostic** (`model(data)`, no time input) so its `md` calls it the same
way at every step; `NequIP_TimeEmbed` requires a `t`, so `export`/`md` fix
`t = sigma_ref / sigma_max_train` (the checkpoint's own trained `sigma_max`) for the **entire** MD
run. This is a real, meaningful difference from `generate`: there is no annealing here — the force
field approximates dynamics at *one* chosen noise/temperature scale (`--sigma-ref`, no default —
you must pick a value inside the checkpoint's trained `[sigma_min, sigma_max]` range), not a
schedule from clean to noisy.

- `export --checkpoint ... --output ... --sigma-ref N` → `cg_score_forcefield.pt`/`.json` (the
  checkpoint + metadata bundle, same format test37.py's own LAMMPS callback expects) and
  `cg_start.data` (a LAMMPS-format starting structure).
- `md --checkpoint ... --output ... --sigma-ref N --steps M` → runs ASE Langevin dynamics with a
  `ScoreCalculator` whose force is `F = -kB*T*model(data, t)/sigma_ref**2` (test37's own formula),
  `FixCom` constraint, Maxwell-Boltzmann initialization. Writes `md.extxyz` (full trajectory),
  `latest.extxyz`/`final.extxyz`, `thermo.csv` (step/time/temperature/max-force/elapsed), and
  `run_metrics.json`.

**This force field is experimental and unvalidated** — exactly as test37.py's own CAVEAT already
says for its version: the model was trained to denoise, not to predict a real conservative force,
so `F = -kB*T*dx/sigma_ref**2` is a heuristic that happens to point roughly the right direction
near `sigma_ref`, not a derived or energy-conserving force field. No energy, no virial, no
guarantee of stability over long runs. `analyze` (test37's third CGMD subcommand, partial-RDF
comparison) is **not** ported here.

## Usage

```bash
git clone git@github.com:haru2225/test55.git
cd test55

module load singularity
singularity build test55.sif Singularity.def

qsub -P PROJECT_ID -v STAGE=prepare,OUTPUT=sio2-tetra/dataset-pilot run_test55.pbs

qsub -P PROJECT_ID -v STAGE=train,DATASET=sio2-tetra/dataset-pilot,OUTPUT=sio2-tetra/checkpoint1 \
    run_test55.pbs

# one-shot reverse-diffusion generation (same as test50)
qsub -P PROJECT_ID -v STAGE=generate,CHECKPOINT=sio2-tetra/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-tetra/checkpoint1/generated,INIT=crystal-noised \
    run_test55.pbs

# CGMD: export the force-field bundle, then run Langevin dynamics with it
qsub -P PROJECT_ID -v STAGE=export,CHECKPOINT=sio2-tetra/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-tetra/forcefield1,SIGMA_REF=0.3 \
    run_test55.pbs
qsub -P PROJECT_ID -v STAGE=md,CHECKPOINT=sio2-tetra/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-tetra/md1,SIGMA_REF=0.3,STEPS=100000,TIMESTEP_FS=0.1,DAMPING_PS=0.1 \
    run_test55.pbs
```

Locally (no PBS/Singularity):

```bash
python test55.py prepare --output sio2-tetra/dataset-pilot   # uses bundled simu_data/
python test55.py train --dataset sio2-tetra/dataset-pilot --output sio2-tetra/checkpoint1 --device cuda
python test55.py export --checkpoint sio2-tetra/checkpoint1/checkpoint.pt \
    --output sio2-tetra/forcefield1 --sigma-ref 0.3
python test55.py md --checkpoint sio2-tetra/checkpoint1/checkpoint.pt \
    --output sio2-tetra/md1 --sigma-ref 0.3 --steps 100000 --device cuda
```

## Status

Smoke-tested locally (CPU): `prepare` correctly builds 64 SiO4-centroid beads (the bridging-O
sanity check — every O should be the nearest-4 neighbor of exactly 2 Si — passes silently on the
bundled reference data; centroids differ from raw Si positions by ~0.03 Å mean, consistent with
real thermal tetrahedral distortion), `train` runs end to end, `export` produces a valid
`cg_score_forcefield.pt`/`.json` + `cg_start.data`, and `md` runs a short Langevin trajectory
(temperature tracked close to the target 300 K, no non-finite forces) producing all expected output
files. **Not yet trained for real, run on GPU, or run for a long MD trajectory** — whether this
force field is stable or physically sensible over any extended time scale is completely unverified
(same "experimental_unvalidated" status test37.py's own version carries).
