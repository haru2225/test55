# test55 — test50's Si-only model, l≤6 irreps, native box, + CGMD

**Rebuilt** on request from an earlier version of this file that used a SiO4-tetrahedron-centroid
coarse-graining. test55 is now:

1. **Si-only coarse-graining** — same mapping as test50/52 (keep the 64 Si, drop all 128 O; no
   averaging, no weighting), **not** the tetrahedron-centroid mapping this file used before.
2. **Spherical-harmonics l_max raised to 6** — one order past test47/48's own l≤5, the highest
   tried anywhere in this project.
3. **Scale pulled back down to compensate**: `--replicate` defaults to 1 (native 13.573 Å box, 64
   Si, no tiling) and `--cutoff`/`--large-cutoff` default to **5.0/5.0** (test47/48/49's own
   original value for this box — not test50's later 6.5/6.7 or 8.0/8.2 — zero rattle margin, the
   same accepted limitation test47/48's own README already documents). l≤6 is already heavier per
   edge than l≤5, which was already what caused test49's CUDA OOM — keeping the box/cutoff small
   is the trade-off made to afford the higher l_max.
4. **CGMD** (`export`/`md`, ported from `test37.py`) is unchanged from the previous version — see
   below.

The model, training objective, sampler, `prepare`/`train`/`generate` CLI structure are otherwise
**exactly test50.py, unchanged**.

## Coarse-graining

Index-preserving subset selection: keep the 64 Si, drop every O, no coordinate transform or
averaging — identical in kind to test50/52's own `prepare()`. See test50.py's own README for why
this throws away the Si-O bonding information needed to disambiguate the true lattice registration
(test53/54, which bring O back instead, are the other side of that experiment).

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

qsub -P PROJECT_ID -v STAGE=prepare,OUTPUT=sio2-si-only/dataset-pilot run_test55.pbs

# cutoff/large-cutoff=5.0/5.0, irreps l<=6, replicate=1 are all defaults -- no flags needed
qsub -P PROJECT_ID -v STAGE=train,DATASET=sio2-si-only/dataset-pilot,OUTPUT=sio2-si-only/checkpoint1 \
    run_test55.pbs

# one-shot reverse-diffusion generation (same as test50)
qsub -P PROJECT_ID -v STAGE=generate,CHECKPOINT=sio2-si-only/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-si-only/checkpoint1/generated,INIT=crystal-noised \
    run_test55.pbs

# CGMD: export the force-field bundle, then run Langevin dynamics with it
qsub -P PROJECT_ID -v STAGE=export,CHECKPOINT=sio2-si-only/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-si-only/forcefield1,SIGMA_REF=0.3 \
    run_test55.pbs
qsub -P PROJECT_ID -v STAGE=md,CHECKPOINT=sio2-si-only/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-si-only/md1,SIGMA_REF=0.3,STEPS=100000,TIMESTEP_FS=0.1,DAMPING_PS=0.1 \
    run_test55.pbs
```

Locally (no PBS/Singularity):

```bash
python test55.py prepare --output sio2-si-only/dataset-pilot   # uses bundled simu_data/
python test55.py train --dataset sio2-si-only/dataset-pilot --output sio2-si-only/checkpoint1 --device cuda
python test55.py export --checkpoint sio2-si-only/checkpoint1/checkpoint.pt \
    --output sio2-si-only/forcefield1 --sigma-ref 0.3
python test55.py md --checkpoint sio2-si-only/checkpoint1/checkpoint.pt \
    --output sio2-si-only/md1 --sigma-ref 0.3 --steps 100000 --device cuda
```

To go back to test50's original l≤1/l≤2 irreps (e.g. to isolate the effect of l_max):

```bash
python test55.py train --dataset sio2-si-only/dataset-pilot --output ... \
    --irreps-hidden "64x0e + 32x1e" --irreps-edge "4x0e + 4x1e + 2x2e"
```

## Status

Smoke-tested locally (CPU): `prepare` correctly builds 64 Si CG sites, `train` runs end to end with
the new defaults (l≤6 irreps, cutoff=5.0, replicate=1, no flags needed), `generate
--init crystal-noised` produces correct output, `export` produces a valid
`cg_score_forcefield.pt`/`.json` + `cg_start.data`, and `md` runs a short Langevin trajectory
(temperature tracks ~300 K, no non-finite forces). **Not yet trained for real, run on GPU, or run
for a long MD trajectory** — whether l≤6 actually helps (vs. test50's l≤4) and whether this force
field is stable over any extended time scale are both completely unverified.
