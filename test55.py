#!/usr/bin/env python3
"""test55: test50's model/sampler, EXACTLY unchanged, on a NEW weighted coarse-graining (one bead
per SiO4 tetrahedron, not "keep Si and drop O" like test50/52/53/54) -- PLUS CGMD: `export`/`md`
subcommands (new, ported from test37.py) that drive actual Langevin dynamics with the trained
score as an approximate force field, instead of only one-shot reverse-diffusion `generate`.

COARSE-GRAINING -- the weighted-average kind test37.py's general `prepare` does, applied here with
a specific, hand-built mapping instead of a user-supplied `mapping.json`: each of the 64 Si atoms
anchors one CG bead, positioned at the mass-weighted centroid of that Si plus its 4 nearest O atoms
(the real SiO4 tetrahedron, PBC-aware minimum-image distances; verified against the reference data
that every Si has a clean nearest-4 shell at 1.53-1.73 A, well separated from the 5th-nearest O).
THE GEOMETRIC SUBTLETY test39.py's own docstring already flagged ("does not combine overlapping
SiO4 and AlO6 groups or silently redistribute their shared oxygen masses"): in a corner-sharing
SiO2 network every O bridges exactly TWO Si tetrahedra (verified: all 128 O atoms are each a
nearest-4 neighbor of exactly 2 different Si, in the bundled reference data) -- so the 64 tetrahedra
are NOT disjoint atom groups the way test37.py's general weighted `prepare` requires. test55 does
NOT solve this by splitting each bridging O's weight 50/50; it uses the simpler, more common
convention of letting each O contribute its FULL mass/position to the centroid of EVERY tetrahedron
it belongs to (so bridging O's are double-counted across different beads -- this is a per-bead
*local-environment* centroid, not a mass-conserving partition of the system). The Si-O4 topology
(which 4 O indices belong to which Si) is fixed once from the reference data's first frame and
reused for every frame and for generation/MD, matching every other file in this project's
"fixed-topology solid crystal" assumption.

Model/sampler/CLI structure for `prepare`/`train`/`generate`: unchanged from test50.py (which
copied them byte-for-byte from test38.py) -- single CG species ("SiO4", atomic_number=14 reused
for ASE export purposes, mass_amu=92.0831=28.0855+4*15.9994), 64 sites, same shape as test50's
Si-only dataset, so none of `NequIP_TimeEmbed`/`graph`/`load_dataset`/etc. needed any changes.

CGMD (`export`, `md` -- new, ported from test37.py's own `export`/`md`, which test50-54 never
had): `export` bundles the checkpoint plus a `cg_start.data` LAMMPS file into a
`cg_score_forcefield.pt`/`.json` pair, same format test37.py uses. `md` runs ASE Langevin dynamics
with a `ScoreCalculator` whose force is `F = -kB*T*model(data, t)/sigma_ref**2` (test37's own
formula) -- EXPERIMENTAL AND UNVALIDATED, exactly as test37's own CAVEAT already says: the model
was trained to denoise, not to predict a real conservative force, so this is a heuristic that
happens to point roughly the right direction near sigma_ref, not a derived or energy-conserving
force field. The one real difference from test37's `md`: test37's model is sigma-AGNOSTIC
(`model(data)`, no time input), so its `md` calls the model unconditionally; test55's
`NequIP_TimeEmbed` requires `t`, so `md`/`export` fix `t = sigma_ref/sigma_max_train` (the
checkpoint's own trained sigma_max) for the ENTIRE run -- the force field this produces is only
meant to approximate dynamics AT that one noise/temperature scale, not to anneal across sigma the
way `generate` does. `analyze` (test37's third CGMD subcommand, RDF comparison) is NOT ported here.

Everything else -- cutoff/large-cutoff/sigma-max/updates defaults, `--irreps-hidden`/
`--irreps-edge`, `--replicate`, the half-box safety guard, `--init crystal`/`crystal-noised`, the
deterministic-steps warning, `trajectory.extxyz` export -- is unchanged from test50.py.

    python test55.py prepare --output sio2-tetra/dataset-pilot   # uses bundled simu_data/
    python test55.py train --dataset sio2-tetra/dataset-pilot --output sio2-tetra/checkpoint1 --device cuda
    python test55.py generate --checkpoint sio2-tetra/checkpoint1/checkpoint.pt \
        --output sio2-tetra/checkpoint1/generated --init crystal-noised --device cuda
    python test55.py export --checkpoint sio2-tetra/checkpoint1/checkpoint.pt --output sio2-tetra/forcefield1
    python test55.py md --checkpoint sio2-tetra/checkpoint1/checkpoint.pt --output sio2-tetra/md1 \
        --sigma-ref 0.3 --steps 10000 --device cuda

Everything test38.py's own CAVEAT says still applies: no scalar energy, no equilibrium claim, no
physical clock, one model per condition. The CGMD force field additionally carries test37's own
"experimental_unvalidated" / "provides_energy=False" / "provides_virial=False" status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import ase.io
import numpy as np
import torch

torch.serialization.add_safe_globals([slice])  # e3nn loads its own constants.pt with torch.load

from ase import Atoms, units
from ase.neighborlist import primitive_neighbor_list
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter

# ===== 基本設定・定数 =====
ROOT = Path(__file__).resolve().parent
FORMAT = "test55-sio2-tetra-cg-time-denoiser-v1"  # このtest55用チェックポイントの識別子(test50のものとは別)
DATASET_FORMAT = "test55-sio2-tetra-cg-v1"  # prepare()が書き出すデータセット形式の識別子
DATASET_FORMATS = {DATASET_FORMAT}  # 読み込めるデータセット形式(load_dataset()はtest38と同じ関数、中身の集合だけ差し替え)
CAVEAT = (
    "test55 sigma-conditioned displacement denoiser with a reverse variance-exploding "
    "SDE sampler -- test50's model and sampler code, unchanged, applied to an SiO2 "
    "(beta-cristobalite) SiO4-tetrahedron-centroid coarse-graining (one CG bead per Si, at the "
    "mass-weighted centroid of that Si and its 4 nearest O; bridging O atoms contribute fully to "
    "both tetrahedra they belong to, not split 50/50, so this is a local-environment centroid, "
    "not a mass-conserving partition). Not a scalar energy or a temperature-conditioned "
    "equilibrium score. No energy/virial, no explicit electrostatics, pressure, shear response "
    "or physical kinetics are provided from `generate`. The `export`/`md` CGMD force field "
    "additionally is experimental and unvalidated: it reuses the denoising output as a heuristic "
    "Langevin force, not a derived or energy-conserving one. One model per condition; "
    "generation/MD frames are not equilibrium MD data."
)
STOP = False  # SIGINT/SIGTERMを受け取ったらTrueにして、train/generateループを安全に中断させるフラグ


def request_stop(signum, frame):
    # シグナルハンドラ: Ctrl-CやHPCのジョブ時間切れ通知を受けたときに呼ばれる。
    # ここで即座に終了せず、ループ側にSTOPを見てもらってから
    # チェックポイントを保存してから終了する(生成/学習の再開性を保つため)。
    global STOP
    STOP = True


def positive(value):
    # argparseの型変換関数: 正の有限値であることを保証する(sigmaや学習率など)。
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def count(value):
    # argparseの型変換関数: 1以上の整数であることを保証する(ステップ数など)。
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_count(value):
    # argparseの型変換関数: 0以上の整数であることを保証する(deterministic-stepsなど、0も許す値)。
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def digest(path):
    # ファイルのSHA-256ハッシュを計算する。データセットやチェックポイントが
    # 途中で書き換わっていないか(再現性)を確認するために使う。
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    # JSONを一時ファイルに書いてからatomicにrename。書き込み途中でプロセスが
    # 落ちても、既存の正しいファイルが壊れた中途半端な内容で上書きされない。
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, obj):
    # チェックポイント(torch.save)も同様にtmp書き込み→rename でatomicに保存する。
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def new_output(path):
    # 出力先ディレクトリを新規作成する。既に中身がある場合はエラーにして、
    # 別の実行結果を誤って上書き・混在させないようにする。
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {path}")
    return path


def device_for(name):
    # "auto"ならCUDA→MPS→CPUの優先順で自動選択。明示指定されたデバイスが
    # 実際には使えない場合はここでエラーにする(黙って別デバイスに落とさない)。
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def rng_state():
    # numpy/torch(+CUDA/MPSがあれば)の乱数状態をまとめて保存用に取得する。
    # --resumeで学習・生成を再開したときに、乱数列を継続させるために使う。
    state = {"numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    # rng_state()で保存した乱数状態を復元する。
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


# ===== test37から移植したデータ・グラフ・種(species)まわりのヘルパー =====

def bessel(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Vendored from DM2/src/graphite/nn/basis.py (function bessel)."""
    # スカラー値(ここではエッジの距離)を、複数のBessel基底関数の値に展開する。
    # NequIPが距離をそのまま数値として使うのではなく、周波数の異なる
    # sin波の重ね合わせとして表現することで、距離依存性を学習しやすくする。
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis + 1, dtype=x.dtype, device=x.device)
    return ((2 / c) ** 0.5) * torch.sin(n * torch.pi * x / c) / x


class InitialEmbedding(nn.Module):
    """Same embedding as test32.py/test37.py: two species embeddings and Bessel edges."""
    # ノード(原子)の「種」を2種類の埋め込みベクトルに変換し、エッジ(ボンド)の
    # 距離をBessel基底に変換する、NequIPモデルの最初の入力層。
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)  # 更新されていくノード特徴量の初期値
        self.embed_node_z = nn.Embedding(num_species, 8)  # モデル全体を通して固定される補助的なノード特徴量
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def architecture(num_species, cutoff, irreps_hidden="64x0e + 32x1e + 16x2e + 8x3e + 4x4e",
                  irreps_edge="4x0e + 4x1e + 2x2e + 2x3e + 1x4e"):
    # モデルの構造(irreps=e3nnの回転等変な特徴量の型、畳み込み層の数など)を
    # 1つの辞書にまとめたもの。チェックポイントに保存しておき、生成時に
    # 同じ構造のモデルを再構築するために使う。test37と同じ構造。
    # irreps_hidden/irreps_edge引数はtest50独自の追加(test38は引数なしのハードコード、
    # l<=1隠れ層/l<=2エッジ固定)。ここでのデフォルトはl<=4構成(下のtrainの--irreps-hidden/
    # --irreps-edgeのデフォルトと合わせてある)。
    return dict(num_species=num_species, cutoff_angstrom=cutoff,
                irreps_node_x="8x0e", irreps_node_z="8x0e",
                irreps_hidden=irreps_hidden, irreps_edge=irreps_edge,
                irreps_out="1x1e", num_convs=3, radial_neurons=[16, 64], num_neighbors=12)


def graph(positions, cell, type_ids, cutoff, device):
    # Same periodic neighbor construction as test32/37; graph vectors are not
    # differentiable w.r.t. positions. This is deliberate for a dx model.
    # 周期境界条件(PBC)付きで、cutoff半径以内の原子対(i, j)とその変位ベクトルvecを列挙し、
    # PyTorch Geometricの`Data`グラフオブジェクトを組み立てる。
    i, j, vec = primitive_neighbor_list("ijD", [True] * 3, cell, positions, cutoff=cutoff)
    if not len(i):
        raise ValueError("No graph edges: check box, units and cutoff")
    return Data(x=torch.as_tensor(type_ids, dtype=torch.long, device=device),
                pos=torch.as_tensor(np.asarray(positions).copy(), dtype=torch.float32, device=device),
                edge_index=torch.as_tensor(np.stack((i, j)), dtype=torch.long, device=device),
                edge_attr=torch.as_tensor(vec, dtype=torch.float32, device=device))


def load_dataset(folder):
    # test50形式のデータセット(positions.npy, cells.npy, metadata.json; prepare()が書き出す)を読み込む。
    # メタデータのフォーマット・単位・配列の形状・保存後の改ざん有無(sha256)を検証してから返す。
    # (関数自体はtest38.pyのload_dataset()と同一ロジック。DATASET_FORMATSの中身だけがtest50用に差し替わっている。)
    folder = Path(folder).resolve()
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("format") not in DATASET_FORMATS or meta.get("length_unit") != "angstrom":
        raise ValueError("Expected a test55 SiO2 tetrahedron-CG dataset with explicit angstrom units")
    pos = np.load(folder / "positions.npy", mmap_mode="r", allow_pickle=False)
    cells = np.load(folder / "cells.npy", mmap_mode="r", allow_pickle=False)
    if pos.shape != (meta["frames"], len(meta["type_ids"]), 3) or cells.shape != (len(pos), 3, 3):
        raise ValueError("Dataset shapes disagree with metadata")
    for name in ("positions.npy", "cells.npy"):
        if digest(folder / name) != meta["sha256"][name]:
            raise ValueError(f"Dataset modified after preparation: {name}")
    return pos, cells, meta


def atoms_from_meta(positions, cell, meta):
    # モデル用のtype_id配列を、可視化・エクスポート用にase.Atomsオブジェクト
    # (実際の原子番号・質量を持つ)へ変換する。test55ではspeciesは「SiO4」1種類のみ
    # (原子番号は可視化のためSi=14を流用、質量はSiO4四面体の合計質量92.0831amu)。
    ids = np.asarray(meta["type_ids"])
    atoms = Atoms(numbers=[meta["species"][i]["atomic_number"] for i in ids],
                  positions=positions, cell=cell, pbc=True,
                  masses=[meta["species"][i]["mass_amu"] for i in ids])
    atoms.set_array("cg_type", ids.copy())
    return atoms


# --- Vendored from DM2/src/graphite (nn/conv/e3nn_nequip.py, nn/models/e3nn_nequip.py,
# transforms/downselect_edges.py, transforms/rattle_particles.py) so that this file does
# not import DM2 or require a DM2 checkout / DM2_ROOT to be present. -----------------

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    # 2つのirreps(既約表現)のテンソル積が、指定した出力既約表現ir_outを
    # 生成しうるかどうかを判定する。e3nnのGate/TensorProductを組み立てる際に、
    # 「この組み合わせは数学的に意味があるか」を事前にチェックするために使う。
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(nn.Module):
    # 2つのモジュール(例: Interaction畳み込み層とGate活性化層)を直列に繋げるだけの
    # 薄いラッパー。NequIP_TimeEmbedの各層は Compose(Interaction, Gate) として作られる。
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class GaussianBasisEmbedding(nn.Module):
    """Embeds a scalar value in [0,1] using a Gaussian basis set followed by a dense layer."""
    # sigma/time条件付けの核心部分: スカラー値t(=sigma/sigma_max_train, 0〜1の値)を
    # 複数のガウス基底関数に展開し(bessel関数と同じ発想)、2層のMLPで
    # ベクトル特徴量に変換する。NequIP_TimeEmbedがこの出力(h_node_t)を
    # 各畳み込み層のノード特徴量に足し込むことで、モデルが「今どれくらいの
    # ノイズレベルを相手にしているか」を認識できるようになる。
    def __init__(self, num_basis=12, embedding_dim=32, min_sigma=0.1,
                 learn_means=False, learn_sigmas=False, min_value=0, max_value=1):
        super().__init__()
        means = torch.linspace(min_value, max_value, num_basis)  # 各ガウス基底の中心位置
        if learn_means:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer('means', means)
        sigmas = torch.ones_like(means) * max(min_sigma, 1.0 / (num_basis - 1))  # 各ガウス基底の幅
        if learn_sigmas:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)
        hidden_dim = max(embedding_dim * 2, num_basis)
        self.layer1 = nn.Linear(num_basis, hidden_dim)
        self.activation = nn.Softplus()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)

    def gaussian_basis(self, x):
        # tの値を、各ガウス基底中心からの距離に応じた「近さ」のベクトルに変換する。
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_expanded = x.expand(-1, self.means.shape[0])
        return torch.exp(-0.5 * ((x_expanded - self.means) / self.sigmas) ** 2)

    def forward(self, x):
        basis_activation = self.gaussian_basis(x)
        hidden = self.activation(self.layer1(basis_activation))
        return self.layer2(hidden)


class Interaction(nn.Module):
    """Equivariant `Interaction` layer from NequIP (https://arxiv.org/pdf/2101.03164.pdf)."""
    # NequIPの中核となる1つの畳み込み(メッセージパッシング)層。
    # 回転・並進に対して等変(equivariant)なテンソル積を使って、
    # 各原子の近傍からの情報を集約し、ノード特徴量を更新する。
    def __init__(self, irreps_in, irreps_node, irreps_edge, irreps_out,
                 radial_neurons=[16, 64], num_neighbors=1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        # irreps_in(ノード特徴量)とirreps_edge(球面調和関数)のテンソル積のうち、
        # 最終的にirreps_outとして使えるものだけを集めて、中間表現irreps_midを組み立てる。
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        assert irreps_mid.dim > 0, (
            f"irreps_in={self.irreps_in} times irreps_edge={self.irreps_edge} "
            f"produces nothing in irreps_out={self.irreps_out}."
        )
        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.sc = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_out)  # 自己結合(残差)経路
        self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_in)  # メッセージパッシング前の線形変換
        self.conv = o3.TensorProduct(
            self.irreps_in, self.irreps_edge, irreps_mid, instructions,
            internal_weights=False, shared_weights=False,  # 重みは下のmlp(ボンド長依存)から供給される
        )
        self.lin2 = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, self.irreps_out)  # メッセージパッシング後の線形変換
        self.mlp = FullyConnectedNet(radial_neurons + [self.conv.weight_numel], torch.nn.functional.silu)  # ボンド長(Bessel基底)からconvの重みを生成するMLP

        # SkipInit mechanism inspired by https://arxiv.org/pdf/2002.10444.pdf
        # 学習開始時点ではconvパスの寄与をゼロにしておき(alpha=0スタート)、
        # 学習が進むにつれて徐々にconvパスの影響を強めていく安定化トリック。
        self.alpha = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, "0e")
        with torch.no_grad():
            self.alpha.weight.zero_()
        assert self.alpha.output_mask[0] == 1.0, (
            f"irreps_mid={irreps_mid} and irreps_node={self.irreps_node} are not able to generate scalars."
        )

    def forward(self, x, node_attr, edge_index, edge_attr, edge_len_emb):
        i, j = edge_index
        num_nodes = x.size(0)
        node_self_connection = self.sc(x, node_attr)  # 残差(自己)経路の出力
        node_features = self.lin1(x, node_attr)
        # 各エッジについて、送り手ノードiの特徴量とエッジの球面調和関数edge_attrとの
        # テンソル積を、ボンド長依存の重み(self.mlp(edge_len_emb))で計算する。
        edge_features = self.conv(node_features[i], edge_attr, weight=self.mlp(edge_len_emb))
        # 受け手ノードjごとにエッジメッセージを合計(scatter)し、近傍数で正規化する。
        node_features = scatter(edge_features, j, dim=0, dim_size=num_nodes).div(self.num_neighbors ** 0.5)
        node_conv_out = self.lin2(node_features, node_attr)
        alpha = self.alpha(node_features, node_attr)
        m = self.sc.output_mask
        alpha = (1 - m) + alpha * m
        # 残差経路 + alphaでスケールした畳み込み経路、を足し合わせて出力する。
        return node_self_connection + alpha * node_conv_out


class NequIP_TimeEmbed(nn.Module):
    """Sigma/time-conditioned NequIP (https://arxiv.org/pdf/2101.03164.pdf), vendored from
    DM2/src/graphite/nn/models/e3nn_nequip.py.

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
        num_neighbors (float): Typical or average node degree (used for normalization).
    """
    def __init__(self, init_embed, irreps_node_x='8x0e', irreps_node_z='8x0e',
                 irreps_hidden='64x0e + 32x1e + 32x2e', irreps_edge='1x0e + 1x1e + 1x2e',
                 irreps_out='1x1e', num_convs=3, radial_neurons=[16, 64], num_neighbors=12):
        super().__init__()
        self.init_embed = init_embed
        self.irreps_node_x = o3.Irreps(irreps_node_x)
        self.irreps_node_z = o3.Irreps(irreps_node_z)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.num_convs = num_convs

        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        # num_convs層分のInteraction+Gateを積み重ねる。各層で、スカラー成分と
        # ベクトル/テンソル成分をGateで非線形活性化しながら特徴量を更新していく。
        irreps = self.irreps_node_x
        self.interactions = nn.ModuleList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                         if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)])
            irreps_gated = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                       if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge, ir)])

            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node_z, self.irreps_edge, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node_z, self.irreps_edge, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps={irreps} times irreps_edge={self.irreps_edge} is unable "
                                      f"to produce gates needed for irreps_gated={irreps_gated}.")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )
            conv = Interaction(
                irreps_in=irreps, irreps_node=self.irreps_node_z, irreps_edge=self.irreps_edge,
                irreps_out=gate.irreps_in, radial_neurons=radial_neurons, num_neighbors=num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps, irreps_in2=self.irreps_node_z, irreps_out=self.irreps_out,
        )

        # ここがtest38の要: sigma/time条件付けのための層を追加する。
        # 各畳み込み層のスカラー隠れ次元数(size_embed)に合わせたGaussianBasisEmbeddingで
        # t(=sigma/sigma_max_train)を埋め込み、t_projectionで各ノード特徴量の次元数に線形変換する。
        size_embed = int(str(irreps).split("x")[0])
        self.t_embed = GaussianBasisEmbedding(embedding_dim=size_embed)
        t_embed_dim = self.t_embed.layer2.out_features
        self.t_projection = nn.Linear(t_embed_dim, irreps.dim)

    def forward(self, data, t):
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge

        # スカラー値t(このバッチ全体で1つの値)を埋め込み、全ノードに同じベクトルとして
        # ブロードキャストする。1バッチ=1つのグラフしか正しく条件付けできない点に注意
        # (下のtrain()内のrattle_atのコメント参照)。
        h_node_t = self.t_embed(t)
        h_node_t = h_node_t.expand(h_node_x.shape[0], -1)
        h_node_t = self.t_projection(h_node_t)

        # エッジベクトルを球面調和関数に変換してから、各Interaction+Gate層を通し、
        # 毎層h_node_tを足し込むことでsigma情報を伝え続ける。
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
            h_node_x = h_node_x + h_node_t

        # 最終的に3次元ベクトル(irreps_out='1x1e')、つまり各原子の変位予測dxを出力する。
        return self.out(h_node_x, h_node_z)


class DownselectEdges(BaseTransform):
    """Vendored from DM2/src/graphite/transforms/downselect_edges.py."""
    # graph()はtraining用に少し大きめのlarge_cutoffで候補エッジを作っておき、
    # このDownselectEdgesで実際のモデルcutoff以内のエッジだけに絞り込む。
    # (RattleParticlesでノイズを加えた後、距離が変化してからこの絞り込みを行うことで、
    # ノイズ後もcutoff以内に収まっているエッジだけを使う。)
    def __init__(self, cutoff, cell=None):
        super().__init__()
        self.cutoff = cutoff
        self.cell = cell

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        mask = (edge_attr[:, :3].norm(dim=1) <= self.cutoff)
        data.edge_index = edge_index[:, mask]
        data.edge_attr = edge_attr[mask]
        return data

    def forward(self, data):
        return self.__call__(data)

    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'


class RattleParticles(BaseTransform):
    """Vendored from DM2/src/graphite/transforms/rattle_particles.py. Applies a random
    Gaussian noise to particle positions, with standard deviation drawn uniformly from
    [sigma_min, sigma_max]."""
    # 学習時に「正解の構造」にガウスノイズを加えて壊し(corrupt)、モデルには
    # 「加えられたノイズdxを予測して元に戻す」タスクを学習させる。これが
    # denoising score matching(スコアベース生成モデル)の学習の基本形。
    def __init__(self, sigma_max, sigma_min=0.001):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, data):
        if data.batch is not None:
            # バッチ内のグラフごとに別々のsigmaを[sigma_min, sigma_max]から一様サンプルする
            # (test38ではtrain()側でsigma_min=sigma_maxに固定して呼ぶため、実質バッチ全体で1つのsigmaになる)。
            sigma = torch.empty(data.num_graphs, device=data.pos.device).uniform_(
                self.sigma_min, self.sigma_max)
            sigma = sigma[data.batch, None]
        else:
            sigma = torch.empty(1, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)

        eps = torch.randn_like(data.pos)  # 標準正規分布ノイズ
        data.dx = sigma * eps  # モデルが予測すべき正解の変位(ノイズそのもの)
        data.pos = data.pos + data.dx  # 座標を実際に壊す

        if data.edge_attr is not None:
            # 座標を動かしたので、既存のエッジベクトル(相対変位)も整合するように更新する。
            i, j = data.edge_index
            data.edge_attr = data.edge_attr + data.dx[j] - data.dx[i]

        data.sigma = sigma  # 後で参照できるように保存(このtest38ではdata.sigmaは直接は使わずtを別途渡す)
        data.eps = eps
        return data

    def forward(self, data):
        return self.__call__(data)


# --- test38-specific code -------------------------------------------------------------

def build_time_model(config: dict, device: torch.device) -> nn.Module:
    # architecture()で作った構成辞書からNequIP_TimeEmbedモデルを組み立てる。
    values = {k: v for k, v in config.items() if k not in ("num_species", "cutoff_angstrom")}
    model = NequIP_TimeEmbed(
        init_embed=InitialEmbedding(config["num_species"], config["cutoff_angstrom"]),
        **values,
    )
    return model.to(device)


def warm_start_from_plain_checkpoint(model: NequIP_TimeEmbed, plain_state_dict: dict) -> None:
    """Copy every weight NequIP_TimeEmbed shares with plain NequIP, then zero the new
    time-projection layer so the warm-started model equals the source checkpoint at
    t=anything until training moves it."""
    # test37(sigma条件付けなしのplain NequIP)で既に学習済みのチェックポイントから
    # 重みを引き継ぐための関数。NequIP_TimeEmbedはplain NequIPと全く同じ
    # Interaction/Gate/出力層を持つので、それらの重みはそのままコピーできる。
    # 新規に追加されたt_embed/t_projection(sigma条件付け用の層)だけは
    # plain側チェックポイントに存在しないので、ここではまだ扱わない。
    own_state = model.state_dict()
    missing = [k for k in own_state if k not in plain_state_dict]
    unexpected = [k for k in plain_state_dict if k not in own_state]
    if unexpected:
        raise ValueError(f"--warm-start checkpoint has unexpected keys for this architecture: {unexpected}")
    if any(not k.startswith(("t_embed.", "t_projection.", "time_scalar_mask")) for k in missing):
        raise ValueError(f"--warm-start checkpoint is missing non-time-conditioning keys: {missing}")
    own_state.update(plain_state_dict)
    model.load_state_dict(own_state)
    # t_projectionの重み・バイアスをゼロにすることで、h_node_t(sigma由来の特徴量)が
    # 各層の出力に何も足さない状態にする。つまりウォームスタート直後は
    # 「sigmaを完全に無視するモデル」= 元のplainチェックポイントと数値的に全く同じ
    # 挙動になり、そこから学習を進めるにつれて徐々にsigma依存性を獲得していく。
    nn.init.zeros_(model.t_projection.weight)
    nn.init.zeros_(model.t_projection.bias)


def checkpoint_time_model(path, device):
    # test50形式のチェックポイントを読み込み、保存されていたarchitectureから
    # モデルを再構築して重みを復元する(生成時に使う)。
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError("Expected a test55 sigma-conditioned checkpoint")
    model = build_time_model(ck["architecture"], device)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval(), ck


# --- test50-specific code: build the Si-only SiO2 CG dataset test38's load_dataset() expects ---

def si_o4_topology(positions0, cell0, si_indices, o_indices):
    # test55の要: 最初のフレームから「どのSiにどの4つのOが属するか」というSiO4四面体の
    # トポロジーを一度だけ決定し、以降全フレーム・生成/MD時も同じ対応を使い続ける
    # (このプロジェクト共通の「固体結晶はトポロジーが変わらない」前提に合わせる)。
    # PBC(最小image)を考慮して、各Siに最も近い4個のOを選ぶ。
    inv = np.linalg.inv(cell0)

    def mic(disp):
        frac = disp @ inv
        frac -= np.round(frac)
        return frac @ cell0

    topology = []  # topology[i] = si_indicesのi番目のSiに対応する、o_indices内のインデックス4つ
    for si in si_indices:
        disp = mic(positions0[o_indices] - positions0[si])
        dist = np.linalg.norm(disp, axis=1)
        nearest4 = np.argsort(dist)[:4]
        topology.append(o_indices[nearest4])
    topology = np.asarray(topology)  # (64, 4)

    # 健全性チェック: 本来のコーナー共有SiO2網目なら、各Oはちょうど2つのSiの最近接4個に
    # 現れるはず(架橋酸素)。これが崩れている場合は、この単純な「最近接4個」ルールが
    # この構造には合わないということなので、黙って進めず知らせる。
    counts = np.zeros(len(positions0), dtype=int)
    for o4 in topology:
        counts[o4] += 1
    bridging_counts = counts[o_indices]
    if not np.all(bridging_counts == 2):
        unique, freq = np.unique(bridging_counts, return_counts=True)
        print(f"NOTE: expected every O to be the nearest-4 neighbor of exactly 2 Si "
              f"(corner-sharing SiO4 network); got counts {dict(zip(unique.tolist(), freq.tolist()))}. "
              f"Proceeding anyway -- each bead is still its own Si's nearest-4-O centroid.",
              flush=True)
    return topology


def tetrahedron_centroids(frames, cell_lengths, si_indices, topology, mass_si, mass_o):
    # 各フレーム・各Siについて、(Si本体 + topologyで決めた4個のO)の質量中心を計算する。
    # PBCを跨いだ重心を正しく扱うため、Siを原点にしたminimum-image変位で各Oの寄与を
    # 足し込む(frames[:, si] + 各Oのmic変位の質量加重平均)。セルは等方(Lx=Ly=Lz)かつ
    # フレームごとに変わりうる前提(呼び出し側で既に検証済み)。
    n_frames = frames.shape[0]
    n_si = len(si_indices)
    total_mass = mass_si + 4 * mass_o
    cell_len = cell_lengths[:, 0].astype(np.float64)  # (n_frames,)
    centroids = np.empty((n_frames, n_si, 3), dtype=np.float32)
    for i, (si, o4) in enumerate(zip(si_indices, topology)):
        si_pos = frames[:, si, :].astype(np.float64)  # (n_frames, 3)
        disp_sum = np.zeros((n_frames, 3), dtype=np.float64)
        for o in o4:
            disp = frames[:, o, :].astype(np.float64) - si_pos  # (n_frames, 3)
            disp -= cell_len[:, None] * np.round(disp / cell_len[:, None])  # minimum image
            disp_sum += mass_o * disp
        centroids[:, i, :] = (si_pos + disp_sum / total_mass).astype(np.float32)
    return centroids


def prepare(args):
    # test55の要: Si原子をそのまま残すのではなく、各Siとその最近接4個のOからなる
    # 「SiO4四面体」の質量中心を1つのCGビーズにする(test37.pyの一般的な重み付きprepare()
    # と同じ発想だが、架橋酸素(隣接する2つの四面体に共有されるO)は両方の重心計算に
    # フルで寄与する -- 分割して足し合わせるのではない。詳細はこのファイルの
    # モジュールdocstring参照。
    archive = np.load(args.reference_frames)
    for key in ("positions", "cell_lengths", "numbers"):
        if key not in archive.files:
            raise ValueError(f"--reference-frames is missing array '{key}'")
    positions, cell_lengths, numbers = archive["positions"], archive["cell_lengths"], archive["numbers"]
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] != cell_lengths.shape[0] \
            or cell_lengths.shape[1] != 3 or positions.shape[1] != numbers.shape[0]:
        raise ValueError("Expected positions (frames,atoms,3), cell_lengths (frames,3), "
                          "numbers (atoms,) arrays of agreeing shape")
    si_indices = np.flatnonzero(numbers == 14)
    o_indices = np.flatnonzero(numbers == 8)
    if len(si_indices) == 0 or len(o_indices) == 0:
        raise ValueError("Need both Si (14) and O (8) atoms in --reference-frames")
    if not np.allclose(cell_lengths, cell_lengths[:, :1]):
        raise ValueError("test55 assumes a cubic cell (Lx=Ly=Lz) at every frame")

    # トポロジー(どのOがどのSiの四面体に属するか)は最初のフレームだけから決める。
    cell0 = np.diag(cell_lengths[0])
    topology = si_o4_topology(positions[0], cell0, si_indices, o_indices)

    mass_si, mass_o = 28.0855, 15.9994
    centroids_all = tetrahedron_centroids(positions, cell_lengths, si_indices, topology, mass_si, mass_o)

    frames = centroids_all[::args.stride]
    lengths = cell_lengths[::args.stride]
    if len(frames) < 2:
        raise ValueError("Need at least two frames after striding")

    replicate = args.replicate
    if replicate > 1:
        # 箱をreplicate^3倍にタイル化する(test50/51/52と同じ手法。重心ビーズをそのまま
        # 複製するだけ)。注意(重要な限界): 複製された像は元の配置を厳密にコピーした
        # だけで、独立した熱ゆらぎを持つ別配置ではない。
        n = replicate
        shifts = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)],
                           dtype=np.float32)  # (n^3, 3)
        old_cell_length = lengths[:, 0].astype(np.float32)  # (frames,)
        tiled = frames[:, None, :, :] + shifts[None, :, None, :] * old_cell_length[:, None, None, None]
        frames = np.ascontiguousarray(tiled.reshape(len(frames), -1, 3)).astype(np.float32)
        lengths = lengths * n

    n_sites = frames.shape[1]
    cells = np.zeros((len(frames), 3, 3), dtype=np.float64)
    for axis in range(3):
        cells[:, axis, axis] = lengths[:, 0].astype(np.float64)

    output = new_output(args.output)
    np.save(output / "positions.npy", np.ascontiguousarray(frames).astype(np.float32))
    np.save(output / "cells.npy", cells)
    species = [{"name": "SiO4", "atomic_number": 14, "mass_amu": float(mass_si + 4 * mass_o)}]
    type_ids = [0] * n_sites

    source_meta_path = args.reference_frames.with_name(
        args.reference_frames.stem + "_metadata.json")
    source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.is_file() else None

    meta = dict(
        format=DATASET_FORMAT, length_unit="angstrom", frames=len(frames), species=species,
        type_ids=type_ids, condition={"temperature_k": args.temperature_k},
        cg_mapping=("test55 SiO4-tetrahedron centroid: one CG bead per Si, positioned at the "
                    "mass-weighted centroid of that Si and its 4 nearest O (fixed topology from "
                    "frame 0). Bridging O atoms contribute fully to both tetrahedra they belong "
                    "to (not split 50/50) -- a local-environment centroid, not a mass-conserving "
                    "partition of the system. See test55.py's own module docstring."),
        si_o4_topology=topology.tolist(),
        original_atom_count=int(numbers.shape[0]),
        si_atom_count=int(len(si_indices)), o_atom_count=int(len(o_indices)),
        replicate=replicate, replicated_site_count=n_sites,
        replicate_caveat=(None if replicate == 1 else
            f"positions are tiled {replicate}x{replicate}x{replicate} ({replicate**3} exact "
            "periodic copies of the same configuration per frame) to safely enlarge the box for "
            "a bigger --cutoff, NOT an independent larger-box MD run -- the copies are perfectly "
            "correlated with each other, unlike real thermal disorder at that box size."),
        source_reference_frames=str(args.reference_frames.resolve()),
        source_reference_frames_metadata=source_meta, source_stride=args.stride,
        scientific_caveat=CAVEAT,
        sha256={name: digest(output / name) for name in ("positions.npy", "cells.npy")},
    )
    save_json(output / "metadata.json", meta)
    print(f"Prepared {len(frames)} frames, {n_sites} SiO4-centroid CG beads"
          f"{f', replicated {replicate}x{replicate}x{replicate}' if replicate > 1 else ''}: {output}")


def train(args):
    # --- 入力チェックとセットアップ ---
    positions, cells, meta = load_dataset(args.dataset)
    if len(positions) < 3:
        raise ValueError("Training requires at least three frames")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between zero and 0.5")
    if args.sigma_max < 0.001 or args.large_cutoff < args.cutoff:
        raise ValueError("Require sigma-max >= 0.001 and large-cutoff >= cutoff")
    # test50固有のガード(test38にはない): cutoff/large_cutoffが半箱以上だと、同じ原子対が
    # 2つ以上の周期像を通じて二重に繋がってしまう(test33/48で文書化された周期像重複バグ)。
    # このバグは黙って学習データを壊すだけで例外を出さないので、ここで明示的に弾く。
    # (--replicateでタイル化した大きい箱のデータセットなら、より大きいcutoffも安全に通る。)
    half_box = float(np.asarray(cells)[:, [0, 1, 2], [0, 1, 2]].min()) / 2
    if args.large_cutoff >= half_box:
        raise ValueError(
            f"--large-cutoff ({args.large_cutoff}) must be strictly less than half the box "
            f"({half_box:.4f} A) -- otherwise the same atom pair gets connected through 2+ "
            f"periodic images at once (the test33/48 duplicate-periodic-image bug). Use a smaller "
            f"cutoff, or rebuild the dataset with `prepare --replicate N` to enlarge the box.")
    device = device_for(args.device)
    output = args.output.resolve() if args.resume else new_output(args.output)
    checkpoint = output / "checkpoint.pt"
    if args.resume and not checkpoint.is_file():
        raise ValueError("--resume requires output/checkpoint.pt")

    # フレームを学習用/検証用に分割(先頭split個が学習、残りが検証)。
    split = max(1, int(len(positions) * (1 - args.validation_fraction)))
    # この実行の設定をすべて記録しておく。--resumeで再開する際、設定が
    # 完全一致するかどうかの検証にも使う(途中でハイパラを変えて再開させない)。
    settings = dict(
        dataset_sha256=meta["sha256"], metadata_sha256=digest(args.dataset / "metadata.json"),
        cutoff=args.cutoff, large_cutoff=args.large_cutoff,
        irreps_hidden=args.irreps_hidden, irreps_edge=args.irreps_edge,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        seed=args.seed, split_frame=split, device=str(device), log_every=args.log_every,
        warm_start_sha256=(digest(args.warm_start) if args.warm_start else None),
    )
    config = architecture(len(meta["species"]), args.cutoff, args.irreps_hidden, args.irreps_edge)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_time_model(config, device)
    if not args.resume and args.warm_start is not None:
        # test37/36で学習済みのplainチェックポイントからウォームスタートする場合。
        # アーキテクチャ(species数・cutoffなど)が一致していることを確認してから、
        # 共有できる重みだけをコピーする。
        source_ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if source_ck.get("architecture") != config:
            raise ValueError("--warm-start checkpoint architecture does not match this dataset")
        warm_start_from_plain_checkpoint(model, source_ck["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, completed = [], 0
    if args.resume:
        # 既存のtest55チェックポイントから学習状態(重み・optimizer・乱数状態・履歴)を復元する。
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("format") != FORMAT or ck["settings"] != settings:
            raise ValueError("Resume settings/data/device differ from checkpoint")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        history, completed = ck["history"], ck["completed_updates"]
    downselect = DownselectEdges(cutoff=args.cutoff)

    def rattle_at(data, sigma_value):
        # NequIP_TimeEmbed broadcasts a single scalar `t` to every node in the call
        # (`h_node_t.expand(n_nodes, -1)`); it does not support one sigma per graph
        # within a batch the way RattleParticles' own per-graph sigma mechanism
        # assumes (which is also silently dropped by PyG's Batch storage on a
        # multi-graph batch). So every graph in a training step is rattled with the
        # SAME drawn sigma, matching what the model can actually condition on; sigma
        # still varies step to step across the full [sigma_min, sigma_max] range
        # over the course of training.
        return RattleParticles(sigma_min=sigma_value, sigma_max=sigma_value)(data)
    deadline = time.monotonic() + args.time_budget_hours * 3600  # HPCジョブの時間切れ前に安全に停止するための締切

    def save():
        # 学習の途中経過(重み・optimizer状態・乱数状態・履歴)をチェックポイントに、
        # 進捗の要約をtraining.jsonに、それぞれ書き出す。
        save_checkpoint(checkpoint, dict(
            format=FORMAT, architecture=config, settings=settings,
            model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), rng=rng_state(), history=history,
            completed_updates=completed, requested_updates=args.updates,
            dataset_metadata=meta, large_cutoff=args.large_cutoff,
            start_positions_angstrom=np.asarray(positions[0]).copy(),
            cell_angstrom=np.asarray(cells[0]).copy(), scientific_caveat=CAVEAT,
        ))
        save_json(output / "training.json", dict(
            completed_updates=completed, requested_updates=args.updates,
            history=history, scientific_caveat=CAVEAT,
        ))

    print(f"train (test55, sigma-conditioned): device={device}, frames={len(positions)}, "
          f"train/validation={split}/{len(positions) - split}, "
          f"warm_start={'yes' if args.warm_start else 'no'}", flush=True)
    # --- 学習ループ本体 ---
    for step in range(completed + 1, args.updates + 1):
        if STOP or time.monotonic() >= deadline:
            # 中断シグナルか時間切れなら、その場でチェックポイントを保存して終了コード75を返す
            # (HPCジョブスケジューラに「再投入すれば続きから再開できる」ことを伝える慣習的な値)。
            save()
            print("Training paused; resume with --resume", flush=True)
            return 75
        model.train()
        # 学習フレームからランダムにbatch_size個選び、
        indices = np.random.randint(split, size=args.batch_size)
        # このステップで使うsigmaを[sigma_min, sigma_max]から1つだけサンプルする
        # (バッチ内の全グラフに同じsigmaを使う。理由は下のコメント参照)。
        sigma_value = float(np.random.uniform(args.sigma_min, args.sigma_max))
        batch = Batch.from_data_list([
            graph(positions[i], cells[i], meta["type_ids"], args.large_cutoff, device)
            for i in indices
        ])
        # ノイズを加えてから(rattle_at)、モデルのcutoffに絞り込む(downselect)。
        batch = downselect(rattle_at(batch, sigma_value))
        # モデルに渡すt(正規化されたsigma)を計算し、順伝播・損失計算・逆伝播。
        t = torch.tensor([sigma_value / args.sigma_max], device=device, dtype=batch.pos.dtype)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, t)
        # 目的関数: 加えたノイズ(batch.dx)をどれだけ正確に予測できたかのMSE(denoising score matching)。
        loss = torch.nn.functional.mse_loss(prediction, batch.dx)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        completed = step
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            # 定期的に検証データでの損失もログに出す。乱数状態を退避・復元することで、
            # 検証評価が学習側の乱数列(データ選択やノイズ)に影響を与えないようにしている。
            model.eval()
            saved_rng = rng_state()
            losses = []
            valid_rng = np.random.default_rng(args.seed + 1)
            for i in range(split, min(split + 4, len(positions))):
                valid_sigma = float(valid_rng.uniform(args.sigma_min, args.sigma_max))
                valid = graph(positions[i], cells[i], meta["type_ids"], args.large_cutoff, device)
                valid = downselect(rattle_at(valid, valid_sigma))
                valid_t = torch.tensor([valid_sigma / args.sigma_max], device=device, dtype=valid.pos.dtype)
                with torch.no_grad():
                    losses.append(torch.nn.functional.mse_loss(
                        model(valid, valid_t), valid.dx
                    ).item())
            restore_rng(saved_rng)
            val_loss = float(np.mean(losses))
            row = dict(step=step, train_mse_A2=float(loss.detach().cpu()), valid_mse_A2=val_loss)
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_every == 0:
            save()
    save()
    print(f"Checkpoint: {checkpoint}")


@torch.no_grad()
def generate(args):
    # --- セットアップ: チェックポイント読み込みと出力先準備 ---
    if args.start_sigma < 0.001:
        raise ValueError("start-sigma must be at least 0.001 angstrom")
    device = device_for(args.device)
    model, ck = checkpoint_time_model(args.checkpoint, device)
    meta, cell = ck["dataset_metadata"], ck["cell_angstrom"]
    cutoff = ck["architecture"]["cutoff_angstrom"]
    sigma_max_train = ck["settings"]["sigma_max"]  # tの正規化に使う、学習時のsigma_max
    output = args.output.resolve() if args.resume else new_output(args.output)
    settings = dict(
        checkpoint_sha256=digest(args.checkpoint), reverse_steps=args.reverse_steps,
        deterministic_steps=args.deterministic_steps, start_sigma=args.start_sigma,
        sigma_min=args.sigma_min, thermal_scale=args.thermal_scale, init=args.init,
        seed=args.seed, device=str(device), cutoff_angstrom=cutoff,
    )
    total = args.reverse_steps
    state_path = output / "generation_restart.pt"
    if args.resume:
        # 中断していた生成を再開する場合: 直前の座標・乱数状態を復元し、
        # 既存のpositions.npyメモリマップを追記モードで開く。
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("Generation resume settings differ; use a new output directory")
        pos, completed = state["positions"].to(device), state["step"]
        restore_rng(state["rng"])
        trajectory = np.lib.format.open_memmap(output / "positions.npy", mode="r+")
    else:
        # 新規に生成を開始する場合: 学習データセットの最初のフレーム(訓練時に
        # 保存しておいたstart_positions_angstrom)を出発点にする。
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        pos = torch.tensor(ck["start_positions_angstrom"], dtype=torch.float32, device=device)
        if args.init == "crystal-noised":
            # ここが本題: 以下の焼きなましループは、posが「sigma=start_sigmaの拡散済み状態」
            # であることを前提にした更新式(variance_drop, score_stepなど)を使っているのに、
            # --init crystal(デフォルト)ではposは実際には一切ノイズを受けていない綺麗な結晶
            # 座標のまま渡されてしまう(ループ内で暗黙にノイズが注入されるのを待つだけ)。
            # crystal-noisedでは、学習時のRattleParticles(sigma_min=sigma_max=start_sigma)と
            # 全く同じノイズモデル(eps~N(0,1); pos += start_sigma*eps)を明示的に一度適用して
            # から焼きなましを始める。これで「拡散過程で実際にsigma=start_sigmaまで拡散させた
            # ノイズ構造」から出発する、本来の意味でのreverse diffusionになる。
            pos = pos + args.start_sigma * torch.randn_like(pos)
        completed = 0
        trajectory = np.lib.format.open_memmap(
            output / "positions.npy", mode="w+", dtype=np.float32, shape=(total + 1, len(pos), 3)
        )
        trajectory[0] = pos.cpu().numpy()

    # --- sigmaスケジュールの構築 ---
    # start_sigmaからsigma_minまで、対数スケールでreverse_steps段階に等分割する
    # (test37_variant_experiment.pyのlangevin ablationにおける
    # geomspace(0.1, 0.003, 20)と同じ発想: 段数を細かく刻むほど、1ステップあたりの
    # variance_drop(=分散の減少量)が小さくなり、歩幅も自然に小さくなる)。
    # 最後にsigma=0のダミー要素を足しておき、最終ステップのnext_sigmaが0になるようにする。
    first_sigma = max(args.start_sigma, args.sigma_min)
    if args.reverse_steps == 1:
        sigma_schedule = torch.tensor([first_sigma, 0.0], device=device)
    else:
        positive_levels = torch.logspace(
            np.log10(first_sigma), np.log10(args.sigma_min), args.reverse_steps, device=device,
        )
        sigma_schedule = torch.cat((positive_levels, positive_levels.new_zeros(1)))
    # 全reverse_stepsのうち、後半deterministic_steps回は確率的更新をやめて
    # 決定論的なDDIM風更新に切り替える(「ノイズありで大まかに→仕上げは決定論的に」という設計)。
    stochastic_steps = args.reverse_steps - args.deterministic_steps
    if stochastic_steps <= 0:
        # --deterministic-steps >= --reverse-stepsだと、確率的(annealed-Langevin)フェーズが
        # 0回になり、全ステップが決定論的DDIM更新になる(それ自体は有効な設定だが、
        # reverse_stepsは変えていないので、スケジュールの段数=分解能は変わらない。
        # 「段数を増やしたい」場合はdeterministic-stepsではなくreverse-steps自体を
        # 増やす必要がある、と誤解しやすいのでここで明示的に知らせる)。
        print(f"NOTE: --deterministic-steps ({args.deterministic_steps}) >= --reverse-steps "
              f"({args.reverse_steps}): every one of the {args.reverse_steps} steps will run "
              f"the deterministic DDIM branch (stochastic_steps=0, no annealed-noise steps at "
              f"all). This does NOT make the schedule finer or longer -- to do that, increase "
              f"--reverse-steps itself.", flush=True)
    deadline = time.monotonic() + args.time_budget_hours * 3600

    def save():
        # 生成中の座標(positions.npy)と再開用チェックポイント、進捗JSONを保存する。
        trajectory.flush()
        save_checkpoint(state_path, dict(
            settings=settings, step=completed, positions=pos.detach().cpu(), rng=rng_state(),
        ))
        save_json(output / "generation.json", dict(
            completed_steps=completed, requested_steps=total, valid_frames=completed + 1,
            complete=completed == total, length_unit="angstrom", settings=settings,
            cell_angstrom=np.asarray(cell).tolist(), dataset_metadata=meta,
            is_equilibrium_trajectory=False, scientific_caveat=CAVEAT,
        ))

    # --- 生成(逆拡散)ループ本体 ---
    for step in range(completed, total):
        if STOP or time.monotonic() >= deadline:
            save()
            print("Generation paused; resume with --resume", flush=True)
            return 75
        sigma, next_sigma = sigma_schedule[step], sigma_schedule[step + 1]
        # 現在の座標からグラフを再構築し(distance依存のedge_attrを使うため、
        # 座標が動くたびにグラフを作り直す必要がある)、モデルにt=sigma/sigma_max_trainを渡して
        # 「このノイズレベルにおける、加えられたノイズの予測」を得る。
        data = graph(pos.cpu().numpy(), cell, meta["type_ids"], cutoff, device)
        t = torch.full((1,), float(sigma / sigma_max_train), device=device, dtype=pos.dtype)
        predicted = model(data, t)
        if step < stochastic_steps:
            # Annealed-Langevin / variance-exploding reverse-diffusion step: the
            # step size and injected-noise scale both come directly from how much
            # variance this sigma interval removes, not from a hand-picked constant.
            # variance_dropは「このsigma区間で本来除去されるはずのノイズ分散」。
            # これをsigma**2で割ったscore_stepが、モデル予測に対する重み(歩幅)になる。
            # sigma, next_sigmaが近い(=スケジュールが細かい)ほどscore_stepは小さくなり、
            # legacyサンプラーのような「毎回フルサイズで補正+フルサイズで再ノイズ」を避けられる。
            variance_drop = torch.clamp(sigma.square() - next_sigma.square(), min=0.0)
            score_step = variance_drop / sigma.square()
            pos = pos - score_step * predicted  # モデル予測方向への小さいドリフト(スコアに沿った移動)
            # 揺動散逸的にバランスの取れたノイズを注入する(sqrt(variance_drop)倍)。
            # thermal_scaleでこのノイズの強さを実験的に調整できる。
            pos = pos + torch.randn_like(pos) * torch.sqrt(variance_drop) * args.thermal_scale
        else:
            # DDIM-style deterministic tail for the final polish steps.
            # 最後の仕上げ区間はノイズを加えず、決定論的に少しずつ補正していく。
            ddim_step = 1.0 - next_sigma / sigma
            pos = pos - ddim_step * predicted
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite generated positions")
        completed = step + 1
        trajectory[completed] = pos.cpu().numpy()
        if completed % args.checkpoint_every == 0:
            save()
            print(f"generation {completed}/{total} (sigma={float(sigma):.4f})", flush=True)
    save()
    # 生成が完了したら、最終フレームをase.Atomsに変換してextxyzファイルとしても書き出す。
    atoms = atoms_from_meta(pos.cpu().numpy(), cell, meta)
    atoms.wrap()
    ase.io.write(output / "final.extxyz", atoms)
    if args.trajectory_stride > 0:
        # --trajectory-stride>0なら、結晶構造がステップを追って組み上がっていく様子を
        # 見られるよう、positions.npy(全ステップの座標)をマルチフレームのextxyzに
        # 書き出す(可視化ソフト(OVITO/VMD等)でそのままアニメーション再生できる)。
        write_trajectory_extxyz(output, meta, cell, args.trajectory_stride)
    print(f"Generated {total + 1} frames: {output}")


# --- test55-specific code: CGMD (new, ported from test37.py's own export()/md()) ---------------

def export(args):
    # test37.pyのexport()と同じ役割: チェックポイントを「スコア力場」バンドル
    # (cg_score_forcefield.pt/.json)と、その出発構造(cg_start.data、LAMMPS形式)として
    # 書き出す。test37の無条件モデルと違い、このモデルはsigma条件付きなので、バンドルに
    # sigma_ref(後でmd()がt=sigma_ref/sigma_max_trainとして使う固定値)も一緒に保存する。
    model, ck = checkpoint_time_model(args.checkpoint, torch.device("cpu"))
    meta, cell = ck["dataset_metadata"], np.asarray(ck["cell_angstrom"])
    if not np.allclose(cell, np.diag(np.diag(cell))):
        raise ValueError("Existing LAMMPS callback supports orthorhombic fixed boxes only")
    output = new_output(args.output)
    atoms = atoms_from_meta(ck["start_positions_angstrom"], cell, meta)
    atoms.wrap()
    lines = ["test55 SiO4-centroid CG start (species order in bundle JSON)", "", f"{len(atoms)} atoms",
             f"{len(meta['species'])} atom types", ""]
    lines += [f"0 {length:.16g} {axis}lo {axis}hi" for length, axis in zip(np.diag(cell), "xyz")]
    lines += ["", "Masses", ""]
    lines += [f"{i + 1} {s['mass_amu']}" for i, s in enumerate(meta["species"])]
    lines += ["", "Atoms # atomic", ""]
    lines += [f"{i + 1} {meta['type_ids'][i] + 1} {p[0]:.16g} {p[1]:.16g} {p[2]:.16g}"
              for i, p in enumerate(atoms.positions)]
    (output / "cg_start.data").write_text("\n".join(lines) + "\n")
    sigma_max_train = ck["settings"]["sigma_max"]
    bundle = dict(format="test55-score-forcefield", format_version=1, kind="coarse_grained_tetrahedron",
                  architecture=ck["architecture"], model_state_dict=ck["model_state_dict"],
                  species_by_id=torch.tensor(meta["type_ids"]),
                  atomic_numbers_by_id=torch.tensor(atoms.numbers), num_atoms=len(atoms),
                  cell_angstrom=torch.tensor(cell), species=meta["species"], condition=meta["condition"],
                  temperature_k=float(meta["condition"]["temperature_k"]), sigma_ref_angstrom=args.sigma_ref,
                  sigma_max_train_angstrom=sigma_max_train, force_clip_ev_per_angstrom=args.force_clip,
                  suggested_timestep_ps=0.0001, suggested_damping_ps=0.1,
                  suggested_data_file="cg_start.data", conservative=False, provides_energy=False,
                  provides_virial=False, scientific_status="experimental_unvalidated",
                  scientific_caveat=CAVEAT, source_checkpoint_sha256=digest(args.checkpoint))
    save_checkpoint(output / "cg_score_forcefield.pt", bundle)
    public = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in bundle.items() if k != "model_state_dict"}
    save_json(output / "cg_score_forcefield.json", public)
    print(f"CGMD force bundle (sigma_ref={args.sigma_ref} A, t={args.sigma_ref / sigma_max_train:.4g}): {output}")


def md(args):
    # test37.pyのmd()と同じ構造(ASE Langevin + ScoreCalculator)。違いはただ1点:
    # test37の無条件モデルはmodel(data)だけで呼べたが、NequIP_TimeEmbedはtを要求するので、
    # t = sigma_ref / sigma_max_train をMD全体を通じて固定値として渡す(焼きなましはしない
    # -- このMDは「ある1つのノイズ/温度スケールでの力学」を近似するだけで、generate()の
    # ような段階的アニーリングとは別物)。
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    import csv
    device = device_for(args.device)
    model, ck = checkpoint_time_model(args.checkpoint, device)
    meta, cell = ck["dataset_metadata"], ck["cell_angstrom"]
    cutoff = ck["architecture"]["cutoff_angstrom"]
    sigma_max_train = ck["settings"]["sigma_max"]
    t_fixed = torch.tensor([args.sigma_ref / sigma_max_train], device=device, dtype=torch.float32)
    temperature = float(meta["condition"]["temperature_k"])
    output = new_output(args.output)
    atoms = atoms_from_meta(ck["start_positions_angstrom"], cell, meta)

    class ScoreCalculator(Calculator):
        implemented_properties = ["forces"]

        def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            with torch.no_grad():
                data = graph(self.atoms.positions, np.asarray(self.atoms.cell), meta["type_ids"],
                             cutoff, device)
                force = -units.kB * temperature * model(data, t_fixed) / args.sigma_ref**2
                force -= force.mean(dim=0, keepdim=True)
                force *= (args.force_clip / force.norm(dim=1, keepdim=True).clamp_min(1.e-12)).clamp(max=1)
                force -= force.mean(dim=0, keepdim=True)
                if not torch.isfinite(force).all():
                    raise RuntimeError("Non-finite score force")
                self.results["forces"] = force.cpu().numpy().astype(float)

    atoms.calc = ScoreCalculator()
    atoms.set_constraint(FixCom())
    random = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=random)
    Stationary(atoms)
    dyn = Langevin(atoms, timestep=args.timestep_fs * units.fs, temperature_K=temperature,
                   friction=1 / (args.damping_ps * 1000 * units.fs), rng=random, fixcm=False)
    started = time.monotonic()
    completed = 0
    with (output / "md.extxyz").open("w") as trajectory, \
            (output / "thermo.csv").open("w", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_ps", "temperature_k", "max_force_eV_A", "elapsed_seconds"])

        def record():
            ase.io.write(trajectory, atoms, format="extxyz", write_results=False)
            trajectory.flush()
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            elapsed = time.monotonic() - started
            writer.writerow([completed, completed * args.timestep_fs / 1000, atoms.get_temperature(), fmax, elapsed])
            save_json(output / "progress.json", dict(completed_steps=completed, requested_steps=args.steps,
                      elapsed_seconds=elapsed, temperature_k=atoms.get_temperature()))
            ase.io.write(output / "latest.extxyz", atoms, write_results=False)
            print(f"MD {completed}/{args.steps}: T={atoms.get_temperature():.2f} K", flush=True)

        record()
        while completed < args.steps:
            if STOP or time.monotonic() - started >= args.time_budget_hours * 3600:
                print("MD paused; latest.extxyz has positions and velocities, not an exact RNG restart")
                return 75
            increment = min(args.save_every, args.steps - completed)
            dyn.run(increment)
            completed += increment
            record()
    ase.io.write(output / "final.extxyz", atoms, write_results=False)
    save_json(output / "run_metrics.json", dict(completed_steps=completed, time_ps=completed * args.timestep_fs / 1000,
              temperature_k=temperature, sigma_ref_angstrom=args.sigma_ref,
              force_clip_eV_A=args.force_clip, timestep_fs=args.timestep_fs, damping_ps=args.damping_ps,
              seed=args.seed, device=str(device), elapsed_seconds=time.monotonic() - started,
              checkpoint_sha256=digest(args.checkpoint), scientific_caveat=CAVEAT))
    print(f"MD complete: {output}")


def write_trajectory_extxyz(output, meta, cell, stride):
    # positions.npy(generate()が保存した全ステップの座標)を読み直し、strideおきの
    # フレーム(+最終フレームは必ず含める)を1つのマルチフレームextxyzに書き出す。
    # 各フレームのase.Atoms.infoに"step"を残しておくので、可視化時にステップ番号が分かる。
    trajectory = np.load(output / "positions.npy", mmap_mode="r")
    path = output / "trajectory.extxyz"
    if path.exists():
        path.unlink()
    steps = list(range(0, len(trajectory), stride))
    if steps[-1] != len(trajectory) - 1:
        steps.append(len(trajectory) - 1)
    for step in steps:
        atoms = atoms_from_meta(np.asarray(trajectory[step]), cell, meta)
        atoms.wrap()
        atoms.info["step"] = step
        ase.io.write(path, atoms, append=True)
    print(f"Trajectory ({len(steps)} frames, stride={stride}): {path}")


def parser():
    # コマンドラインインターフェース定義: "prepare"・"train"・"generate"の3つのサブコマンドを持つ。
    # ("prepare"はtest50独自の追加。train/generateはtest38.pyと同一。)
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("prepare", help="SiO2 SiO4-tetrahedron-centroid CG dataset: one bead per Si+its 4 nearest O")
    p.add_argument("--reference-frames", type=Path, default=ROOT / "simu_data" / "reference_frames.npz",
                   help="npz with 'positions' (frames,atoms,3), 'cell_lengths' (frames,3), "
                        "'numbers' (atoms,) arrays in Angstrom; defaults to the bundled "
                        "simu_data/reference_frames.npz (test47/48/49's own beta-cristobalite "
                        "NVT reference)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--temperature-k", type=positive, default=300.0)
    p.add_argument("--stride", type=count, default=1)
    p.add_argument("--replicate", type=count, default=1,
                   help="Tile the box N x N x N (exact periodic copies, not independent thermal "
                        "samples) to safely enlarge it for a bigger --cutoff at train time -- e.g. "
                        "--replicate 2 turns the 13.573 A box into 27.146 A (half-box 13.573 A), "
                        "safely covering --cutoff 8. Default 1 (no tiling, unchanged behavior).")
    p.set_defaults(handler=prepare)

    p = sub.add_parser("train", help="sigma-conditioned NequIP_TimeEmbed + RattleParticles(sigma_min, sigma_max) + displacement MSE")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, default=None,
                   help="Plain (non-time-conditioned) NequIP checkpoint, same architecture as "
                        "this dataset, to initialize shared weights from")
    p.add_argument("--updates", type=count, default=50000)  # 勾配更新の総回数(元は6000→20000→50000と要望に合わせて引き上げ)
    p.add_argument("--batch-size", type=count, default=16)
    p.add_argument("--learning-rate", type=positive, default=2.e-4)
    # test38の元のデフォルトは10.0(粘土系の大きな箱用)。このSiO2結晶の素の箱は13.573A
    # (半箱~6.7865A)しかなく、それを超えるcutoffは周期像重複バグ(test33/48)を踏むため、
    # 以前は安全な範囲内で6.5/6.7にとどめていた。今回cutoff=8の要望に対応するため、
    # `prepare --replicate 2`で箱を2x2x2タイル化する前提に変更: 箱が27.146A(半箱13.573A)
    # になるので、cutoff=8/large-cutoff=8.2は余裕を持って安全(半箱まで5.57A以上の余裕)。
    # train()側にもガードを追加済みで、--replicateしていない小さい箱のデータセットに
    # このデフォルトをうっかり使うと、黙って壊れる代わりに明示的なエラーで弾かれる。
    p.add_argument("--cutoff", type=positive, default=8.0)  # モデルが実際に使うグラフcutoff(--replicate 2の箱が前提)
    p.add_argument("--large-cutoff", type=positive, default=8.2)  # ノイズを加える前に候補として作っておくcutoff(cutoff以上必須)
    p.add_argument("--sigma-min", type=positive, default=0.001)
    # 元は0.75(test38/DM2共通のデフォルト)。ノイズ最大値を増やす要望に合わせて2倍の1.5に。
    # (Si-Si最近接距離が~2.97Aなので、sigma=1.5は既に「原子がほぼ完全にかき乱された」
    # 領域までσレンジを広げることになる。)
    p.add_argument("--sigma-max", type=positive, default=1.5)  # 学習時に使うノイズ幅の上限(生成時のt正規化にも使われる)
    # test50独自の追加(test38にはこの2つのCLI引数はなく、architecture()内にl<=1隠れ層/
    # l<=2エッジでハードコードされている)。要望により、l=4まで広げた構成
    # (test47/48のl<=5構成を1段階切り詰めたもの)をデフォルトに変更 -- 追加のフラグなしで
    # l=4になる。l_maxを上げるほどe3nnのテンソル積のパス数・中間テンソルが急増しGPU
    # メモリを多く使う(test49のCUDA OOMと同じ理由)ので、OOMが出たら--batch-sizeを
    # 下げること(元のtest38相当のl<=1/l<=2に戻したい場合は明示的に
    # --irreps-hidden "64x0e + 32x1e" --irreps-edge "4x0e + 4x1e + 2x2e" を渡す)。
    p.add_argument("--irreps-hidden", type=str, default="64x0e + 32x1e + 16x2e + 8x3e + 4x4e")
    p.add_argument("--irreps-edge", type=str, default="4x0e + 4x1e + 2x2e + 2x3e + 1x4e")
    p.add_argument("--validation-fraction", type=positive, default=0.1)
    p.add_argument("--log-every", type=count, default=100)
    p.set_defaults(handler=train)

    p = sub.add_parser("generate", help="annealed-Langevin / variance-exploding reverse-SDE sampler with a DDIM polish tail")
    p.add_argument("--init", choices=("crystal", "crystal-noised"), default="crystal",
                   help="'crystal' (default, test38's original behavior): start from the clean "
                        "training frame and let the annealed-Langevin loop's own noise injection "
                        "be the only corruption applied. 'crystal-noised': explicitly corrupt the "
                        "clean frame with the SAME noise model training uses "
                        "(pos += start_sigma * randn, i.e. RattleParticles(sigma=start_sigma)) "
                        "before the reverse loop starts, so generation genuinely begins from a "
                        "forward-diffused noisy structure at sigma=start_sigma, not a clean one "
                        "the loop only pretends is already noised.")
    p.add_argument("--reverse-steps", type=count, default=300)  # sigmaスケジュールの段数(細かいほど1歩あたりの補正が小さくなる)
    p.add_argument("--deterministic-steps", type=nonnegative_count, default=30)  # 末尾何ステップをDDIM風の決定論的更新にするか
    p.add_argument("--start-sigma", type=positive, default=0.75)  # 生成開始時のノイズレベル
    p.add_argument("--sigma-min", type=positive, default=0.001)  # 確率的ステップを終える下限ノイズレベル
    p.add_argument("--thermal-scale", type=positive, default=1.0)  # 注入ノイズの強さを調整する倍率
    p.add_argument("--trajectory-stride", type=nonnegative_count, default=1,
                   help="Write every Nth generation step (plus the final step) to "
                        "output/trajectory.extxyz as one multi-frame trajectory, so the "
                        "structure's step-by-step formation can be viewed (e.g. in OVITO/VMD). "
                        "0 disables this and only writes final.extxyz.")
    p.set_defaults(handler=generate)

    # test55独自の追加(test50にはない。test37.pyのexport/mdをそのまま移植): CGMD。
    p = sub.add_parser("export", help="Bundle the checkpoint + cg_start.data for the LAMMPS/CGMD callback")
    p.set_defaults(handler=export)
    p = sub.add_parser("md", help="Experimental fixed-cell ASE Langevin score MD (t=sigma_ref/sigma_max_train fixed)")
    p.add_argument("--steps", type=count, default=1000)
    p.add_argument("--timestep-fs", type=positive, default=0.1)
    p.add_argument("--damping-ps", type=positive, default=0.1)
    p.add_argument("--save-every", type=count, default=100)
    p.set_defaults(handler=md)

    # train/generate/export/md共通の引数(出力先・デバイス・乱数シード・時間予算・
    # 再開オプションなど)。("prepare"はここでは扱わない -- --outputは既にprepare自身の
    # サブパーサーで定義済みで、device/seed/time-budget/resumeのような概念はデータセット
    # 準備には存在しないため。)
    for name, p in sub.choices.items():
        if name == "prepare":
            continue
        p.add_argument("--output", type=Path, required=True)
        if name in ("generate", "export", "md"):
            p.add_argument("--checkpoint", type=Path, required=True)
        if name in ("train", "generate", "md"):
            p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
            p.add_argument("--seed", type=int, default=1337)
            p.add_argument("--time-budget-hours", type=positive, default=11.5)  # この時間を超えたら自動で中断・保存する(HPCジョブの壁時計制限対策)
        if name in ("train", "generate"):
            p.add_argument("--resume", action="store_true")
            p.add_argument("--checkpoint-every", type=count, default=25)
        if name in ("export", "md"):
            # test37.pyと同じ: sigma_refはシステム・目的ごとに意味のある値が違うので、
            # デフォルトを設けずユーザーに明示的に選ばせる。
            p.add_argument("--sigma-ref", type=positive, required=True,
                           help="Fixed noise/temperature scale (Angstrom) at which the model's "
                                "denoising output is reinterpreted as a force (t = sigma_ref / "
                                "sigma_max_train). No sampler-wide default -- pick a value within "
                                "the checkpoint's trained [sigma_min, sigma_max] range.")
            p.add_argument("--force-clip", type=positive, default=10.0)
    return root


def main():
    # SIGTERM/SIGINTを受けたらrequest_stop()でSTOPフラグを立てるようにしてから、
    # 指定されたサブコマンド(train/generate)のハンドラを実行する。
    import signal
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args = parser().parse_args()
    print(CAVEAT, flush=True)
    return args.handler(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
