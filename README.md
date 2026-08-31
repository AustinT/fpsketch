# fpsketch

Compress molecular count fingerprints to a fixed low dimension while
(approximately) preserving Tanimoto similarity as a plain dot product.

Given two count fingerprints `x`, `x'` (e.g. Morgan fingerprints with counts,
`{feature_id: count}`), the standard chemistry similarity metric is the
min-max Tanimoto:

```
T_MM(x, x') = sum_i min(x_i, x'_i) / sum_i max(x_i, x'_i)
```

`T_MM` isn't a dot product, so you can't drop count fingerprints straight into
models (nearest-neighbor search, GPs, kernel methods, ...) that expect a plain
inner product. fpsketch sketches count fingerprints into a fixed-width dense
vector `s = encode(x)` such that

```
T_DP(s, s') = s.s' / (||s||^2 + ||s'||^2 - s.s')  ≈  T_MM(x, x')
```

i.e. an ordinary dot product on the sketch approximates `T_MM` on the
original fingerprints.

## Why this works

A unary encoding turns each count into a set of indicators,
`psi(x)_{i,k} = 1[x_i > k]` for `k = 0 .. x_i - 1`. Since
`min(u, v) = sum_k 1[u > k] * 1[v > k]`, this makes `T_DP(psi(x), psi(x'))`
exactly equal to `T_MM(x, x')` -- no approximation yet, just a reformulation.
fpsketch then applies a CountSketch (hashing each `(feature_id, level)` pair
into one of `m` signed buckets) to that unary expansion, which keeps the
dot product unbiased while collapsing it to a fixed, low dimension.

## Install

```
pip install fpsketch          # encode_sparse only, numpy-only
pip install fpsketch[chem]    # + encode_mols, pulls in rdkit
```

## Quickstart

```python
from fpsketch import encode_sparse

# Sparse count fingerprints you already have.
fps = [{1: 1, 2: 2, 3: 3}, {4: 4, 5: 5, 6: 6}]
sketch = encode_sparse(fps, dim=2048, seed=0)

# T_DP as a plain dot product / normalized similarity.
G = sketch @ sketch.T
sq = (sketch**2).sum(axis=1)
similarity = G / (sq[:, None] + sq[None, :] - G)
```

```python
from rdkit import Chem
from fpsketch import encode_mols

mols = [Chem.MolFromSmiles(s) for s in smiles_list]
sketch = encode_mols(mols, dim=2048, seed=0)  # defaults to a Morgan(radius=2) generator
```

If you already have your counts vectorized as a COO sparse array (molecules x
features), `encode_coo` skips the per-molecule dict traversal `encode_sparse`
does internally:

```python
from scipy.sparse import coo_array
from fpsketch import encode_coo

counts = coo_array(...)  # shape (n_molecules, n_features)
sketch = encode_coo(counts, dim=2048, seed=0)
```

Two sketches are only comparable if built with the same `seed`.

## Choosing `dim` and `num_blocks`

`dim=2048` is a strong default -- at that width the sketch already beats the
best possible elementwise approximation of `T_MM`. For extra safety margin,
`dim` at 2-4x the fingerprint's effective (unfolded) dimension is a reasonable
range to sweep. `num_blocks` (default 4) splits `dim` into that many disjoint
sub-sketches averaged together; it trades a small amount of raw accuracy for
better tail concentration across single-draw sketches, which matters when a
sketch is computed once and fed straight into a downstream model (e.g. a GP)
rather than averaged over many random seeds.

## The `scale` parameter

By default (`scale=True`), the output is divided by `sqrt(num_blocks)` so
that raw dot products and squared norms directly approximate the true,
unnormalized dot product / count mass of the original fingerprints -- useful
if you compare sketches built with different `num_blocks`, or use the sketch
for anything beyond the `T_DP` ratio above (cosine similarity, nearest
neighbors on raw dot product, etc). That factor cancels out of the `T_DP`
ratio itself, so if you only ever compute Tanimoto similarity through that
ratio, `scale=False` is equivalent and skips one pass over the output array.

## Performance note

Hashing is vectorized with numpy (a pure-numpy splitmix64 mixer, not
`hashlib` per element) rather than hashing one `(feature, count-level)` pair
at a time -- see `src/fpsketch/sketching.py` for details.

## Development

```
uv sync --extra chem
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pre-commit install  # run the above automatically on each commit
```

## Releasing

Versions are derived from git tags (`hatch-vcs`); there is no version to bump
by hand. Pushing a tag matching `v*` (e.g. `v0.2.0`) triggers
`.github/workflows/publish.yml`, which builds and publishes to PyPI via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) -- no API
token needed, but the `pypi` environment must be configured as a trusted
publisher for this repo in the PyPI project settings first.

## License

MIT, see [LICENSE](LICENSE).
