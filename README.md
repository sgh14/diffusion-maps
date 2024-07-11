# diffusion-maps
Diffusion maps algorithm for manifold learning (dimensionality reduction) with Python and optimized with Numba.

## Example

```python
from sklearn.datasets import make_swiss_roll
from DiffusionMaps import DiffusionMaps
from aux_functions import get_sigma

X, color = make_swiss_roll(1000)
model = DiffusionMaps(sigma=get_sigma(X, 0.005), n_components=2, step=1, alpha=1)
X_red = model.fit_transform(X)
```
