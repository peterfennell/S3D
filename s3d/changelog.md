changelog for pys3d
---

07-02-2018
- exclude line 452-453 in `_evaluate` function, thanks to lisette. [diff](https://github.com/zhiyzuo/S3D/commit/fabf0aeb96b46e297cc86d5d72586aa977ee28c8#diff-277225d258ab602f1bd716fc7c4c60e9L452)
- set the minimal number of color palettes to be 3
- set the color list for network visualization using list comprehension to maintain the node order
- use different metrics for classification/regression

06-28-2018
- change the way heatmaps produced. see [quickstart.ipynb](quickstart.ipynb) section 4.2 for more details
