changelog for pys3d
---

08-06-2018
- error message for misuse of `visualize_s3d_model` for `dim=1`

08-02-2018
- update visualization - limit number of features in steps bar chart

07-28~07-29-2018
- update visualization for variance explanation

07-25-2018
- add visualization for 1d frequency/intensity map

07-02-2018
- exclude line 452-453 in `_evaluate` function, thanks to lisette. [diff](https://github.com/zhiyzuo/S3D/commit/fabf0aeb96b46e297cc86d5d72586aa977ee28c8#diff-277225d258ab602f1bd716fc7c4c60e9L452)
- set the minimal number of color palettes to be 3
- set the color list for network visualization using list comprehension to maintain the node order
- use different metrics for classification/regression

06-28-2018
- change the way heatmaps produced. see [quickstart.ipynb](quickstart.ipynb) section 4.2 for more details
