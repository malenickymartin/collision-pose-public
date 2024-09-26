# collision-pose-pub
Object Pose Estimation from Images with Geometrical and Physical Consistency

# Install
Clone this repository and unstall requirements using:
`pip install -r setup/requirements.txt`

# Usage
If you plan to use datasets from the BOP Challenge, do the following for each of them:
1. Download your meshes to the data/meshes directory, or use the following example script to download meshes from [BOP Website](https://bop.felk.cvut.cz/datasets/):\
`sh setup/download_meshes.sh {dataset name}`
2. Make a convex decomposition of the meshes and store it in the data/meshes_decomp folder, you can use script:\
`python3 setup/mesh_decomposition.py {dataset name}`
3. _Optional_: Some scripts (table pose estimation, point-cloud visualization) also require downloading the entire BOP dataset to the data/datasets folder, for this you can use the script:\
`sh setup/download_test_dataset.sh {dataset name}`

# Inference
Several scripts are prepared for inference. You can use the complete pipeline using [Happypose](https://agimus-project.github.io/happypose/index.html) for inference on individual images using `scripts/coll_megapose_pipeline.py`, evaluate the entire dataset on the output of your pose estimator using `eval/eval_with_optim.py` or embed our method directly into your code using `src/optimizer.optim` or `src/optimizer.three_phase_optim`.
If you are evaluating the entire dataset, the poses should be stored in [BOP format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).