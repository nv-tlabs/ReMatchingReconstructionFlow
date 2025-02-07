# ReMatching Dynamic Reconstruction Flow

**[Paper](https://arxiv.org/abs/2411.00705), [Project Page](https://research.nvidia.com/labs/toronto-ai/ReMatchingDynamicReconstructionFlow/)**

**ReMatching Dynamic Reconstruction Flow**<br>
Sara Oblak,
[Despoina Paschalidou](https://paschalidoud.github.io/),
[Sanja Fidler](https://www.cs.toronto.edu/~fidler/),
[Matan Atzmon](https://matanatz.github.io/)
<br>

Abstract: *Reconstructing dynamic scenes from image inputs is a fundamental computer
vision task with many downstream applications. Despite recent advancements, existing approaches still struggle to achieve high-quality reconstructions from unseen
viewpoints and timestamps. This work introduces the ReMatching framework,
designed to improve generalization quality by incorporating deformation priors into
dynamic reconstruction models. Our approach advocates for velocity-field-based
priors, for which we suggest a matching procedure that can seamlessly supplement
existing dynamic reconstruction pipelines. The framework is highly adaptable
and can be applied to various dynamic representations. Moreover, it supports
integrating multiple types of model priors and enables combining simpler ones to
create more complex classes. Our evaluations on popular benchmarks involving
both synthetic and real-world dynamic scenes demonstrate a clear improvement in
reconstruction accuracy of current state-of-the-art models.*




## Run our code

### Environment setup

```bash
git clone https://github.com/nv-tlabs/ReMatchingReconstructionFlow.git --recursive
cd ReMatchingReconstructionFlow

conda create -n rematching python=3.7
conda activate rematching

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Setting ReMatching framework hyperparameters
The ReMatching framework hyperparameters can be set in the configuration file *./rematching/arguments.conf*.
- #### Prior selection
    We currently support the following prior classes (names of parameters are consistent with the ones used in the paper):
    | Prior | Prior parameters |
    |---------|-------|
    |   P1      |prior.name = P1<br>prior.P1.V = [selected base tensors]<tr></tr>|
    |   P3      |prior.name = P3<br>prior.adaptive_prior.K = [number of parts]<br>prior.P3.B = [basis function hyperparameter]<tr></tr>|
    |   P4      |prior.name = P4<br>prior.adaptive_prior.K = [number of parts]<tr></tr>|
    |   P1 + P3|prior.name = P1_P3<br>prior.adaptive_prior.K = [number of parts]<br>prior.P1.V = [selected base tensors]<br>prior.P3.B = [basis function hyperparameter]<tr></tr>|
    |   P1 + P4|prior.name = P1_P4<br>prior.adaptive_prior.K = [number of parts]<br>prior.P1.V = [selected base tensors]<tr></tr>|
    |   P3 (image level)|prior.name = P3_Image<br>prior.adaptive_prior.K = [number of parts]<br>prior.P3.B = [basis function hyperparameter]<br>prior.cam_time = [view selection for the image-level loss]<tr></tr>|

- #### General hyperparameters
    ```
      general.rm_weight = [ReMatching loss weight]  
      prior.adaptive_prior.entropy_weight = [entropy loss weight for adaptive prior prediction]
    ```



### Training

```bash
python train.py -s path/to/your/dataset -m output/exp-name
```

### Datasets
We conducted our evaluation on the following three datasets: 
- #### [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) ([data](https://www.dropbox.com/scl/fi/cdcmkufncwcikk1dzbgb4/data.zip?rlkey=n5m21i84v2b2xk6h7qgiu8nkg&e=1&dl=0))
    Download and unzip the data folder. When running training, input the path to a specific scene from the dataset, for example:
    ```bash
    python train.py -s [location of downloaded data folder]/data/jumpingjacks -m output/dnerf_jumpingjacks
    ```

- #### [HyperNerf](https://hypernerf.github.io/) ([data](https://github.com/google/hypernerf/releases/tag/v0.1)) 
    Download and unzip for each selected scene. When running training, input the path to the selected scene, for example:
    ```bash
    python train.py -s [location of downloaded data folder]/slice-banana -m output/hypernerf_banana
    ```
- #### [Dynamic Scenes](https://research.nvidia.com/publication/2020-06_novel-view-synthesis-dynamic-scenes-globally-coherent-depths) ([data](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV)). 
    Download and unzip for each selected scene in the Nvidia_long folder. Then for each scene run:
    ```
    mkdir [location of downloaded scene]/dense/sparse/0
    mv [location of downloaded scene]/dense/sparse/*.bin [location of downloaded scene]/dense/sparse/0
    ```
    
    When running training, input the path to the selected scene, for example:
    ```bash
    python train.py -s [location of downloaded data folder]/Jumping/dense -m output/dynamicscenes_jumping
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License
Copyright &copy; 2025, NVIDIA Corporation & affiliates. All rights reserved.
This work is made available under the [Nvidia Source Code License](LICENSE.txt).

The code repository is based on the [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) repository, which is made available under the [MIT License](LICENSE_MIT.txt). The repository also contains the [Differentiable Gaussian Rasterization](https://github.com/ingra14m/diff-gaussian-rasterization-extentions) submodule, which is under the [Gaussian-Splatting License](LICENSE_GS.txt).

## Citation
```
@article{
  oblak2024rematching,
  title={ReMatching Dynamic Reconstruction Flow},
  author={Oblak, Sara and Paschalidou, Despoina and Fidler, Sanja and Atzmon, Matan},
  journal={arXiv preprint arXiv:2411.00705},
  year={2024}
}
```

