# Rafiance Fields from VGGSfM and Mast3r, and Their Comparison

This project aims to explore Gaussian Splatting using two different wild deep-based camera-pose & 3D pointcloud reconstruction methodologies: VGGSfM and Mast3r. 
The objective is to compare their performances and understand the advantages and limitations of each approach.

## Project Structure

- **VGGSfM**: Results of (2D) Gaussian Splatting using VGGSfM.
- **Mast3r**: Implementation and results of (2D) Gaussian Splatting using Mast3r.
- **Comparison**: Detailed comparison of the two methods based on various performance metrics.

### Usage 
You should install Viser and plyfile 
```bash
pip install viser==0.1.29
pip install plyfile
```
**Features**
- Convert MASt3R's capability to the COLMAP-compatible format: (in MASt3R installed environment)
```bash
python colmap_from_mast3r.py 
```
- Online COLMAP results viewer
```bash
python colmap_vis.py --images_dir <path/to/images> --colmap_path <path/to/colmaps/> 
```


## Results Comparison 

**1) PointCloud**
| MASt3R | VGGSfM |
| --- | --- |
| <img src="assets/NLE_mast3r.png" alt="NLE MASt3R" /> | <img src="assets/NLE_vggsfm.png" alt="NLE VGGSfM" /> |
| <img src="assets/guitar_mast3r.PNG" alt="Guitar MASt3R" /> | <img src="assets/guitar_vggsfm.PNG" alt="Guitar VGGSfM" /> |
| <img src="assets/pen_sparse_mast3r.PNG" alt="Pen Sparse MASt3R" /> | <img src="assets/pen_sparse_vggsfm.PNG" alt="Pen Sparse VGGSfM" /> |

**2) Radiance Fields Reconstruction**
| MASt3R | VGGSfM |
| --- | --- |
| ![visualization](assets/NLE_mast3r_2dgs.gif) | ![visualization](assets/NLE_vggsfm_2dgs.gif)|
| ![visualization](assets/guitar_mast3r_2dgs.gif) | ![visualization](assets/guitar_vggsfm_2dgs.gif)|
| ![visualization](assets/pen_sparse_mast3r_2dgs.gif) | ![visualization](assets/pen_sparse_vggsfm_2dgs.gif)|


## About VGGSfM and Mast3r 

###  VGGSfM: Visual Geometry Group Structure-from-Motion
VGGSfM introduces a fully differentiable SfM pipeline, designed to integrate deep learning models into every stage of the SfM process. This method includes:
- End-to-End Differentiability: The pipeline is fully differentiable, allowing for end-to-end training and optimization.
- Superior Camera Reconstruction: Provides highly accurate camera parameter reconstruction, which is beneficial for downstream tasks like neural rendering.
   - Global Optimization: Simultaneously optimizes all camera poses, avoiding the pitfalls of incremental methods.
   - Differentiable Bundle Adjustment (BA): Refines both camera parameters and 3D points to minimize reprojection errors.

### MASt3R: Grounding Image Matching in 3D with MASt3R
MASt3R enhances stereo matching by integrating dense local feature prediction and fast reciprocal matching upon Dust3r baseline. It focuses on leveraging stereo vision to improve 3D point and camera parameter estimation.

- Dense Matching: Excels in dense feature matching, providing detailed 3D reconstructions.
- Attention Mechanism: Utilizes cross-attention between image pairs for robust feature extraction and matching.
- Camera Parameter Reconstruction: Not the primary focus, leading to less accurate camera pose estimations compared to VGGSfM and COLMAP.
