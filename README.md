

## Introduction
3D reconstruction/rendering project based on Neural Radiance Field (NeRF), TensoRF and Gaussian splatting technology to achieve high-precision scene reconstruction and dynamic rendering.  


## Dataset Preprocessing and Structure 

In the data processing stage, the video video.mp4 is first converted into an image sequence using ffmpeg at a frame rate of 5 frames per second, and output to the output directory, the file named frame_%04d.jpg. Then, COLMAP was used for camera parameter estimation and scene reconstruction. Image features were extracted by feature_extractor, and the single camera mode was set and GPU acceleration was disabled. Feature matching using exhaustive_matcher Then the mapper is executed to generate the sparse point cloud model and save it to the sparse directory. For later processing, the model is converted to TXT format and stored in the colmap_text directory. Finally, we run the colmap2nerf.py script to convert the data output from COLMAP to the format our NeRF model needs, where the image data path is my_data/output and the COLMAP text data path is my_data/colmap_text.

You can use the following code to process your own dataset:
```bash
ffmpeg -i video.mp4 -vf "fps=5" output/frame_%04d.jpg

colmap feature_extractor --database_path ./database.db --image_path ./output --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0

colmap exhaustive_matcher --database_path ./database.db --SiftMatching.use_gpu 0

mkdir sparse
colmap mapper \
    --database_path ./database.db \
    --image_path ./output \
    --output_path ./sparse

mkdir -p colmap_text
colmap model_converter \
    --input_path sparse/0 \
    --output_path colmap_text \
    --output_type TXT

python colmap2nerf.py --images my_data/output --text my_data/colmap_text
```
Finally, structure the dataset as follows:
```plaintext
my_data
|-- my_data
|   |-- colmap_text
|   |-- database.db
|   |-- output              # Store the images
|   `-- sparse
|-- transforms_test.json
|-- transforms_train.json
`-- transforms_val.json
```  


##  Experiment Set up

Caution: In addition to the common packages, Gaussian-splatting requires the installation of three repositories' libraries, which could be installed as follows:
```bash
# In the submodules (diff-gaussian-rasterization/fused-ssim/simple-knn)
cd gaussian-splatting/diff-gaussian-rasterization
pip install .
```

For **nerf-pytorch**, we modified configs/lego.txt to our dataset. So just run the following code to train and render our dataset:
```bash
python run_nerf.py --config configs/lego.txt
```
You can also train and render your dataset:
```bash
python run_nerf.py --config configs/{DATASET}.txt
```

For **tensorf**, just modify configs/your_own_data.txt and train a TensoRF as follows:
```bash
python train.py --config configs/your_own_data.txt
```
Then use the trained TensoRF to render as follows:
```bash
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
```

For **gaussian-splatting**, simply use this to train the optimizer:
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
Then run the following code to render datasets and evaluate the metrics:
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```
