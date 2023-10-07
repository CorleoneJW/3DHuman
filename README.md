# 3DHuman project brief introduction
Master project in UOM.</br>
This project is mainly about 3D Human Model, utilising deep-learning to do segmentation.</br>
Project consists of three stages:</br>
1. Segmentation stage
2. Voxelization stage
3. 3D Printing stage

# Usage
Download the dataset Spineweb dataset15.</br>
https://spineweb.digitalimaginggroup.ca/</br>
Put the data into dataset folder and run the **Spineweb_preprocess.ipynb**</br>
Select model and run the corresponding file **Spineweb_xxxNet.ipynb** to do the segmentation.</br>
Using **Visualize_vtk.ipynb** to generate voxel model by the result of segmentation.
Run **VTK2STL.ipynb** to transfer the Voxel model into STL file.
