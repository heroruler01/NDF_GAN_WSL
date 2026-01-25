Start training of the implicit generative model
Folder distance_based_gan
1. Preprocessing: 
	--> prepare_shapenet_daraset for normal preprocessing of sphere distances
	--> prepare_watertight_dataset for insertion of models in watertught models 
	--> prepare_internal_detail for automated model selection of non-watertight models 
2. Create a textfile with create_textfile.py
3. Start Training 
	--> train_point_gan.py: change "writer" and "checkpoint_path" for storing checkpoint and training loss
	--> point_sdf_net.py is the architecture, for Pointnet-mix comment out in forward and follow instructions in the file 
	--> fourier_features.py: Code for fourier features, hyperparameters are in train_point_gan.py
4. Rendering results: 
	--> DeepSDF_renderer/renderer.py for Marching Cubes
	--> PointCloud_Renderer/pc_renderer.py for Point Cloud Sampling 

Start training of the image based implicit AE 
Folder image_based_ae/gan
1. train.py, all other functions are called from there 

