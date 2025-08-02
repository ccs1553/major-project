Satellite Image Segmentation using Deep Learning and Earth Engine
This project involves satellite image processing and land cover segmentation using multiple deep learning architectures. The system integrates Google Earth Engine (GEE) for satellite data access and uses Python with libraries such as TensorFlow, PyTorch, and segmentation models for training and inference.

📁 Project Structure
The notebook follows this pipeline:

Import and Install Required Libraries

Uses pip to install Earth Engine, geemap, folium, and deep learning libraries.

Authenticate and Initialize Earth Engine

Authenticates with a specified GEE project and sets up folium for visualization.

Define Paths and Load Data

Sets local paths and reads input satellite imagery using GDAL and OpenCV.

Data Preprocessing

Processes satellite images into training-ready datasets.

Includes resizing, augmentation, and label encoding.

Model Training and Prediction

Trains and evaluates the following segmentation models:

U-Net

FPN with VGG16

Fully Convolutional Network (FCN)

Additional Custom Model

📦 Dependencies
Install all required libraries using:

bash
Copy
Edit
pip install earthengine-api geemap folium geehydro gdal segmentation-models torch torchvision torchaudio tensorflow
Other packages:

numpy, opencv-python, matplotlib, scikit-image, pandas, tqdm, Pillow, osgeo

🧠 Models Used
Model	Description
U-Net	A CNN architecture widely used for biomedical and satellite image segmentation.
FPN + VGG16	Feature Pyramid Network with VGG16 encoder for multi-scale feature learning.
FCN	Fully Convolutional Network for pixel-wise prediction.
Model 4	Custom or experimental architecture for further comparison.

🌍 Earth Engine Integration
The project uses earthengine-api for downloading and visualizing satellite data.

Folium maps are enhanced with Earth Engine imagery layers.

📊 Results and Evaluation
Metrics and model comparison will be added after training is completed.

🧪 How to Run
Install all dependencies.

Authenticate your Google Earth Engine account.

Run the notebook cells in order.

Check visualizations and metrics for model evaluation.
