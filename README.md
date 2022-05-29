# Visual Data Extractor using TensorFlow

The project was tested successfully on Python 3.7.9 and TensorFlow 2.3.1 respectively.

It is assumed that you already have following things installed on your machine.
- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy

If not already, kindly install these first before you move ahead.

Follow the steps below to train an end-to-end custom object detection model for detecting visual data in images/dashboards.

## Steps:
 
1. Clone the repository on your local machine.

2. Upload the ```Visual_Data_Extractor_using_TensorFlow.ipynb``` notebook file on Google Colab and run the cells one-by-one by following the instructions.

3. Install the **TensorFlow Object Detection API** (if not already) using ```pip install tensorflow-object-detection-api```.

4. Open the ```label_map_util.py``` file and Edit **Line 132** by replacing ```with tf.gfile.GFile(path, 'r') as fid:``` with ```with tf.io.gfile.GFile(path, 'r') as fid:```. 
You will find this file in the following path: ```C:\Users\<your-username>\AppData\Local\Programs\Python\Python37\Lib\site-packages\object_detection\utils```

5. Once the model is trained completely, download and extract the ```saved_model.zip``` file inside the ```Visual-Data-Extractor-using-TensorFlow``` folder. You will already find a pre-trained model [here](https://github.com/NSTiwari/Visual-Data-Extractor-using-TensorFlow/tree/main/content/).
 
6. Open command prompt and run ```py detect_graphs.py``` file.


## Facts: 
- The model was trained for 5 classes of visualizations viz. Pie Chart, Bar Graph, Donut, Line Chart and Area Chart respectively.
- Take a look at the [dataset](https://github.com/NSTiwari/Visual-Data-Extractor-using-TensorFlow/blob/main/graph-detection.zip) (you'll need to extract the .zip file) used for training this model to get an idea about the directory structures.

## Output:

![GitHub Logo](Output.gif)
