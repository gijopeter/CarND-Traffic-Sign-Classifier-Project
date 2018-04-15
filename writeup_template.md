# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./test_images/001.jpg "Traffic Sign 1"
[image5]: ./test_images/005.jpg "Traffic Sign 1"
[image6]: ./test_images/010.jpg "Traffic Sign 3"
[image7]: ./test_images/008.jpg "Traffic Sign 4"
[image8]: ./test_images/012.jpg "Traffic Sign 5"
[image9]: ./examples/Traffic_sign_class_hist.png "Traffic sign Histogram"
[image10]: ./examples/Lenet_mod.png "Modified Lenet"
[image11]: ./loss_plot_Lenet.png "Loss plot Lenet"
[image12]: ./loss_plot_Lenet_dropout.png "Loss plot Lenet with Dropout"
[image13]: ./softmax-prob.png "Soft Max Probability"


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### 1. Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

### 2. visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing the number of traffic sign images present for each class(43) in dataset.

![alt text][image9]

### 3. Design and Test a Model Architecture

I started with default lenet architecture for training traffic sign data set because lenet was providing good prediction results with MNIST data set for identifying digits. I normalized the image to have cell values in the range [-0.5 to 0.5] in preprocessing step. Also introduced 2 drop layers (with probablity =0.6) between fully connected layers of Lenet architecture to improve the loss by reducing the chance for over fitting.

Loss plot of Lenet without dropout

![alt text][image11]

Loss plot of Lenet with dropout

![alt text][image12]


#### 4. Model Architecture


#### | Layer         		|     Description	        					|   ####

----------

- | Input         		| 32x32x3 RGB image   							| 
- | Convolution 5x5     | 1x1 stride, VALID padding, outputs 28x28x6 	|
- | RELU				|												|
- | Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
- | Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
- | RELU				|												|
- | Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
- | Flatten	      		| 5x5x16,  -> 400 								|
- | Fully connected		| 400 -> 120        							|
- | RELU				|												|
- | Dropout				| Keep probablity = 0.6							|
- | Fully connected		| 120 -> 84        								|
- | RELU				|												|
- | Dropout				| Keep probablity = 0.6							|
- | Fully connected		| 84 -> 43 (No of traffic sign classes)			|


![alt text][image10]

### 5. Training Method
I used Adam Optimizer to train the model with learning rate of 0.001. Used 15 EPOCHS to train with batch size of 128.I used to shuffle to get random training samples during each epoch.


My final model results were:

- training set accuracy of 0.996
- validation set accuracy of 0.945 
-  test set accuracy of 0.942

 

###6. Test a Model on New Images

I Chose 12 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five (uut of 12) German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



Here are the results of the prediction:

#### | Image			        |     Prediction	        					| 

----------

- | Speed limit  60Kmph			| Speed limit  60Kmph	   							| 
- | Wild animal crossing    	| Wild animal crossing								|
- | Traffic light				| Traffic light										|
- | Domestic animal crossing	| Bumpy Road					 				|
- | Stop						| Stop      										|


The model was able to correctly guess 9 of the 12 traffic signs, which gives an accuracy of 75%. 

Since traffic signal sign and pedastrian crossing have simlar structure as caution sign , it is observed that the second max probablity  traffic signal sign and pedastrian crossing was for caution sign. The two new traffic signs,  domestic animal crossing and minimum speed limit sign was chosen (though it is not available in the training set) . I was expecting th domestic animal sign would be classified as wild animal crossing and miminum speed limit(30) would be classified as speed limit(30) because of the similarity in the features, but it was misclassified to other labels. 

If i exclude the entirely new signs, model was able to predict 9 out of 10 signs accurately.

Please find the top softmax probabilities for each new reaffic sign in the figure below 


![alt text][image13]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


