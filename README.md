## What’s this project for?

![index2.png](https://github.com/KatyaKalache/style_transfer/blob/master/html/static/img/index2.png)
As I am completing my 2-year program at Holberton School where I picked Machine Learning as a specialization, I wanted to leverage the web & machine learning skills I learned to build something cool. Which is how this neural style transfer project came to be.
You can visit project link at www.style-transfer.tech to try it.

👩‍💻🔍 I am looking for a machine learning position. I am ready to start whenever. I live in San Francisco but am willing to take a remote position.


## Dependencies for this project
I used a number of different open-source tools ❤️ to build this website. Here are some of them
* Tensorflow
* NumPy
* Vgg19
* OpenCV
* Bootstrap

## Process
0. Upload and preprocess images
* rescale both content and style images to equal shape: image's pixels values are between 0 and 1 and its largest side is 512 pixels
* store as a numpy.ndarray
* set Tensorflow to execute eagerly
1. Load VGG19 Keras as base model
* my  model’s input is  the VGG19 input
* the model’s output a list containing the outputs of the VGG19 
2. Adding method that calculates gram matricies
* to capture style features from the style reference image and from the generated image
3. Extracting the features used to calculate neural style cost
4. Calculating the style, content and total costs for generated image
* Calculating the style cost for a single layer
![layer_style_cost](https://latex.codecogs.com/gif.latex?E_{l}&space;=&space;\frac{1}{C_{l}^{2}}\sum_{i}^{C_{l}}\sum_{j}^{C_{l}}(G^{l}_{ij}&space;-&space;A^{l}_{ij})^{2})
* Calculating the style cost for generated image
![total_style_cost](https://latex.codecogs.com/gif.latex?L_{style}&space;=&space;\sum_{l}w_{l}E_{l})
* Calculating the content cost for the generated image
![total_content_cost](https://latex.codecogs.com/gif.latex?L_{content}&space;=&space;\frac{1}{H_{l}W_{l}C_{l}}\sum_{i}^{H_{l}}\sum_{j}^{W_{l}}\sum_{k}^{C_{l}}(F_{ijk}^{l}-P_{ijk}^{l})^2)
5. Computing the gradients for the generated image

6. Generating  the neural style transfered image
* gradient descent is performed using Adam optimization
* keeping track of the best cost and the image associated with that cost

Ekaterina Kalache: [github](https://github.com/KatyaKalache), [linkedin](https://www.linkedin.com/in/ekaterinakalache/), [twitter](https://twitter.com/KatyaKalache)