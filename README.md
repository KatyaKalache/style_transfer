
👩‍💻🔍 I am looking for a machine learning, computer vision position – I live in San Francisco but am willing to take a remote position.

I did [Holberton School](https://www.holbertonschool.com/) 2-year software engineering program where I picked machine learning as a specialization. In between the 2 years of the program, I did 2 internships: 
1. QA Engineer at Uber
1. Software Developer at OSIsoft (recently acquired by Areva for $4B).

Find me on:
* [linkedin](https://www.linkedin.com/in/ekaterinakalache/)
* [twitter](https://twitter.com/KatyaKalache)

I wanted to leverage the web & machine learning skills I learned to build something cool. Which is how this fun neural style transfer project came to be: www.style-transfer.tech

I used a number of different open-source tools  ❤️  to build this website: Tensorflow, NumPy, Vgg19, OpenCV and Bootstrap.

![readme photo]([https://i.imgur.com/jKeqXBy.jpg)

## Project's flow
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


