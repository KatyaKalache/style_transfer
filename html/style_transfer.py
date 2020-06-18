# coding: utf-8
#!/usr/bin/env python3
"""
Create a class NST that performs tasks for neural style transfer
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
import traceback
tf.enable_eager_execution()


class NST:
    """
    Class NST
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor
        """        
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or \
           style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or \
           content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.model = model
        self.generate_features()
        self.gram_style_features = style_features
        self.content_feature = content_features

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its values are between 0 and 1
        and its largest side is 512 pixels
        Returns the scaled image
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or \
           image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        height = image.shape[0]
        width = image.shape[1]
        scale = 512.0 / max(height, width)
        new_height = int(scale*height)
        new_shape = (int(scale * height), int(scale * width))
        image = np.expand_dims(image, axis=0)
        image_scaled = tf.clip_by_value(tf.image.resize_images
                                        (image, new_shape) / 255.0, 0.0, 1.0)
        return image_scaled

    def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False)
        x = vgg.input

        for layer in vgg.layers[1:]:
            layer.trainable = False
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(name=layer.name)(x)
            else:
                x = layer(x)
        global model
        model = tf.keras.models.Model(vgg.input, x, name="model")
        outputs = [model.get_layer(layer).get_output_at(1)
                   for layer in self.style_layers + [self.content_layer]]
        model = tf.keras.models.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate gram matrices
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(
                input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, shape=[-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(tf.transpose(a), a) / tf.cast(n, tf.float32)
        gram = tf.reshape(gram, shape=[1, -1, channels])
        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost
        Sets the public instance attributes
        """
        global style_features
        global content_features
        st_lay_len = len(self.style_layers)
        style_image = preprocess_input(self.style_image*255)
        content_image = preprocess_input(self.content_image*255)
        features = model(style_image) + model(content_image)
        style_features = [self.gram_matrix(layer) for
                          layer in features[:st_lay_len]]
        content_features = features[-1:]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        Returns: the layer’s style cost
        """
        style_output = self.gram_matrix(style_output)
        style_loss = tf.reduce_mean((style_output-gram_target)**2)
        return style_loss

    def style_cost(self, style_outputs):
        """
        Calculates the style cost
        Returns: the style cost
        """
        if len(style_outputs) != len(self.style_layers):
            raise TypeError("style_outputs must be a list with a length of \
            {}").format(len(style_layers))
        style_cost = 0
        weight_per_layer = 1.0 / len(style_outputs)
        for i in range(len(style_outputs)):
            layer_style_cost = self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i])
            style_cost += tf.reduce_sum(layer_style_cost) * weight_per_layer
        return style_cost

    def content_cost(self, content_output):
        """
        Calculate the content cost
        Returns: the content cost
        """
        content_cost = tf.reduce_mean((content_output -
                                       self.content_feature)**2)
        return content_cost

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image
        Returns: total cost, content cost, style cost
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image must be a tensor \
            of shape {}".format(self.content_image.shape))
        preprocecced = preprocess_input(generated_image * 255)
        model_outputs = self.model(preprocecced)
        content_cost = self.content_cost(model_outputs[-1])
        style_cost = self.style_cost(model_outputs[:5])
        total_cost = content_cost*self.alpha + style_cost*self.beta
        return total_cost, content_cost, style_cost

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the tf.Tensor generated image
        Returns: gradients, J_total, J_content, J_style
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image must be a tensor \
        of shape {}".format(self.content_image.shape))
        J_total, J_content, J_style = self.total_cost(generated_image)
        with tf.GradientTape() as tape:
            loss, _, _ = self.total_cost(generated_image)
        grads = tape.gradient(loss, generated_image)
        return grads, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.999):
        """
        Generates image, gradient descent with Adam optimizer
        Returns: generated_image, cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and step < 0 or step > iterations:
            raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr < 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, (float, int)):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, (float, int)):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")
        opt = tf.train.AdamOptimizer(lr, beta1, beta2)
        generated_image = tf.Variable(self.content_image)
        best_loss, best_img = float('inf'), None
        for i in range(iterations):
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image)
            opt.apply_gradients([(grads, generated_image)])
            
            if J_total < best_loss:
                best_loss = J_total
                best_img = generated_image
            if i % step == 0:
                print("Cost at iteration {}: {}, content {}, style {}".format(
                    i, J_total, J_content, J_style))
            best_img = tf.reshape(best_img, best_img.shape[1:])
        return best_img.numpy(), best_loss
