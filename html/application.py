from flask import Flask, render_template, request, send_from_directory
import os
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from matplotlib import pyplot as plt

NST = __import__('style_transfer').NST
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
application = Flask(__name__)
pictures = {}
class APP:
    @application.route("/",  methods=["GET", "POST"])
    def index():
        return render_template("index.html")
    @application.route("/upload_content",  methods=["POST"])
    def upload_content():
        target = os.path.join(PROJECT_ROOT, 'images/')
        if not os.path.isdir(target):
            os.mkdir(target)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        upload = request.files["file"]
        if len(upload.filename) == 0:
            return render_template("index.html")
        else:
            global filename_content
            global destination_content
            filename_content = upload.filename
            destination_content = '/'.join([target, filename_content])
            upload.save(destination_content)
            return render_template("upload_content.html")
    
    @application.route("/upload_style", methods=["POST"])
    def upload_style():
        target = os.path.join(PROJECT_ROOT, 'images/')
        if not os.path.isdir(target):
            os.mkdir(target)
        upload_style = request.files["file"]
        if len(upload_style.filename) == 0:
            return render_template("upload_content.html")
        else:
            global filename_style
            global destination_style
            global filename_content
            filename_style = upload_style.filename
            destination_style = '/'.join([target, filename_style])
            upload_style.save(destination_style)
            return render_template("upload_style.html",
                                   filename_content=filename_content,
                                   filename_style=filename_style)

    @application.route("/content_style", methods=["GET", "POST"])
    def show_content_style():
        pictures = {}
        generated_image = ""
        target = os.path.join(PROJECT_ROOT, 'images/')
        global filename_content
        if filename_content is None:
            return render_template("index.html")
        if filename_style is None:
            return render_template("upload_content.html")
        content_image = cv2.imread(destination_content)
        style_image = cv2.imread(destination_style)
        nst = NST(style_image, content_image)
        generated_image, cost = nst.generate_image(iterations=100, step=1, lr=0.002)
        generated_image = (255.0 / generated_image.max() * (generated_image - generated_image.min())).astype(np.uint8)
        mpimg.imsave(target + 'gen_'+filename_content, generated_image)
        images = os.listdir(target)
#        generated_pic = cv2.imread(target + 'gen_'+filename_content)
        gen_pic_name='gen_'+filename_content
        return render_template("content_style.html",
                               images=images,
                               gen_pic_name=gen_pic_name)

    @application.route('/upload/<filename>')
    def send_images(filename):
        return send_from_directory("images", filename=filename)

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000, debug=True)
