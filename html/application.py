from flask import Flask, render_template, request, send_from_directory, session
import os
import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
import glob

NST = __import__('style_transfer').NST
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(PROJECT_ROOT, 'images')
application = Flask(__name__)
application.secret_key = os.urandom(24)
application.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'images/')
application.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
application.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


class APP:
    def upload():
        """ Reads user's input and saves it to flask session """
        if 'file' not in request.files:
            return redirect('/')

        upload = request.files["file"]

        if upload.filename == "":
            return render_template("index.html")

        if upload:
            # reads user's image name
            filename = upload.filename
            # saving route as session key
            # filename as key value
            calling_route = request.url.split('/')[-1]
            session[calling_route] = filename
            # saves image to /images
            upload.save(os.path.join(
                application.config['UPLOAD_FOLDER'],
                str(filename)))

    @application.route("/",  methods=["GET", "POST"])
    def index():
        """
        Renders index page
        Empties /images dir
        """
        files = glob.glob("images/*")
        for f in files:
            os.remove(f)

        return render_template("index.html")

    @application.route("/upload_content",  methods=["GET", "POST"])
    def upload_content():
        """
        Gets user's content image
        """
        APP.upload()

        return render_template("upload_content.html")

    @application.route("/upload_style", methods=["GET", "POST"])
    def upload_style():
        """
        Gets user's style image
        """
        APP.upload()
        print(session)
        return render_template("upload_style.html",
                               filename_content=session.get("upload_content"),
                               filename_style=session.get("upload_style"))

    @application.route("/content_style", methods=["GET", "POST"])
    def show_content_style():
        """
        Generates restyled image
        """
        generated_image = ""
        images = os.listdir(target)
        filename_content = session.get('upload_content')
        filename_style = session.get('upload_style')
        content_image = cv2.imread('/'.join([target, filename_content]))
        style_image = cv2.imread('/'.join([target, filename_style]))
        nst = NST(style_image, content_image)
        generated_image, cost = nst.generate_image(
            iterations=100, step=1, lr=0.002)
        generated_image = (255.0 / generated_image.max() * (
            generated_image - generated_image.min())).astype(np.uint8)
        mpimg.imsave(application.config['UPLOAD_FOLDER']+"gen.jpg",
                     generated_image)
        images = os.listdir(application.config['UPLOAD_FOLDER'])
        return render_template("content_style.html",
                               images=images,
                               gen_pic_name="gen.jpg")

    @application.route('/upload/<filename>')
    def send_images(filename):
        return send_from_directory("images", filename)


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)
