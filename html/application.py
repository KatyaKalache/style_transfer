from flask import Flask, render_template, request, send_from_directory
import os
import matplotlib.image as mpimg
import numpy as np
import cv2
import PIL

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
        upload = request.files["file"]
        if len(upload.filename) == 0:
            return render_template("index.html")
        else:
            global filename_content
            global destination_content
            global pictures
            filename_content = upload.filename
            pictures['content'] = filename_content
            print("filename_content in upload_content", filename_content)
            destination_content = '/'.join([target, filename_content])
            upload.save(destination_content)
            return render_template("upload_content.html", image_name=filename_content)
    
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
            global pictures
            filename_style = upload_style.filename
            print("filename_style in upload_style", filename_style)
            destination_style = '/'.join([target, filename_style])
            upload_style.save(destination_style)
            pictures["style"] = filename_style
            pictures = pictures
            return render_template("upload_style.html",
                                   image_name=filename_style,
                                   pictures = pictures)

    @application.route("/content_style", methods=["GET", "POST"])
    def show_content_style():
        pictures = {}
        generated_image = ""
        target = os.path.join(PROJECT_ROOT, 'images/')
        if filename_content is None:
            return render_template("index.html")
        print("filename_content", filename_content)
        print("destination_content", destination_content)
        if filename_style is None:
            return render_template("upload_content.html")
        print("filename_style", filename_style)
        print("destination_style", destination_style)
        pictures["content"] = filename_content
        pictures["style"] = filename_style
        pictures = pictures
        print("pictures", pictures)
        content_image = cv2.imread(destination_content)
        style_image = cv2.imread(destination_style)
        nst = NST(style_image, content_image)
        generated_image, cost = nst.generate_image(iterations=200, step=1, lr=0.002)
        mpimg.imsave(target + 'gen_'+filename_content, generated_image)
        pictures["gen"] = 'gen_'+filename_content
        pic_gen_name = pictures["gen"]
        generated_pic = cv2.imread(target + 'gen_'+filename_content)
        images = os.listdir(target)
        print("Best cost:", cost)
        return render_template("content_style.html",
                               images=images,
                               pictures = pictures,
                               pic_gen=pic_gen_name,
                               generated_pic=generated_pic)

    @application.route('/upload/<filename>')
    def send_images(filename):
        return send_from_directory("images", filename=filename)

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000, debug=True)
