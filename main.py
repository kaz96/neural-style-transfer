from flask import Flask, Response, render_template
import cv2
import tensorflow_hub as hub
import tensorflow as tf
import  jyserver.Flask as jsf
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_img(img):
    max_dim = 512
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

app = Flask(__name__,
            static_folder = "assets",
            template_folder= ""
            )

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'assets/files'

@jsf.use(app)
class App:
    def __init__(self):
        self.style_list = ["starry","cubism","fire"]

    def change_image_border(self,style_name):
        self.js.document.getElementById(style_name).style.border = '3px solid white'
        remove_list = [x for x in self.style_list if x != style_name]
        for style_to_remove in remove_list:
            self.js.document.getElementById(style_to_remove).style.border = ''

    def change_style(self,style_name):
        global style

        self.change_image_border(style_name)

        style_temp = tf.io.read_file("assets/"+style_name+".jpg")
        style_temp = tf.image.decode_image(style_temp, channels=3)
        style = tf.constant(load_img(style_temp))





video = cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)

def gen(video):
    global style
    style = tf.io.read_file("assets/starry.jpg")
    style = tf.image.decode_image(style, channels=3)
    style = tf.constant(load_img(style))

    while True:
        success, image = video.read()
        preprocess_img = load_img(image)  # 1 black


        stylized_image = hub_model(preprocess_img, style)[0]

        stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)

        ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(stylized_image[0].numpy(), cv2.COLOR_BGR2RGB))

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

class upload_file_form(FlaskForm):
    file = FileField("File", validators=[InputRequired()] )
    submit = SubmitField("Upload File")

@app.route('/', methods = ['GET','POST'])
def index():
    form = upload_file_form()
    if form.validate_on_submit():
        file = form.file.data # get the file
        print(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # save the file
    return App.render(render_template('index.html', form = form))

@app.route('/video_feed')
def video_feed():
		# Set to global because we refer the video variable on global scope,
		# Or in other words outside the function
    global video

		# Return the result on the web
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,threaded=True)