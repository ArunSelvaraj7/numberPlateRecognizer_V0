from flask import Flask,render_template,Response,request,url_for,redirect
from flask_bootstrap import Bootstrap
from flask_wtf.csrf import CsrfProtect
from forms import uploadImage
import os
from werkzeug.utils import secure_filename
import cv2
from detect import detect_plate
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform


app = Flask(__name__)
bootstrap = Bootstrap(app)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = "HARD_TO_GUESS"
app.config['WTF_CSRF_CHECK_DEFAULT'] = False
CsrfProtect(app)
modelConfiguration = r'darknet-yolo/obj.cfg'
modelWeights = r'darknet-yolo/obj_60000.weights'


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    charModel = load_model( r'charRecognition/model.h5')


UPLOAD_FOLDER = r'static/images'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

for file in os.listdir(UPLOAD_FOLDER):
    os.remove(os.path.join(UPLOAD_FOLDER,file) )

@app.route('/',methods=['GET','POST'])
def home():
    output = ''
    form = uploadImage()
    if form.validate_on_submit():
        file = request.files.getlist('url')
        filename = ''
        if file:
            for f in file:
                if f.filename:
                    filename = secure_filename(f.filename)
                    f.save(os.path.join(UPLOAD_FOLDER, filename))
                    output = ''
                    output = detect_plate(net, charModel, filename)
                    
        # output = json.dumps({"main":"Condition failed on page baz"})
        # session['messages'] = messages
            return redirect(url_for('detector',output=output))
    return render_template('home.html',form=form)

@app.route('/detect',methods=['GET','POST'])
def detector():
    if request.method =='POST':
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER,file) )
        return redirect(url_for('home'))

    output = request.args['output']
    # filename = request.args['file']
    if len(os.listdir(UPLOAD_FOLDER)) > 0:
        file = os.listdir(UPLOAD_FOLDER)[0]
    else:
        file = ''
    
    return render_template('detector.html',file = file, output = output)


if __name__=='__main__':
    app.run(debug=False)