
from flask_wtf import FlaskForm
from wtforms import SubmitField,FileField

class uploadImage(FlaskForm):
    url = FileField('Photo')
    
    
    submit=SubmitField()