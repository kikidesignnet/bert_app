from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired


class QueryForm(FlaskForm):
    #the_document = StringField('Document', validators=[DataRequired()])
    the_wik_search = StringField('Search for a Wikipedia Page (e.g. "Janis Joplin")', validators=[DataRequired()])
    the_query = StringField('Your Question (e.g. "When was Janis born")', validators=[DataRequired()])
    submit = SubmitField('Submit')