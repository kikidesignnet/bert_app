from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired


class QueryForm(FlaskForm):
    the_document = StringField('Document', validators=[DataRequired()])
    the_wik_search = StringField('Wikipedia Page', validators=[DataRequired()])
    the_query = StringField('Your Question', validators=[DataRequired()])
    submit = SubmitField('Submit')