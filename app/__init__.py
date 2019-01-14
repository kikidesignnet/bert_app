from flask import Flask
from config import Config

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'you-will-never-guess'
app.config.from_object(Config) #FIGURE THIS OUT WHEN I DEPLOY FOR REAL

from app import routes