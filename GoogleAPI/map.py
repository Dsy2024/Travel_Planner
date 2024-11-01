from flask import Flask, render_template
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

@app.route('/')
def index():
    api_key = os.getenv("API_KEY")
    return render_template('map.html', api_key=api_key)

if __name__ == '__main__':
    app.run()
