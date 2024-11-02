from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    api_key = os.getenv("API_KEY")
    print(api_key)
    return render_template('map.html', api_key=api_key)

if __name__ == '__main__':
    app.run()
