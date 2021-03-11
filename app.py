from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle



app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return render_template('index.html')

@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))

    loaded_model = pickle.load(open('first.sav', 'rb'))
    df['prediction'] = loaded_model.predict(df)

    

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.form['view_file']
        data = pd.read_csv(file)
        return render_template("data.html", data=data.to_html())

if __name__ == "__main__":
    app.run(debug=False,port=9000)