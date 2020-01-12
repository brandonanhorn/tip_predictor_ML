import pickle

from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__,)
pipe = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    print(args)
    new = pd.DataFrame({
    'total_bill': [args.get('total_bill')],
     'sex': [args.get('sex')],
     'smoker': [args.get('smoker')],
     'day': [args.get('day')],
     'time': [args.get('time')],
     'size': [args.get('size')]})

    prediction = round(float(pipe.predict(new)[0]), 2)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
