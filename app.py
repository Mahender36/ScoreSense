from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            hours = float(request.form['hours'])
            scores = float(request.form['scores'])
            activities = 1 if request.form['activities'] == 'Yes' else 0
            sleep = float(request.form['sleep'])
            papers = float(request.form['papers'])

            input_data = np.array([[hours, scores, activities, sleep, papers]])
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]

            # Clip prediction to range 0–100
            prediction = max(0, min(prediction, 100))

            # Round for display
            prediction = round(prediction, 2)


        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
