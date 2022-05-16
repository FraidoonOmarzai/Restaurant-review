from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('reviewR.html')
    # return "working"


@app.route('/predictR', methods=['POST'])
def predictR():
    model = joblib.load('Trained Model/review-model.pkl')
    tfv = joblib.load('Trained Model/tfv-transform.pkl')

    if request.method == 'POST':
        review = request.form['review']
        data = [review]
        vect = tfv.transform(data).toarray()
        result = model.predict(vect)

    if (int(result) == 1):
        prediction = "This is a POSITIVE Review"
    else:
        prediction = "This is a NEGATIVE Review"
    return render_template('result.html', prediction_text=prediction)


if __name__ == '__main__':
    app.run(debug=True)
