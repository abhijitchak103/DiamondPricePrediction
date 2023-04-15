from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData


application = Flask(__name__)

app = application

@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict_datapoint():
    if request == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            carat=float(request.form.get('carat')),
            table=float(request.form.get('table')),
            depth=float(request.form.get('depth')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut = request.form.get('cut'),
            color = request.form.get('color'),
            clarity = request.form.get('clarity')
        )

        final_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_df)

        result = round(pred[0], 2)

        return render_template('results.html', final_result = result)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')