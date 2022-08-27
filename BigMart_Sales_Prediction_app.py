from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("default.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    item_weight = float(request.form['item_weight'])
    item_fat_content = int(request.form["item_fat_content"])
    item_visibility = float(request.form["item_visibility"])
    item_type = int(request.form["item_type"])
    item_mrp = float(request.form["item_mrp"])
    outlet_establishment_year = float(request.form["outlet_establishment_year"])
    outlet_size = int(request.form["outlet_size"])
    outlet_location_type = int(request.form["outlet_location_type"])
    outlet_type = int(request.form["outlet_type"])

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    standard_scaler_path = r'D:\Samarth\BigMart Sales Prediction\Model\standarized_label.sav'

    sc = joblib.load(standard_scaler_path)

    X_std = sc.transform(X)

    random_forest_grid_label_path = r'D:\Samarth\BigMart Sales Prediction\Model\random_forest_grid_label.sav'

    rfg = joblib.load(random_forest_grid_label_path)

    Y_pred = rfg.predict(X_std)

    return jsonify({"Prediction of the sale of the item is ": float(Y_pred)})


if __name__ == "__main__":
    app.run(debug=True, port=6358)
