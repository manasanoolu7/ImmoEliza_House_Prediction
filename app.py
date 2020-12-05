from flask import Flask, request, jsonify
import pickle
# import pipeline.model.util as util
import pandas as pd

app = Flask(__name__)

pickle_in = open('houseprice_model.pkl', 'rb')

predictable = pickle.load(pickle_in)


@app.route('/')
def home():
    return "Welcome to House Price Prediction. Enter /welcome and /house for respective pages"


@app.route('/welcome')
def welcome():
    return jsonify("Alive!")


@app.route('/house', methods=["POST", "GET"])
def house():
    data = request.get_json()
    try:
        df = pd.DataFrame([data])

        df["postcode"] = int(df["postcode"])

        df["house_is"] = bool(str(df["house_is"]))

        df["rooms_number"] = float(df["rooms_number"])
        df["area"] = float(df["area"])

        df["equipped_kitchen_has"] = bool(str(df["equipped_kitchen_has"]))
        df["open_fire"] = bool(str(df["open_fire"]))
        df["terrace"] = bool(str(df["terrace"]))
        df["terrace_area"] = float(df["terrace_area"])
        df["garden_area"] = float(df["garden_area"])
        df["facades_number"] = float(df["facades_number"])

        result = predictable.predict(df)
        result = str(result).replace('[', '').replace(']', '')
        return str("the predicted price is {}".format(result))
    except:
        return "enter valid input"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=2000)
