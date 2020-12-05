{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from flask import Flask, request,render_template,jsonify\n",
    "import pickle\n",
    "\n",
    "app=Flask(__name__)\n",
    "model=pickle.load(open('houseprice_model.pkl','rb))\n",
    "\n",
    "@app.route('/')\n",
    "def  home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict',method=['POST'])\n",
    "def predict():\n",
    "                       \n",
    "    int_features=[int(x) for x in request.form.values()]\n",
    "    final_features=[np.array(int_features)]\n",
    "    prediction=model.predict(final_features)\n",
    "                       \n",
    "    output=round(prediction[0],2)\n",
    "                       \n",
    "    return render_template('index.html',prediction_text='price is ${}'.format(output))\n",
    "                       \n",
    "@app.route('/predict_api',methods=['POST'])\n",
    "def predict_api():\n",
    "                       \n",
    "        data=request.get_json(force=True)\n",
    "        prediction=model.predict([np.array(list(data.values()))])\n",
    "        output=prediction[0]\n",
    "        return jsonify(output)\n",
    "\n",
    "if __name__==\"__main__\":\n",
    "                       app.run(debug=True)\n",
    "                       \n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
