from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

reviews_df = pd.read_csv('data/reviews.csv', nrows=1000)
products_df = pd.read_csv('data/products_df.csv', nrows=1000)
also_buy_df = pd.read_csv('data/also_buy.csv', nrows=1000)
also_view_df = pd.read_csv('data/also_view.csv', nrows=1000)
categories_df = pd.read_csv('data/categories.csv', nrows=1000)
products_description_df = pd.read_csv(
    'data/products_description.csv', nrows=1000)
products_feature_df = pd.read_csv('data/products_feature.csv', nrows=1000)
products_images_df = pd.read_csv('data/products_images.csv', nrows=1000)


def create_app():

    #-----------------------------------------------------------------------------------#
    # INITIALISATION DE L'APPLICATION                                                   #
    #-----------------------------------------------------------------------------------#
    app = Flask(__name__)

    checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    #-----------------------------------------------------------------------------------#
    # PAGES                                                                             #
    #-----------------------------------------------------------------------------------#

    # @requiredLogin

    @app.route('/')
    def homePage():
        return render_template("index.html")

    @app.route('/stars')
    def firstPage():
        return render_template("stars.html")

    @app.route('/qa')
    def secondPage():
        return render_template("qa.html")

    #-----------------------------------------------------------------------------------#
    # APIs                                                                              #
    #-----------------------------------------------------------------------------------#

    @app.route('/api/getstars', methods=['POST'])
    def getStars():
        datas = request.json
        for data in datas:
            comment = data["comment"]
        tokens = tokenizer(comment, truncation=True,
                           padding=True, return_tensors='pt')
        with torch.no_grad():
            model.eval()
            stars = model(tokens['input_ids'])
        res = torch.nn.functional.softmax(stars.logits, dim=-1)
        return {"1": res[0][0].item(), "2": res[0][1].item(), "3": res[0][2].item(), "4": res[0][3].item(), "5": res[0][4].item()}

    @app.route('/api/get_products')
    def get_products():
        data = products_df[['asin', 'title']].to_dict('list')
        return jsonify(data)

    @app.route('api/get_product_details', methods=['POST'])
    def get_product_details():
        data = request.json  # data : {'id': ___}
        product_id = data['id']

        row = products_df.loc[products_df.asin == product_id, [
            'main_cat', 'price', 'description', 'image']]
        response = row.to_dict('record')[0]

        response.update(reviews_df.loc[reviews_df.asin == product_id, [
                        'reviewerID', 'reviewerName']].to_dict('list'))
        return jsonify(response)

    @app.route('api/get_reviews', methods=['POST'])
    def get_reviews():
        data = request.json  # data : {'product_id': ___, 'reviewer_id': _____}
        product_id = data['product_id']
        reviewer_id = data['reviewer_id']

        raw_response = reviews_df.loc[(reviews_df.asin == product_id) & (
            reviews_df.reviewerID == reviewer_id), ['reviewTime', 'vote', 'summary', 'reviewText', 'overall']]

        response = raw_response.to_dict('list')
        return jsonify(response)

    @app.route('api/get_all_reviews', methods=['POST'])
    def get_all_reviews():
        data = request.json  # data : {'id': ___}
        product_id = data['id']

        raw_response = reviews_df.loc[reviews_df.asin == product_id, [
            'reviewerID', 'reviewerName', 'reviewTime', 'vote', 'summary', 'reviewText', 'overall']]

        response = raw_response.to_dict('list')
        return jsonify(response)

    return app
