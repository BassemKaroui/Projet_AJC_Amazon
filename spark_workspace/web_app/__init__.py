from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

reviews_df = pd.read_csv('web_app/data/reviews.csv', nrows=1000)
products_df = pd.read_csv('web_app/data/products.csv')
products_df = products_df.loc[products_df.title.notna()]
also_buy_df = pd.read_csv('web_app/data/also_buy.csv')
also_view_df = pd.read_csv('web_app/data/also_view.csv')
categories_df = pd.read_csv('web_app/data/categories.csv',)
products_description_df = pd.read_csv(
    'web_app/data/products_description.csv')
products_feature_df = pd.read_csv(
    'web_app/data/products_feature.csv')
products_images_df = pd.read_csv(
    'web_app/data/products_images.csv')


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

    @app.route('/reviews')
    def firstPage():
        return render_template("reviews.html")

    @app.route('/price')
    def secondPage():
        return render_template("price.html")

    @app.route('/rating')
    def thirdPage():
        return render_template("rating.html")

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

    # @app.route('/api/get_products')
    # def get_products():
    #     data = products_df[['asin', 'title']].to_dict('list')
    #     return jsonify(data)

    @app.route('/api/get_product_details', methods=['POST'])
    def get_product_details():
        data = request.json  # data : {'id': ___}
        product_id = data['id']

        row = products_df.loc[products_df.asin == product_id, [
            'main_cat', 'price', 'description', 'image']]
        response = row.to_json(orient='records')

        return response

    @app.route('/api/get_all_reviews', methods=['POST'])
    def get_all_reviews():
        data = request.json  # data : {'id': ___}
        product_id = data['id']

        raw_response = reviews_df.loc[reviews_df.asin == product_id, [
            'reviewerID', 'reviewerName', 'reviewTime', 'vote', 'summary', 'reviewText', 'overall']].sort_values('reviewTime', ascending=False)

        response = raw_response.to_json(orient='records')
        return response

    @app.route('/api/autocomp', methods=['POST'])
    def sendWordList():
        data = request.json  # data : {"pattern" : ______}
        pattern = data["pattern"].lower()
        if pattern != "":
            response = products_df.loc[products_df.title.str.lower().str.startswith(pattern), [
                'asin', 'title']].to_dict('list')
            return jsonify(response)
        else:
            return jsonify({'asin': [], 'title': []})

    return app
