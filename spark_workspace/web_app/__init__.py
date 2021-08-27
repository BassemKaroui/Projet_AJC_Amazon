from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


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
    # INITIALISATION du modèle - Price Estimator                                        #
    #-----------------------------------------------------------------------------------#
    checkpoint_price = 'distilroberta-base'
    use_amp = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class PriceModel(nn.Module):

        def __init__(self, checkpoint, device):
            super(PriceModel, self).__init__()
            self.checkpoint = checkpoint
            self._device = device
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            # Image encoder
            self.resnext = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
            self.resnext.requires_grad_(False)
            self.resnext.fc = nn.Identity()
            # Text encoder
            self.distilroberta = AutoModel.from_pretrained(self.checkpoint)
            self.distilroberta.requires_grad_(False)
            # Attention related
            self.cnn_projection = nn.Sequential(
                nn.Linear(2048, 768, device=device),
                nn.Tanh()
            )
            self.titles_cross_att = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1,
                                                          batch_first=True, device=device)
            self.categories_cross_att = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1,
                                                              batch_first=True, device=device)
            self.descriptions_cross_att = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1,
                                                                batch_first=True, device=device)
            self.intermediate_layer_1 = nn.Sequential(
                nn.Linear(768, 768, device=device),
                nn.LayerNorm(768, device=device),
                nn.Tanh(),
                nn.Dropout(0.1)
            )
            self.full_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.1,
                                                        batch_first=True, device=device)
            self.intermediate_layer_2 = nn.Sequential(
                nn.Linear(768, 768, device=device),
                nn.LayerNorm(768, device=device),
                nn.Tanh(),
                nn.Dropout(0.1)
            )

            self.regressor = nn.Sequential(
                nn.Linear(768, 768, device=device),
                nn.LeakyReLU(0.1),
                nn.Linear(768, 1, device=device))

            self.top = nn.ModuleList([self.cnn_projection,
                                      self.titles_cross_att,
                                      self.categories_cross_att,
                                      self.descriptions_cross_att,
                                      self.intermediate_layer_1,
                                      self.full_attention,
                                      self.intermediate_layer_2,
                                      self.regressor])
            # Device
            self.to(device)
            # Query
            self._query = self.tokenizer(
                ['What is the price of this Amazon product?'], return_tensors='pt')
            self._query = {k: v.to(device) for k, v in self._query.items()}
            with amp.autocast(enabled=use_amp):
                self.register_buffer('_query_ctx_embeddings', self.distilroberta(
                    **self._query).last_hidden_state)
            self._query_ctx_embeddings = self._query_ctx_embeddings.mean(
                dim=1)  # shape : (1 x 768)

        @amp.autocast(enabled=use_amp)
        def forward(self, titles, categories, descriptions, mask_imgs, imgs, *args):
            titles = {k: v.to(self._device, non_blocking=True)
                      for k, v in titles.items()}
            categories = {k: v.to(self._device, non_blocking=True)
                          for k, v in categories.items()}
            descriptions = {k: v.to(self._device, non_blocking=True)
                            for k, v in descriptions.items()}
            imgs = imgs.to(device, non_blocking=True)
            mask_imgs = mask_imgs.unsqueeze(
                dim=-1).to(device, non_blocking=True)
            mask_imgs = torch.cat([mask_imgs, torch.zeros(
                imgs.shape[0], 3, dtype=torch.bool, device=self._device)], dim=-1)

            query_ctx_embeddings = self._query_ctx_embeddings.repeat(
                imgs.shape[0], 1)  # shape : (Batch_size x 768)

            imgs_encodings = self.resnext(imgs)
            imgs_encodings = self.cnn_projection(
                imgs_encodings).unsqueeze(dim=1)

            titles_ctx_embeddings = self.distilroberta(
                **titles).last_hidden_state
            titles_ctx_embeddings, _ = self.titles_cross_att(query=query_ctx_embeddings.unsqueeze(dim=1),
                                                             key=titles_ctx_embeddings,
                                                             value=titles_ctx_embeddings)

            categories_ctx_embeddings = self.distilroberta(
                **categories).last_hidden_state
            categories_ctx_embeddings, _ = self.categories_cross_att(query=query_ctx_embeddings.unsqueeze(dim=1),
                                                                     key=categories_ctx_embeddings,
                                                                     value=categories_ctx_embeddings)

            descriptions_ctx_embeddings = self.distilroberta(
                **descriptions).last_hidden_state
            descriptions_ctx_embeddings, _ = self.categories_cross_att(query=query_ctx_embeddings.unsqueeze(dim=1),
                                                                       key=descriptions_ctx_embeddings,
                                                                       value=descriptions_ctx_embeddings)

            outputs = torch.cat([imgs_encodings,
                                titles_ctx_embeddings,
                                categories_ctx_embeddings,
                                descriptions_ctx_embeddings],
                                dim=1)

            outputs = self.intermediate_layer_1(outputs)
            outputs, _ = self.full_attention(query=query_ctx_embeddings.unsqueeze(dim=1),
                                             key=outputs, value=outputs,
                                             key_padding_mask=mask_imgs)
            outputs = self.intermediate_layer_2(outputs).squeeze(
                dim=1)  # shape : (Batch_size x 768)
            outputs = self.regressor(outputs)
            return outputs

    price_estimator = PriceModel(checkpoint_price, device)
    # price_estimator.load_state_dict(torch.load(''))
    price_estimator.top.load_state_dict(torch.load(
        './price_calculator_checkpoints/model_top_epoch_15.pth'))

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

    #-----------------------------------------------------------------------------------#
    # APIs : price estimator                                                            #
    #-----------------------------------------------------------------------------------#

    @app.route('/api/price', methods=['POST'])
    def price():
        # {'title':_, 'description':_, 'main_cat': _, 'image':_}
        data = request.json
        root_path_img = 'web_app/data/product_images'

        image = data['image']
        title = 'Amazon product title : ' + data['title']
        description = 'Amazon product description : ' + data['description']
        category = 'Amazon product category : ' + data['main_cat']
        mask_img = True if image == '' else False
        if not mask_img:
            img_name = image.split(
                'https://images-na.ssl-images-amazon.com/images/I/')[-1]
            img_path = os.path.join(root_path_img, img_name)
            img = Image.open(img_path).convert('RGB')
            all_img_channels_stats = ((0.5143011212348938, 0.48371440172195435, 0.46702200174331665),
                                      (0.3150539994239807, 0.3088625967502594, 0.3137829899787903))
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(*all_img_channels_stats)
            ])
            img = transform(img)
        else:
            img = torch.zeros(3, 224, 224)
        img = img.unsqueeze(dim=0)
        mask_img = torch.tensor([mask_img])
        title = price_estimator.tokenizer(
            title, truncation=True, return_tensors='pt')
        description = price_estimator.tokenizer(
            description, truncation=True, return_tensors='pt')
        category = price_estimator.tokenizer(
            category, truncation=True, return_tensors='pt')
        pred_price = price_estimator(
            title, category, description, mask_img, img)
        response = {'price': pred_price.item()}
        return jsonify(response)
    return app
