from flask import Flask
from flask import render_template
from flask import request

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
            comment=data["comment"]
        tokens = tokenizer(comment, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            model.eval()
            stars = model(tokens['input_ids'])
        res=torch.nn.functional.softmax(stars.logits, dim=-1)
        return {"1":res[0][0].item(),"2":res[0][1].item(), "3":res[0][2].item(), "4":res[0][3].item(), "5":res[0][4].item()}

    return app