#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import DataFrameReader
from pyspark.sql.functions import explode, col
import pyspark.sql.functions as F
import pyspark.sql.types as T

from datetime import datetime
import re
try:
    import pandas as pd
except ModuleNotFoundError:
    os.system("pip install pandas")
    import pandas as pd

url = 'jdbc:postgresql://postgres/amazon' 

properties = {'user': 'postgres', 
              'password':'spark123', 
              'driver':'org.postgresql.Driver'}

#---------------- Reviews file
reviews_path = './Movies_and_TV.json'
raw_reviews_df = sqlContext.read.json(reviews_path)

@F.udf(returnType=T.TimestampType())
def to_datetime(x):
    return datetime.fromtimestamp(x)

@F.udf(returnType=T.IntegerType())
def clean_vote(x):
    return 0 if x is None else int(x.replace(',', ''))

reviews_df = raw_reviews_df\
.withColumn('reviewTime', to_datetime(col('unixReviewTime')))\
.withColumn('vote', clean_vote((col('vote'))))\
.select('asin', 'reviewerID', 'reviewerName', 'reviewTime', 'verified', 'vote', 'summary', 'reviewText', 'overall')

reviews_df.write.jdbc(url=url, properties=properties, table='reviews', mode='overwrite')


#------------- Products metadata file

products_path = './meta_Movies_and_TV.json'
raw_products_meta_df = sqlContext.read.json(products_path)

@F.udf(returnType=T.FloatType())
def clean_price(x):
    price = x.replace("$", '').replace(',', '')
    return float(price) if price.replace('.', '').isdigit() else None

@F.udf(returnType=T.StringType())
def clean_main_cat(x):
    return None if x.startswith('<img') else x

@F.udf(returnType=T.StringType())
def clean_brand(x):
    return None if x == '' else x

@F.udf(returnType=T.StructType([T.StructField('rank_', T.IntegerType()), 
                                T.StructField('rank_cat', T.StringType())]))
def clean_rank(x):
    if x == '[]':
        return {'rank_': None, 'rank_cat': None}
    else:
        out = x.replace(',', '')
        out = re.search(r'\D*(\d{1,}) in ([^(]+)\s+\(?', out)
        if out is not None:
            out = out.groups()
            return {'rank_': int(out[0]), 'rank_cat': out[1].replace('&amp;', '&')}
        else:
            return {'rank_': out, 'rank_cat': None}

product_df = raw_products_meta_df\
.withColumn('price', clean_price(col('price')))\
.withColumn('main_cat', clean_main_cat(col('main_cat')))\
.withColumn('brand', clean_brand(col('brand')))\
.withColumn('rank', clean_rank(col('rank')))\
.withColumn('rank_cat', col('rank').getItem('rank_cat'))\
.withColumn('rank_', col('rank').getItem('rank_'))\
.withColumn('item_weight', col('details').getItem('\n    Item Weight: \n    ').alias('item_weight'))\
.withColumn('package_dimensions', col('details').getItem('\n    Package Dimensions: \n    ').alias('package_dimensions'))\
.withColumn('product_dimensions', col('details').getItem('\n    Product Dimensions: \n    ').alias('product_dimensions'))\
.withColumn('asin1', col('details').getItem('ASIN:').alias('asin1'))\
.withColumn('asin2', col('details').getItem('ASIN: ').alias('asin2'))\
.withColumn('audio_cd', col('details').getItem('Audio CD').alias('audio_cd'))\
.withColumn('audio_description', col('details').getItem('Audio Description:').alias('audio_description'))\
.withColumn('blue_ray_audio', col('details').getItem('Blu-ray Audio').alias('blue_ray_audio'))\
.withColumn('dvd_audio', col('details').getItem('DVD Audio').alias('dvd_audio'))\
.withColumn('digital_copy_expiration_date', col('details').getItem('Digital Copy Expiration Date:').alias('digital_copy_expiration_date'))\
.withColumn('domestic_shipping', col('details').getItem('Domestic Shipping: ').alias('domestic_shipping'))\
.withColumn('dubbed', col('details').getItem('Dubbed:').alias('dubbed'))\
.withColumn('isbn10', col('details').getItem('ISBN-10:').alias('isbn10'))\
.withColumn('isbn13', col('details').getItem('ISBN-13:').alias('isbn13'))\
.withColumn('international_shipping', col('details').getItem('International Shipping: ').alias('international_shipping'))\
.withColumn('item_model_number', col('details').getItem('Item model number:').alias('item_model_number'))\
.withColumn('label', col('details').getItem('Label:').alias('label'))\
.withColumn('language', col('details').getItem('Language:').alias('language'))\
.withColumn('n_discs', col('details').getItem('Number of Discs:').alias('n_discs'))\
.withColumn('please_note', col('details').getItem('Please Note:').alias('please_note'))\
.withColumn('publisher', col('details').getItem('Publisher:').alias('publisher'))\
.withColumn('run_time', col('details').getItem('Run Time:').alias('run_time'))\
.withColumn('spars_code', col('details').getItem('SPARS Code:').alias('spars_code'))\
.withColumn('series', col('details').getItem('Series:').alias('series'))\
.withColumn('shipping_weight', col('details').getItem('Shipping Weight:').alias('shipping_weight'))\
.withColumn('subtitles', col('details').getItem('Subtitles:').alias('subtitles'))\
.withColumn('subtitles_hearing_impaired', col('details').getItem('Subtitles for the Hearing Impaired:').alias('subtitles_hearing_impaired'))\
.withColumn('upc', col('details').getItem('UPC:').alias('upc'))\
.select('asin', 'title', 'main_cat', 'price', 'brand', 'rank_', 'rank_cat', 'item_weight', 
        'package_dimensions', 'product_dimensions', 'asin1', 'asin2', 'audio_cd', 'audio_description', 
        'blue_ray_audio', 'dvd_audio', 'digital_copy_expiration_date', 'domestic_shipping', 'dubbed', 
        'isbn10', 'isbn13', 'international_shipping', 'item_model_number', 'label', 'language', 
        'n_discs', 'please_note', 'publisher', 'run_time', 'spars_code', 'series', 'shipping_weight', 
        'subtitles', 'subtitles_hearing_impaired', 'upc')

product_df.write.jdbc(url=url, properties=properties, table='products', mode='overwrite')

also_buy_df = raw_products_meta_df.select(col('asin'), explode(col('also_buy')).alias('also_buy'))
also_view_df = raw_products_meta_df.select(col('asin'), explode(col('also_view')).alias('also_view'))
categories_df = raw_products_meta_df.select(col('asin'), explode(col('category')).alias('category'))
products_description_df = raw_products_meta_df.select(col('asin'), explode(col('description')).alias('description'))
products_feature_df = raw_products_meta_df.select(col('asin'), explode(col('feature')).alias('feature'))

also_buy_df.write.jdbc(url=url, properties=properties, table='also_buy', mode='overwrite')
also_view_df.write.jdbc(url=url, properties=properties, table='also_view', mode='overwrite')
categories_df.write.jdbc(url=url, properties=properties, table='categories', mode='overwrite')
products_description_df.write.jdbc(url=url, properties=properties, table='products_description', mode='overwrite')
products_feature_df.write.jdbc(url=url, properties=properties, table='products_feature', mode='overwrite')