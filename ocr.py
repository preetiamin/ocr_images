#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:33:59 2020

@author: preetiamin
"""
import imutils
import cv2
import pytesseract
import re
import os
 # Imports the Google Cloud client library
from google.cloud import vision
#from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
"preeti-amin-6072020-6372dbbd3873.json"

def find_barcode(x):
    '''
    Find 8 digit number in the text
    
    Input: String to search within
    Output: 8-digit barcode
    '''
    matches = re.findall(r"\b\d{8}\b",str(x))
    if matches:
        matches = [int(x) for x in matches]
        matches.sort()
        return matches[-1]
    return None

def handwritten_ocr(url):
    '''
    Takes in a url, processes it using pytesseract dataframe, cleans the 
    dataframe and returns the barode, ocr text and the segmented image
    '''

    #try:
    img = imutils.url_to_image(url)
    img = img[int(img.shape[0]*0.7):,:]

    
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()
    image.source.image_uri = url

    response = client.document_text_detection(image=image)
    text = response.full_text_annotation.text
    barcode = find_barcode(text)
    if barcode is None:
        barcode = '9999999'

    # Look for text in the right half of the image,  
    # brighten the image and load results to dataframe
    img = img[:,int(img.shape[1]*0.5):]
    return barcode, text.replace('\n', ' '), img

    #except:
    #    return '999999', 'error loading url', None
    #    pass
    

def printed_ocr(url):
    '''
    Takes in a url, processes it using pytesseract dataframe, cleans the 
    dataframe and returns the barode, ocr text and the segmented image
    '''

    try:
        image = imutils.url_to_image(url)
        image = image[int(image.shape[0]*0.7):,:]
        w,h = image.shape[0],image.shape[1]

        # Look for barcode, use a resized image for this, 
        # if it can't be found, set to '9999999'
        small_img = cv2.resize(image,(int(h*0.5),int(w*0.5)), 
                               interpolation = cv2.INTER_AREA)
        top_to_look = int(small_img.shape[0])
        for j in range(20):
            text = pytesseract.image_to_string(small_img[-top_to_look:,:])
            barcode = find_barcode(text)
            if barcode:
                break
            else:
                top_to_look = int(top_to_look*0.8)
        if barcode is None:
            barcode = '9999999'

        # Look for text in the right half of the image, 
        # brighten the image and load results to dataframe
        image = image[:,int(image.shape[1]*0.5):]
        (_, image) = cv2.threshold(image,200, 255, cv2.THRESH_BINARY)
        d = pytesseract.image_to_data(image, output_type='data.frame')

        # Drop rows with Nan values, create columns for right, bottom and group
        d.dropna(subset=['text'],inplace=True)
        d['right'] = d['left'] + d['width']
        d['bottom'] = d['top'] + d['height']
        d['group'] = d.groupby(['block_num','par_num','line_num']).ngroup()

        # Drop text with confidence lower than 20% &
        # where it extends to boundaries of the image
        d = d[d['conf']>20]
        d = d[(d['left']>0) & (d['right']<image.shape[1]) & 
              (d['top']>0) & (d['bottom']<image.shape[0])]

        # Drop rows where there is a | character becuase of background noise
        d['text'] = d['text'].apply(lambda x:re.sub(r'\||\s|_','',x))

        # Loop through the dataframe and adjust to optimize text grouping
        for i in range(d.shape[0]-1):    
            # If horizontal spacing between words is large, create new group
            spacing = d.iloc[i+1,d.columns.get_loc('left')] - \
            d.iloc[i,d.columns.get_loc('right')]
            if (spacing > 100) & (d.iloc[i,-1] == d.iloc[i+1,-1]):
                d.iloc[i+1:,-1] +=1

        tokens = d.groupby('group')['text'].apply(lambda x:' '.join(x)).values
        tokens = [x.strip() for x in tokens if re.search(r'\w',x)]
        ocr_text = '\n'.join(tokens)
        
        # Draw a bounding box around main text box
        cv2.rectangle(image, (d['left'].min()-20, d['top'].min()-20), 
                (d['right'].max()+20, d['bottom'].max()+20), (36,255,12), 3)
        
        #Write image to a file, print results and add results to a dataframe
        text_image = image[d['top'].min()-20:d['bottom'].max()+20,
                                         d['left'].min()-20:d['right'].max()+20]
        if (text_image.shape[0]==0) or (text_image.shape[1]==0):
            text_image = image
            
        return int(barcode), ocr_text, text_image

    except:
        return '999999', 'error loading url', None
        pass


    