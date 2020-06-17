#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:33:59 2020

@author: preetiamin
"""
import imutils
import cv2
import os
import pytesseract
import re
import pandas as pd
 # Imports the Google Cloud client library
from google.cloud import vision
#from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_VISION_API_KEY.json"

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

def handwritten_ocr(url, save_img=False):
    '''
    Takes in a url, processes it using pytesseract dataframe, cleans the 
    dataframe and returns the barode, ocr text and the segmented image
    
    Save image as jpg if save_img is True
    '''

    try:
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
        
        if save_img:
            cv2.imwrite(str(barcode) + '.jpg',img)
            
        return barcode, text.replace('\n', ' '), img

    except:
        return None, None, None
    

def printed_ocr(url, save_img=False):
    '''
    Takes in a url, processes it using pytesseract dataframe, cleans the 
    dataframe and returns the barode, ocr text and the segmented image
    
    Save image as jpg if save_img is True
    '''

    try:
        # for reading image from url
        image = imutils.url_to_image(url)
        
        # for reading image from local folder
        #image = cv2.imread()  
        
        image = image[int(image.shape[0]*0.7):,:]
        
        #w,h = image.shape[0],image.shape[1]
        # Look for barcode, use a resized image for this, 
        # if it can't be found, set to '9999999'
        #small_img = cv2.resize(image,(int(h*0.5),int(w*0.5)), 
        #                       interpolation = cv2.INTER_AREA)
        # commented this out to use full image, as it increased accuracy
        
        top_to_look = int(image.shape[0])
        barcode = None
        
        for angle in [0, 2,-2,4,-4]:
            rot_img = imutils.rotate_bound(image, angle)
            text = pytesseract.image_to_string(rot_img)
            barcode = find_barcode(text)
            if barcode:
                break
            
        if barcode is None:
            for j in range(20):
                text = pytesseract.image_to_string(image[-top_to_look:,:])
                barcode = find_barcode(text)
                if barcode:
                    break
                else:
                    top_to_look = int(top_to_look*0.8)
                
        if barcode is None:
            bc_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            bc_img = cv2.GaussianBlur(bc_img,(5,5),0)
            text = pytesseract.image_to_string(bc_img)
            barcode = find_barcode(text)
        
        # Look for text in the right half of the image, 
        # brighten the image and load results to dataframe
        image = image[:,int(image.shape[1]*0.5):]
        (_, image) = cv2.threshold(image,200, 255, cv2.THRESH_BINARY)
        
        for angle in [0, 2,-2]:
            rot_img = imutils.rotate_bound(image, angle)
            d = pytesseract.image_to_data(rot_img, output_type='data.frame')
            if d.shape[0] > 1:
                break
            
        # Drop rows with Nan values, create columns for right, bottom and group
        d.dropna(subset=['text'],inplace=True)
        d['right'] = d['left'] + d['width']
        d['bottom'] = d['top'] + d['height']
        d['group'] = d.groupby(['block_num','par_num','line_num']).ngroup()

        # Find the first clean group with high confidence and  more than two
        # letters and drop all text before that
        d['mean_conf'] = d.groupby('group')['conf'].transform('mean')
        d = d[d['mean_conf']>50]

        
        # Drop text where it extends to boundaries of the image
        d = d[(d['left']>0) & (d['right']<image.shape[1]) & 
              (d['top']>0) & (d['bottom']<image.shape[0])]

        # Drop rows where there is a | character becuase of background noise
        d['text'] = d['text'].apply(lambda x:re.sub(r'\||\s|_','',x))
        conf = d['conf'].mean()

        # Loop through the dataframe and adjust to optimize text grouping
        for i in range(d.shape[0]-1):    
            # If horizontal spacing between words is large, create new group
            spacing = d.iloc[i+1,d.columns.get_loc('left')] - \
            d.iloc[i,d.columns.get_loc('right')]
            if (spacing > 100) & (d.iloc[i,d.columns.get_loc('group')] == 
                d.iloc[i+1,d.columns.get_loc('group')]):
                d.iloc[i+1:,d.columns.get_loc('group')] +=1

        tokens = d.groupby('group')['text'].apply(lambda x:' '.join(x)).values
        #tokens = [x.strip() for x in tokens if re.search(r'\w',x)]
        ocr_text = '\n'.join(tokens)

    
        # Draw a bounding box around main text box
        cv2.rectangle(image, (d['left'].min()-20, d['top'].min()-20), 
                (d['right'].max()+20, d['bottom'].max()+20), (36,255,12), 3)

        
        #Write image to a file, print results and add results to a dataframe
        text_image = image[d['top'].min()-20:d['bottom'].max()+20,
                                        d['left'].min()-20:d['right'].max()+20]

        if (text_image.shape[0]==0) or (text_image.shape[1]==0):
            text_image = image
            
        if save_img:
            cv2.imwrite(str(barcode) + '.jpg', text_image)
             
        return barcode, ocr_text, conf, text_image

    except:
        return None, None, None, None
    
def printed_ocr_df(df, url_col, save_img = False):
    '''
    Input: Dataframe with urls, name of url column and whether image should
    be saved as jpg
    Output: Dataframe with columns for bacode and OCR Text added
    '''
    
    df1 = pd.DataFrame(columns=['Barcode','OCRTExt','Confidence'])
    for index, row in df.iterrows():
        bc, ocr, conf, img = printed_ocr(row[url_col], save_img)
        df1 = df1.append({'Barcode':bc, 'OCRText':ocr, 'Confidence':conf}, 
                         ignore_index=True)
    return pd.concat([df,df1], axis=1)

def handwritten_ocr_df(df, url_col, save_img = False):
    '''
    Input: Dataframe with urls, name of url column and whether image should
    be saved as jpg
    Output: Dataframe with columns for bacode and OCR Text added
    '''
    
    df1 = pd.DataFrame(columns=['Barcode','OCRText'])
    for index, row in df.iterrows():
        bc, oc, img = handwritten_ocr(row[url_col], save_img)
        df1 = df1.append({'OCRText':oc, 'Barcode': bc}, ignore_index=True)
    return pd.concat([df,df1], axis=1)
    
    


    