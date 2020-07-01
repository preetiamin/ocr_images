# ocr_images
Entity Extraction from botanical images using Optical Character Recognition (OCR) and Natural Language Processing (NLP)

## Introduction
This project extracts raw text from botanical specimen images using OCR with Google Pytesseract and Google Vision API; it then extracts entities (barcode, location, species name, collector, date) from this raw text.

## Project Motivation
The project was completed for startup solving the problem of entity extraction for a huge number of scanned images at a botanical garden.

## Files

- model<br/>
 -- train_ner.py - Train custom NER model in SpaCy<br/>
 -- ocr.py - OCR extraction using PyTesseract and Google Vision API<br/>
 -- helper.py - ntity extraction model inference<br/>
- data<br/>
 -- handwritten_train.csv - Train data to train handwritten model<br/>
 -- printed_train.csv - Train data to train printed model<br/>
 -- printed_raw_ocr_data.csv - Test data for model prediction/evaluation<br/>
 -- States.csv. Lookup table for states<br/>
 -- PlantSpecies.csv - Lookup table for plant species names<br/>
- Handwritten_Prediction_Analysis.ipynb - Prediction/evaluation for handwritten text images<br/>
- Printed_Prediction_Analysis.ipynb - Prediction/evaluation for printed text images<br/>
- requirements.txt - List of libraries required<br/>
- app.py. Streamlit app<br/>

## Instructions for running the scripts:
1. Run the following commands in the project's root directory to set up your database and model.

    - python model/train_ner.py data/printed_train.csv model/printed_model
    - python model/train_ner.py data/handwritten_train.csv model/handwritten_model

2. Run the following command in the app's directory to run your web app.
    
    - streamlit run app.py

3. Go to http://0.0.0.0:8502/

## Examples

Below are examples of entity extraction with the streamlit app.

![SS 1](images/SS_1.png)
![SS 2](images/SS_2.png)

## Results

Printed Text Model Evaluation

![Printed](images/Printed_Results.png)

Sources of Error

![Error](images/Printed_Errors.png)

Handwritten Text Model Evaluation

![Handwritten](images/Handwritten_Results.png)




