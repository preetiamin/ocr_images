import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
#from sklearn.utils import shuffle
import numpy as np
import re
import json

import plac
import random
import warnings
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
nlp = spacy.load("en")


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    n_iter=("Number of training iterations", "option", "n", int),
)
def train_model(TRAIN_DATA, model=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    n_iter=100
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)
    return nlp


def evaluate_model(model, examples):
    '''
    Determine evaluation metrics for nlp model
    
    Parameters:
    model: nlp model
    examples: data to evaluate in spacy format
    
    Returns:
    Dictionary containing evaluation metrics
    '''
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

def predict(model, examples):
    # test the trained model
    for text, _ in examples:
        doc = model(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# save model to output directory
def save_model(model, model_filepath):
    '''
    Saves nlp model to output directory
    
    Parameters:
    model: nlp model
    model_filepath: directory to save model to 
    
    '''
    if model_filepath is not None:
        model_filepath = Path(model_filepath)
        if not model_filepath.exists():
            model_filepath.mkdir()
        model.to_disk(model_filepath)
        print("Saved model to", model_filepath)


## creating data in spacy data input format
def gen_data(X, Y):
    '''
    Generates spacy ner training data from X and Y
    
    Parameters: 
    X: dataframe containing single column of input text data
    Y: dataframe containing the 3 columns to be used as the entity outputs
    Returns:
    train_data: spacy ner formatted data 
    missing_data: data for rows where one or more entities were missing
    overlap_data: data containing rows where one ore more entities overlapped
    
    '''
    train_data = []
    missing_data = []
    overlap_data = []
    for i in range(X.shape[0]):
        text = X.iloc[i]
        if text:
            entlist = []
            text = str(text)
            is_occupied = np.array([0] * len(text))
            found_missing = False
            found_overlap = False
            for col in Y.columns:
                j = Y.columns.get_loc(col)
                ent = Y.iloc[i,j]
                if ent:
                    ent = str(ent)
                    match = re.search(r'[^a-zA-Z](' + re.escape(ent) + 
                                      ')[^a-zA-Z]', text, flags=re.IGNORECASE)
                    if match:
                        ent_x = match.start(1)
                        ent_y = ent_x + len(ent)
                        if sum(is_occupied[ent_x:ent_y]) > 0:
                            found_overlap = True
                        is_occupied[ent_x:ent_y] = 1
                        entlist.append([ent_x, ent_y, col])
                    else:
                        found_missing = True
            if found_missing == True:
                missing_data.append((text, {"entities": entlist}))
            if found_overlap == True:
                overlap_data.append((text, {"entities": entlist}))
            else:
                train_data.append((text, {"entities": entlist}))
    if missing_data or overlap_data:
        print('Found errors in data, see data_errors.txt')
        f = open("data_errors.txt", "w")
        f.write(json.dumps(missing_data))
        f.write(json.dumps(overlap_data))
        f.close()
        
    return train_data, missing_data, overlap_data


def load_data(database_filepath):
    '''
    Loads data from a pandas dataframe. Assigns X and Y to be used 
    to model the data
    
    Parameters: 
    database_filepath: Name of the pandas dataframe
    Returns:
    X: dataframe containing single column of input text data
    Y: dataframe containing the 3 columns to be used as the entity outputs
    
    '''
    df = pd.read_csv(database_filepath)
    colnum = df.columns.get_loc('OCRText')
    X = df['OCRText']
    Y = df.iloc[:,colnum+1:]
    return X, Y, list(Y.columns)


def main():
    if len(sys.argv) == 3:
        dataframe_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATAFRAME: {}'.format(dataframe_filepath))
        X, Y, category_names = load_data(dataframe_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
        print('Generting training data...')
        train_data, missing_data, overlap_data = gen_data(X_train, Y_train)
        
        print('Training model...')
        model = train_model(train_data)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
    
        print('Trained model saved!')
        
        print('Evaluating model...')
        
        test_data = gen_data(X_test, Y_test)[0]
        score = evaluate_model(model, train_data)
        print('Model Evaluation for Training Data:\n',score)
        
        score = evaluate_model(model, test_data)
        print('Model Evaluation for Test Data:\n',score)

    else:
        print('Please provide filepath of the OCRText dataframe as the first '\
              'argument and the filepath to save the model to as the second ' \
              'argument. \n\nExample: python ' \
              'train_ner.py ../data/train.csv ner_model')

if __name__ == '__main__':
    main()
