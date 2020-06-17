#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:32:54 2020

@author: preetiamin
"""

import re
import spacy
import pandas as pd
from datetime import datetime

class printed_model:
    
    nlp_gen = spacy.load('en')
    nlp_cus = spacy.load('model/printed_model')
    # Load species names
    df_plant = pd.read_csv('data/PlantSpecies.csv')
    df_plant['Species'] = df_plant['Scientific Name with Author'].apply(
            lambda x: str(x).split()[0])
    species_list = [x for x in df_plant['Species'].dropna().unique().tolist()]
    
    # Load states names
    df_states = pd.read_csv('data/States.csv')
    states_list = [x.lower() for x in df_states['name']]
    
    def __init__(self, name='printed model'):
        self.name = name

    def predict_state(self, x):
        doc = self.nlp_cus(x)
        ents = doc.ents
        for ent in ents:
            if ent.label_ == 'State':
                return ent.text.title()
        return None
    
    def predict_county(self, x):
        doc = self.nlp_cus(x)
        ents = doc.ents
        for ent in ents:
            if ent.label_ == 'County':
                return ent.text.title()
        return None
    
    def predict_species(self, x):
        doc = self.nlp_cus(x)
        ents = doc.ents
        for ent in ents:
            if ent.label_ == 'Species':
                return ent.text
        return None
    
    def predict_collector(self, x):
        x = '\n'.join(x.split('\n')[::-1])
        doc = self.nlp_gen(x)
        ents = doc.ents
        for ent in ents:
            if (ent.label_ == 'PERSON') & (len(ent.text.split()) > 1):
                name = re.sub(r'( #)|( No)|[0-9]*','',ent.text)
                return name.title()
        return None
    
    def predict_date(self, x):
        x = '\n'.join(x.split('\n')[::-1])
        x = re.sub(r'\d{5,}','',x)
        doc = self.nlp_gen(x)
        ents = doc.ents
        for ent in ents:
            if (ent.label_ == 'DATE') & (len(ent.text.split()) > 1) & \
            (len(ent.text)>5):
                return(ent.text)
        return None
        
    def clean_text(self, x):
        '''
        Remove lines with no text, combine lines if 2nd starts with lowercase,
        split line at [# Eleve No], split after date, split after 5 or more
        digits, remove 'new york botanical garden'
        '''
        
        x = re.sub(r'(\n\s*\n)|:','\n',str(x))
        x = re.sub('(?:(\\n))+([a-z]+)',' \\2',x)
        x = re.sub(r'((of|and|on|with|from|&|,)\n{1})','\\2 ', x)
        x = re.sub(r'([^\n])( elev| #| no.| No.| Elev.)','\\1\n\\2', x)
        x = re.sub(r'([12][0-9]{3})( )','\\1\n', x)
        x = re.sub(r'([0-9]{5,})( )','\\1\n', x)
        x = re.sub(r'new york botanical garden','', x, flags=re.I)
    
        tokens = x.split('\n')
        tokens = [line.strip() for line in tokens if len(line)>5]
        return '\n'.join(tokens)
    
    def find_barcode(self, x):
        '''
        Find 8 digit number in the text
        
        Input: String to search within
        Output: 8-digit barcode
        '''
        matches = re.search(r"\b\d{8}\b",str(x))
        return matches[0].strip() if matches else None
    
    def find_state(self, x):
        '''
        Find state in the text, search top-down in string
        checking against list of states
        
        Input: String to search within
        Output: State
        '''
        tokens = x.split('\n')
        for token in tokens:
            matches = [i for i in self.states_list 
                       if i in re.sub(r'[^\w\s]','',token).lower()]
            if matches:
                return matches[0].title()
        return None
    
    def find_county(self, x):
        '''
        Find county in the text, search top-down looking for Co or County
        '''
        matches = re.search(r'[a-z]*[ ]*[a-z]+\sco([\.:,]|(?:unty)|\n)',
                            str(x),flags=re.I)
        if matches:
            re.sub(matches[0], matches[0]+'\n', x)
            return ' '.join(matches[0].strip().split()[:-1]).title()
        else:
            return None
    
    def find_species(self, x):
        '''
        Find county in the text, search top-down looking for 
        '''
        matches = re.search(r'\n([A-Z]{1,}[a-z]{3,}\s[a-z]{4,}).*\n',str(x))
        if matches:
            species = matches[1].strip() 
            if species.split()[0] in self.species_list:
                return species
        return None
    
    def find_date(self, x):
        rev_list = x.split('\n')[::-1]
        for item in rev_list:
            item_fixed = re.sub(r'(Sept)[.\s]','Sep ',item)
            for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%B %d, %Y', 
                        '%d %b %Y','%d %B %Y', '%d %b. %Y', '%d %B, %Y', 
                        '%B %d %Y', '%d-%m-%Y', '%m/%d/%y','%d-%b-%Y',
                        '%B %d. %Y', '%d %b. %Y', '%d %b %y', '%d %B %y'):
                try:
                    this_date = datetime.strptime(item_fixed,fmt)
                    return this_date.strftime("%m/%d/%Y")
                except:
                    pass
        return None
    
    def find_collector(self, x):
        rev_list = x.split('\n')[::-1]
        for item in rev_list:
            item = re.sub(r'\d+|','',item)           # remove digits
            item = re.sub(r'[^\w\s]','',item)        # remove punctuation
            item = re.sub(r' (and|&) ',' ',item)     # remove and,&
            if (item.istitle()) & (len(item.split()) >1) & ('herbarium' not in item.lower()):
                return item.strip()
        return None
    
class handwritten_model:
    
    nlp_gen = spacy.load('en')
    nlp_cus = spacy.load('model/handwritten_model')
    
    def __init__(self, name='model'):
        self.name = name

    def predict_state(self, x):
        doc = self.nlp_cus(x)
        ents = doc.ents
        for ent in ents:
            if ent.label_ == 'State':
                return ent.text.title()
        return None
    
    def predict_species(self, x):
        doc = self.nlp_cus(x)
        ents = doc.ents
        for ent in ents:
            if ent.label_ == 'Species':
                return ent.text
        return None
    
    
    def find_year(self, x):
        dates = re.findall(r'[^\d](1[89]\d{2})[^\d]',x)
        if dates:
            dates.sort()
            return int(dates[0])
        return None