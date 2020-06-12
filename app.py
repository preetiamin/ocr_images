import ocr
import streamlit as st
import helper

st.title('Botanical Images Text Extraction')
type_text = st.radio('Type of Text', ('Printed', 'Handwritten'))

url = st.text_input('Enter url')

if url:
    if type_text =='Printed':
        barcode, text, img = ocr.printed_ocr(url)
        mod = helper.printed_model()
        if img is not None:
            st.image(img, width = 400, channels="BGR")
        if barcode:
            st.write('Barcode: ', barcode)
        if text:
            st.write('Species: ', mod.predict_species(text))
            st.write('Location: ', mod.predict_county(text),
                     ' County, ', mod.predict_state(text))
            st.write('Collected: ', mod.find_date(mod.clean_text(text)), 
                     ' by ', mod.predict_collector(text))
            st.write('OCR Text: ', text)
    else:
        
        barcode, text, img = ocr.handwritten_ocr(url)
        mod = helper.handwritten_model()
        if img is not None:
            st.image(img, width = 500, channels="BGR")
        if barcode:
            st.write('Barcode: ', barcode)
        if text:
            st.write('Species: ', mod.predict_species(text))
            st.write('Location: ', mod.predict_state(text))
            st.write('Year: ', mod.find_year(text))
            st.write('OCR Text: ', text)



