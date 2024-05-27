import streamlit as st 
import time

# Titel eingeben
st.title('Die erste App in den LifeSciences')

#Text eingeben
st.write('In dieser App kann man das maschinelle Lernen anhand des k-nearest neighbor Algorithmus ausprobieren')

# Nutzer Input als Text
nutzer_name = st.text_input('Bitte gib deinen Namen ein:')

# Aktualisierung mit Nutzerinput
st.write('Herzlich Willkommen ' + nutzer_name + '. Schön, dass du die App ausprobierst')

# Browserfenster in Spalten aufteilen
col1, col2, col3 = st.columns(3)

col1.write('Das ist die 1. Spalte')
col2.write('Das ist die 2. Splate')
col3.write('Das ist die 3. Spalte')

col1.markdown('# Erste Überschrift')
col1.markdown('## Unter-Überschrift')

# File hochladen lassen
uploaded_file = col2.file_uploader('Upload a file!')

# Kamera anspielen
photoshot = col3.camera_input('Nimm ein Foto von Dir auf!')

#col1.image(photoshot)

# Progress bar
progress_bar = col2.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i)
    
col2.success('Das Foto wurde erfolgreich hochgeladen')

# Metrische Angaben schön darstellen
col3.metric(label='Aktuelle Lebenserwartung', value='3500s', delta= '-120s')

# Arbeiten mit Expanders
with st.expander('Click to expand'):
    st.write('Hallo')
    if photoshot is not None:
        st.image(photoshot)
    
    st.code('y = m * c')
    
