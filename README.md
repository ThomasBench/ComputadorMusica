# Computador e Musica, a model to help song writers
In this repository you will find several notebooks for several key parts of the project:  
- Data Scrapping : dataExtractor 
    - the `dataExtractor.ipynb` notebook was used to scrap both chords and lyrics from [Ultimate Guitar](https://www.ultimate-guitar.com/explore) into exploitable format
    - the `helpers.py` provides several functions that were used to scrap the dataset

- Pre-processing : dataProcessor
    - the `dataProcessor.ipynb` notebook was used to pre-process the data obtained through the dataExtractor pipeline 

- Data Analysis :  chordAnalysis , lyricsAnalysis & preliminaryAnalysis
    - The `chordAnalysis.ipnyb` notebook was used to analyse the chords progressions from the dataset
    - The `lyricsAnalysis.ipnyb` notebook was used to analyse the lyrics from the dataset
    - The `preliminaryAnalysis.ipnyb` notebook was used to analyse thoe chords progressions as well
    
- Modelling : dataTrain & dataPredict
    - The `dataTrain.ipnyb` notebook was used to train the chosen models 
    - The `model_helpers.py` file provides helpers function to train the model and the `rnn_model.py` and `transformer_model.py` files both contains models that were tried during this project. Only the RNN model was kept for the prototype
    - the `dataPredict.ipynb` notebook was used to load the trained model and predict a 4 chord sequence for the lyrics that was chosen 


