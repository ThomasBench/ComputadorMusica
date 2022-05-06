import pandas as pd 
from bs4 import BeautifulSoup
from tqdm import tqdm
from driver import DRIVER

def songs_to_df(songArray): 
    return pd.DataFrame([s.__dict__ for s in songArray])
def get_source(link): 
    DRIVER.get(link)
    return DRIVER.page_source
def genre_extractor(filepath):
    dico = {}
    with open(filepath, "r", encoding = "utf-8") as f:
        data = BeautifulSoup(f, 'lxml').findAll('a')
    for elem in data:
        value = int(elem["value"])
        genre = elem.findAll("span")[0].text
        nb_songs = int(elem.findAll("span")[1].text.replace(',', ''))
        dico[genre] = { "value":value, "nb_songs" : nb_songs}
    return dico


GENRE_DICT = genre_extractor("./data/genre_selector.html")

def create_link(page,genre, date):
    return f"https://www.ultimate-guitar.com/explore?decade[]={date}&genres[]={GENRE_DICT[genre]['value']}&page={page}&type[]=Chords"


def pageExist(link):
    return len(BeautifulSoup(get_source(link), "lxml").findAll("div", class_ = "b-error-page wallpaper")) == 0

def get_songs_from_link( link):

    # 1. Retrieve the content of the landing page 
    contentTable = BeautifulSoup(get_source(link), 'lxml').findAll("div", class_ = "_3uKbA")

    # 2. Initialise the song array 
    songs = []

    # 3. Iterate through each song in the page to create a Song() object
    for content in tqdm(contentTable[1:]):
        songName = content.find("a", class_ = "_3DU-x JoRLr _3dYeW").text
        artist = content.find('a',"_3DU-x hn34w _3dYeW").text
        link = content.find('a',"_3DU-x JoRLr _3dYeW")['href']
        lyrics, chords = get_lyrics_chords_from_link( link)
        songs.append(Song(songName, artist, link, lyrics, chords))
    
    return songs


def get_lyrics_chords_from_link( link):

    # 1. Retrieve the content of the song page
    songData  = BeautifulSoup(get_source(link), "lxml").find("pre", class_ = "_3F2CP _3hukP")        # The _3F2CP _3hukP class seems to refer to the song content of the page

    # 2. Initialise the lyrics and chords arrays
    lyrics = ""
    chords = []

    # 3. Iterate over all "content span" of the html page
    for elem in songData.findAll("span", class_ = "_2jIGi"):     # the _2jIGi class seems to refer to the group of lyrics + chords among other things, later refered as lineContent
        lyricsChords = elem.findAll("span", class_ = "_3rlxz")   # Get all the lineContents
        if len(lyricsChords) == 2: # If there is more than 2 lines in one group, then it is not a Lyrics + Chords duet (can also be tabs, which we do not want) 
            chords += [e["data-name"] for e in lyricsChords[0].findAll()]
            lyrics += lyricsChords[1].text 
    
    return lyrics,chords





class Song:
    "Small Song class to store all the infos of a specific song, might evolve later"
    def __init__(self, songName, author, link, lyrics, chords):
        self.name = songName
        self.author = author
        self.link = link
        self.lyrics = lyrics
        self.chords = chords
    
