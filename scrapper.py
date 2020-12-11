import re
import unicodedata

import nltk
from nltk.corpus import stopwords

from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

from tqdm import trange

nltk.download('stopwords')


def normalize(text):
    text = re.sub("\d+", "", text) # number removing
    text = re.sub(" {2,}", " ", text).lower()  # space removing
    text = re.sub("[^\w\s]", "", text) # special symbols removing
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore')  # symbol replacement
    return text.decode("utf-8")


def remove_stop_words(text):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    clean = [w for w in words if not w in stop_words]
    return ' '.join(clean)


def clean_collected(images: list):
    for image in images:
        # tags = remove_stop_words(normalize(image['alt']))
        tags = normalize(image['alt'])
        if 'template' in tags:
            print(tags)
            print(image)


if __name__ == '__main__':
    data = list()

    for i in trange(10):
        try:
            html = urlopen(f'https://www.freepik.com/search?dates=any&format=search&page={i}&query=tattoo+sketch')
        except HTTPError as error:
            print(f'Page number {i} doesnt exist!!')
            break
        bs = BeautifulSoup(html, 'html.parser')
        images = bs.find_all('img', {'src':re.compile('.jpg')})
        data.extend(images)

    print(len(data))
    clean_list = list()
    for i in data:
        if 'set' not in i['alt'].lower() and 'collection' not in i['alt'].lower() and 'template' not in i['alt'].lower():
            clean_list.append(i)

    print(len(clean_list))
    # clean_collected(images)
    # for image in images:
    #     print(image)
    #     print(image['src']+'\n')
