import re
import json
import requests
from tqdm import tqdm
import multiprocessing
from bs4 import BeautifulSoup


def getImageUrl(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features="html.parser")
        meta = soup.find('meta', attrs={'property': 'og:image'}) or \
               soup.find('meta', attrs={'property': 'twitter:image'})[0]
        meta = str(meta)
        matches = re.finditer("(https?):\/\/(www\.)?[a-z0-9\.:].*?(?=\s)", meta)
        for matchNum, match in enumerate(matches, start=1):
            return str(match.group()).replace('\"', '')
    except:
        return None


def saveImage(image_url, image_name, folder='dev'):
    ext = image_url.split('/')[-1].split('.')[-1]
    save_path = folder + '/' + image_name + '.' + ext
    img_data = requests.get(image_url).content

    pbar.update(1)
    with open(save_path, 'wb') as handler:
        handler.write(img_data)


def process_(d, folder):
    url = d['image_url']
    image_url = getImageUrl(url)
    image_name = d['image_id']
    if image_url: saveImage(image_url, image_name, folder)


folder = './data/test/'
data = json.load(open('./data/test/testset_images_metadata.json'))['images']
print(len(data))
sfds
pbar = tqdm(folder)

pool = multiprocessing.Pool(processes=40)
for d in data:
    pool.apply_async(process_, args=(d, folder))

pool.close()
pool.join()