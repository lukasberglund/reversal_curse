import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.common import attach_debugger, save_to_txt

NUM_CELEBRITIES = 10000
PAGE_LENGTH = 50
SAVE_DIR = "data/celebrity_relations"


def get_link(start_num: int) -> str:
    return f"https://www.imdb.com/search/name/?match_all=true&start={start_num}&ref_=rlm"


def extract_names(url: str) -> list[str]:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    matches = []
    for h3_tag in soup.find_all("h3", class_="lister-item-header"):
        for a in h3_tag.find_all("a", href=lambda href: href and href.startswith("/name/")):
            matches.append(a)

    names = [match.text.strip() for match in matches]
    names = [name for name in names if name != ""]

    assert len(set(names)) <= PAGE_LENGTH
    if len(set(names)) < PAGE_LENGTH - 5:
        print(f"WARNING: Only found {len(set(names))} names on page {url}")

    return names


if __name__ == "__main__":
    attach_debugger()

    names = []
    for i in tqdm(range(0, NUM_CELEBRITIES, PAGE_LENGTH)):
        url = get_link(i)
        names.extend(extract_names(url))

    save_to_txt(names, os.path.join(SAVE_DIR, "top_celebrities.txt"))
