import os
from argparse import ArgumentParser

import requests
from bs4 import BeautifulSoup
from diffusers import DiffusionPipeline

URL = "https://www.stw-ma.de/en/men%C3%BCplan_schlossmensa.html"


pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()


def menu_to_image(menu: str):
    return pipe(f"{menu} at German Uni Mensa").images[0]


def main(o: str):
    res = requests.get(URL)
    soup = BeautifulSoup(res.text, "html.parser")

    for i, menu_tag in enumerate(
        soup.find_all("td", class_="speiseplan-table-menu-content")[:2]
    ):
        menu = menu_tag.text.strip()
        image = menu_to_image(menu)
        image.save(os.path.join(o, f"{i}.jpg"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", type=str, default="./")
    args = parser.parse_args()
    main(args.o)
