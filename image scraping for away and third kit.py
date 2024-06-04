import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#change below URL to whichever season and league you want to scrape
url = "https://www.footballkitarchive.com/ligue-1-2013-14-kits/"

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    
    kit_elements = soup.find_all("div", class_="kit")
    
    save_folder = "yourfolderhere"
    os.makedirs(save_folder, exist_ok=True)
    
    kit_types = ["Third", "Fourth", "Special", "Anniversary", "European"]
    
    for kit_element in kit_elements:
        team_name_element = kit_element.find(class_="kit-teamname")
        kit_season_element = kit_element.find(class_="kit-season")
        
        if team_name_element and kit_season_element:
            team_name = team_name_element.text.strip()
            kit_season = kit_season_element.text.strip()
            
            if any(kit_type.lower() in kit_season.lower() for kit_type in kit_types):
                img = kit_element.find("img")
                
                if img and 'src' in img.attrs:
                    img_url = urljoin(url, img["src"])
                    
                    img_name = img_url.split("/")[-1]
                    
                    save_path = os.path.join(save_folder, img_name)
                    
                    img_response = requests.get(img_url)

                    if img_response.status_code == 200:
                        with open(save_path, "wb") as f:
                            f.write(img_response.content)
                        print(f"Image '{img_name}' saved successfully for {team_name}: {kit_season}.")
                    else:
                        print(f"Failed to download image '{img_name}'. Status code: {img_response.status_code}")
                else:
                    print(f"Image tag does not exist or does not have a 'src' attribute for {team_name}: {kit_season}.")
        else:
            print("Team name element or kit season element not found.")
else:
    print("Failed to fetch web page. Status code:", response.status_code)
