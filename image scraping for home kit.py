import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#change below webpage to whichever season and leage you want to scrape
url = "https://www.footballkitarchive.com/premier-league-2003-04-kits/"

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    
    kit_elements = soup.find_all("div", class_="kit")
    
    save_folder = "yourfolderhere"
    os.makedirs(save_folder, exist_ok=True)
    
    for kit_element in kit_elements:
        team_name_element = kit_element.find(class_="kit-teamname")
        kit_season_element = kit_element.find(class_="kit-season")
        
        if team_name_element and kit_season_element:
            team_name = team_name_element.text.strip()
            kit_season = kit_season_element.text.strip()
            
            if "home" in kit_season.lower() or "away" in kit_season.lower():
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

