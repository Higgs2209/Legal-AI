from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import time
import os

# URL Definitions
base_url = "https://www.austlii.edu.au/cgi-bin/viewdb/au/cases/vic/VSC/"
forms_page = ""

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.headless = True
driver = webdriver.Chrome(options=options)

def download_file(url, directory, filename):
    response = requests.get(url)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), 'wb') as file:
        file.write(response.content)

def download_forms():
    page = 0
    base_directory = 'supreme_court_forms'

    while True:
        driver.get(f"{base_url}{forms_page}{page}")
        forms = driver.find_elements(By.XPATH, "//a[contains(@href, '.rtf') or contains(@href, '.docx')]")

        if not forms:
            print("No more forms found.")
            break

        for form in forms:
            file_url = form.get_attribute('href')
            file_name = file_url.split('/')[-1]
            directory = os.path.join(base_directory, file_name.split('.')[0])

            print(f"Downloading {file_name}...")
            download_file(file_url, directory, file_name)

        page += 1
        time.sleep(2) # Sleep to prevent overwhelming the server

    driver.quit()

download_forms()
