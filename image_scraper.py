import os
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys

from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import json
import time

chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--disable-web-security')
chrome_options.add_argument('--allow-running-insecure-content')
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Path to ChromeDriver
chromedriver_path = "C:/Users/bekarm/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"  # Update with actual path

# ChromeDriver service
service = Service(chromedriver_path)

# WebDriver instance
driver = webdriver.Chrome(service=service, options=chrome_options)

base_url = ("https://www.cars.com/shopping/results/?dealer_id=&include_shippable=false&keyword=&list_price_max=35000&list_price_min=4000&makes%5B%5D={brand}&maximum_distance=100&mileage_max=&page_size=20&sort=best_match_desc&stock_type=used&year_max=2020&year_min=2008&zip=12180&page={page_num}")
vehicle_url_template = "https://www.cars.com/vehicledetail/{listing_id}"

popular_brands = ['toyota', 'honda', 'ford', 'bmw', 'mercedes_benz', 'audi', 'nissan']

# Folder to save images
image_folder = "vehicle_images_more"
os.makedirs(image_folder, exist_ok=True)

# Folder to save brand-specific CSV files
csv_folder = "brand_csvs_more"
os.makedirs(csv_folder, exist_ok=True)

# Download images from URL
def download_image(image_url, save_path, resize_to=(224, 224)):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as img:
            img = img.convert("RGB")  # Ensure image is in RGB format
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(save_path, "JPEG")
    except Exception as e:
        print(f"Error downloading or resizing image {image_url}: {e}")

# Scrape images for a specific vehicle
def scrape_vehicle_images(listing_id):
    vehicle_url = f"https://www.cars.com/vehicledetail/{listing_id}"
    print(f"Accessing URL: {vehicle_url}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            driver.set_page_load_timeout(10)
            driver.get(vehicle_url)
            break
        except TimeoutException:
            print(f"Timeout while accessing {vehicle_url}. Retrying ({attempt + 1}/{max_retries})...")
            if attempt == max_retries - 1:
                print(f"Failed to load {vehicle_url} after {max_retries} attempts.")
                return []

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//gallery-slides"))
        )
        image_elements = driver.find_elements(By.XPATH, "//gallery-slides//img[contains(@class, 'image-index')]")
        if not image_elements:
            print(f"No images found in gallery for {listing_id}.")
            return []
        image_elements = image_elements[:3]
        image_urls = [img.get_attribute("src") for img in image_elements if img.get_attribute("src")]

        downloaded_images = []
        for idx, image_url in enumerate(image_urls):
            image_name = f"{listing_id}_{idx + 1}.jpg"
            save_path = os.path.join(image_folder, image_name)
            download_image(image_url, save_path)
            downloaded_images.append(save_path)
        return downloaded_images

    except TimeoutException:
        print(f"Timeout while waiting for gallery-slides for {listing_id}.")
        return []
    except NoSuchElementException as e:
        print(f"Error finding elements for {listing_id}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error for {listing_id}: {e}")
        return []

# Scrape data from each page
def scrape_page(brand, page_num):
    driver.get(base_url.format(brand=brand, page_num=page_num))
    time.sleep(1)
    listings_page_div = driver.find_element(By.CLASS_NAME, 'listings-page')
    data = listings_page_div.get_attribute("data-site-activity")
    listings = json.loads(data).get("vehicleArray", [])
    cars_data = []
    for car in listings:
        listing_id = car.get("listing_id", "Unknown")
        print(f"Scraping listing {listing_id}")
        images = scrape_vehicle_images(listing_id) if listing_id != "Unknown" else []
        car_data = {
            'Brand': car.get('make', 'Unknown'),
            'Model': car.get('model', 'Unknown'),
            'Year': car.get('year', 'Unknown'),
            'Price': car.get('price', 'Unknown'),
            'Mileage': car.get('mileage', 'Unknown'),
            'Exterior Color': car.get('exterior_color', 'Unknown'),
            'Interior Color': car.get('interior_color', 'Unknown'),
            'Fuel Type': car.get('fuel_type', 'Unknown'),
            'Drivetrain': car.get('drivetrain', 'Unknown'),
            'VIN': car.get('vin', 'Unknown'),
            'Dealer Name': car.get('dealer_name', 'Unknown'),
            'Dealer Zip': car.get('dealer_zip', 'Unknown'),
            'Bodystyle': car.get('bodystyle', 'Unknown'),
            'Image Paths': images
        }
        cars_data.append(car_data)
    return cars_data

# Get maximum pages
def get_max_pages(driver):
    try:
        pagination_element = driver.find_element(By.XPATH, '//spark-pagination')
        page_links = pagination_element.find_elements(By.XPATH, './/a[@phx-value-page]')
        max_pages = max(int(link.get_attribute("phx-value-page")) for link in page_links)
        return max_pages
    except Exception as e:
        print(f"Error while extracting max pages: {e}")
        return 1

# Scrape brand and save per page
def scrape_brand(brand):
    all_cars = []
    driver.get(base_url.format(brand=brand, page_num=1))
    max_pages = get_max_pages(driver)
    max_pages = max_pages - 2  # Ensure at least 1 page
    print(f"Found {max_pages} pages for {brand}")
    for page_num in range(1, max_pages + 1):
        print(f"Scraping {brand}, Page {page_num}")
        cars = scrape_page(brand, page_num)
        if not cars:
            break
        all_cars.extend(cars)
        # Save to CSV per page
        page_df = pd.DataFrame(cars)
        page_csv_path = os.path.join(csv_folder, f"{brand}_page_{page_num}.csv")
        page_df.to_csv(page_csv_path, index=False)
        print(f"Saved {brand} page {page_num} to {page_csv_path}")
    return all_cars

# Main Scraping Loop
all_data = []
for brand in popular_brands:
    brand_data = scrape_brand(brand)
    all_data.extend(brand_data)
    # Save brand-specific CSV
    brand_df = pd.DataFrame(brand_data)
    brand_csv_path = os.path.join(csv_folder, f"{brand}_data.csv")
    brand_df.to_csv(brand_csv_path, index=False)
    print(f"Saved all data for {brand} to {brand_csv_path}")

# Combine all brand CSVs into one
all_data_df = pd.DataFrame(all_data)
all_data_df.to_csv('all_brands_combined_more.csv', index=False)
print("Data scraping and saving complete!")
