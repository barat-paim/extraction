# image extraction from zara webpage

from playwright.sync_api import sync_playwright
import time

def extract_image_url(page_url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        high_res_images = set()  # Use set to avoid duplicates
        
        def handle_response(response):
            try:
                url = response.url
                # Check for product images with w=563
                if ('static.zara.net/assets' in url and 
                    '.jpg' in url and 
                    'w=563' in url):
                    print(f"Found high-res image: {url}")
                    high_res_images.add(url)
            except Exception as e:
                print(f"Error handling response: {e}")
        
        page.on("response", handle_response)
        print("Navigating to page...")
        page.goto(page_url)
        print("Waiting for network idle...")
        page.wait_for_load_state('networkidle')
        time.sleep(3)  # Wait for any delayed image loads
        browser.close()
        
        return list(high_res_images)

url = 'https://www.zara.com/us/en/dtrt-jckt-13-p04164921.html?v1=405265154&v2=2467336'

print("Starting image extraction...")
images = extract_image_url(url)
if images:
    print(f"\nFound {len(images)} high-resolution images:")
    for url in images:
        print(f"Image URL: {url}")
else:
    print("\nNo high-resolution images found")
