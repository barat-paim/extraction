# image extraction from zara webpage

from playwright.sync_api import sync_playwright

def extract_image_url(page_url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        # Store image URLs
        image_urls = []
        
        def handle_response(response):
            try:
                url = response.url
                if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    print(f"Found image response: {url}")
                    image_urls.append(url)
            except Exception as e:
                print(f"Error handling response: {e}")
        
        # Listen to all responses
        page.on("response", handle_response)
        
        print("Navigating to page...")
        page.goto(page_url)
        print("Waiting for network idle...")
        page.wait_for_load_state('networkidle', timeout=30000)
        print("Network idle reached")
        
        # Add a small delay to ensure we capture everything
        page.wait_for_timeout(5000)
        
        browser.close()
        return image_urls

url = 'https://www.zara.com/us/en/dtrt-jckt-13-p04164921.html?v1=405265154&v2=2467336'

# Example usage
print("Starting image extraction...")
image_urls = extract_image_url(url)
if image_urls:
    print(f'\nFound {len(image_urls)} images:')
    for url in image_urls:
        print(f'Image URL: {url}')
else:
    print('\nNo images found')
