import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True)]


def is_folder(link):
    return link.endswith("/")


def download_file(file_url, local_path):
    try:
        # Check remote file size
        head = requests.head(file_url, allow_redirects=True)
        if head.status_code != 200:
            print(f"Failed to fetch headers: {file_url} (status: {head.status_code})")
            return

        remote_size = int(head.headers.get("Content-Length", 0))

        # Check local file status
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if local_size == remote_size:
                print(f"**This file is skipped (already downloaded): {file_url}")
                return
            else:
                print(f"File size mismatch — re-downloading: {file_url} (local: {local_size}, remote: {remote_size})")
        else:
            print(f"Downloading new file: {file_url}")

        # Download the file
        r = requests.get(file_url)
        r.raise_for_status()
        time.sleep(2)
        with open(local_path, "wb") as f:
            f.write(r.content)

        print(f"✔ Downloaded: {file_url}")

    except Exception as e:
        print(f"Error downloading {file_url}: {e}")

def crawl_and_download(base_url, current_dir):
    '''
    Downloads all files listed in subdirectories of base_url (an FTP sever such as https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn/) to current_dir.
    Requires global TARGET_EXTENSIONS to be set (recommend ".cdf").
    '''
    links = get_links(base_url)
    for link in links:
        full_url = urljoin(base_url, link)
        if (
            is_folder(link)
            and link not in ("../", "./")
            and not link.startswith("/pub/data/")
        ):
            # Make local folder
            sub_dir = os.path.join(current_dir, link.strip("/"))
            os.makedirs(sub_dir, exist_ok=True)
            # Recurse
            crawl_and_download(full_url, sub_dir)
        elif link.endswith(TARGET_EXTENSIONS):
            local_file_path = os.path.join(current_dir, link)
            download_file(full_url, local_file_path)

if __name__ == "__main__":
    """
    Scrapes a public SPDF FTP server and downloads all CDF files to a specified directory.
    """
    # Base URL to start from. 
    BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spe/l3/spe_sf0_pad/"

    # Directory to download data to.
    DOWNLOAD_DIR = "~/data/mms/fpi" #Directory to download data to. FTP server filestructure is maintained

    # File types to download. Here we choose CDFs.
    TARGET_EXTENSIONS = ".cdf"

    # Create base download directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Start crawling
    crawl_and_download(BASE_URL, DOWNLOAD_DIR)