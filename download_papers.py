import os
import re
import urllib.request
import urllib.error

def sanitize_filename(name):
    # Remove invalid characters for windows filenames
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

import ssl

def download_papers():
    readme_path = 'README.md'
    papers_dir = 'papers'
    
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all markdown links: [Title](URL)
    links = re.findall(r'\[(.*?)\]\((.*?)\)', content)
    
    print(f"Found {len(links)} links in the README.")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for title, url in links:
        # Convert arxiv /abs/ to /pdf/ if needed
        if 'arxiv.org/abs/' in url:
            url = url.replace('/abs/', '/pdf/')
            
        # Determine file extension based on URL, default to pdf for arxiv
        ext = '.pdf'
        if 'github.com' in url and '.md' in url:
            ext = '.md'
            # Convert github blob to raw for downloading
            url = url.replace('/blob/', '/raw/')
            
        filename = sanitize_filename(title) + ext
        filepath = os.path.join(papers_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Already exists: {filename}")
            continue
            
        print(f"Downloading: {filename} from {url}")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, context=ctx) as response, open(filepath, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            print(f"Successfully downloaded: {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == '__main__':
    download_papers()
