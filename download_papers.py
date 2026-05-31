import os
import re
import urllib.request
import urllib.error

def sanitize_filename(name):
    # Remove invalid characters for windows filenames
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

import ssl

def normalize_name(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    return text.strip('_')

def download_papers():
    readme_path = 'README.md'
    papers_dir = 'papers'
    
    if not os.path.exists(papers_dir):
        os.makedirs(papers_dir)

    with open(readme_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    current_category = "uncategorized"

    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            current_category = normalize_name(line.replace('## ', '').strip())
        elif line.startswith('- [') and '](' in line:
            match = re.search(r'- \[(.*?)\]\((.*?)\)', line)
            if match:
                title = match.group(1)
                url = match.group(2)
                
                # Convert arxiv /abs/ to /pdf/ if needed
                if 'arxiv.org/abs/' in url:
                    url = url.replace('/abs/', '/pdf/')
                    
                ext = '.pdf'
                if 'github.com' in url and '.md' in url:
                    ext = '.md'
                    url = url.replace('/blob/', '/raw/')
                    
                filename = sanitize_filename(title) + ext
                cat_dir = os.path.join(papers_dir, current_category)
                
                if not os.path.exists(cat_dir):
                    os.makedirs(cat_dir)
                    
                filepath = os.path.join(cat_dir, filename)
                
                if os.path.exists(filepath):
                    print(f"Already exists: {current_category}/{filename}")
                    continue
                    
                print(f"Downloading: {filename} to {current_category}/")
                
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
