import os
import re
import shutil

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def normalize_name(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    return text.strip('_')

def migrate():
    with open('README.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    papers_dir = 'papers'
    current_category = "uncategorized"
    
    moved_count = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            current_category = normalize_name(line.replace('## ', '').strip())
        elif line.startswith('- [') and '](' in line:
            match = re.search(r'- \[(.*?)\]\((.*?)\)', line)
            if match:
                title = match.group(1)
                url = match.group(2)
                
                ext = '.pdf'
                if 'github.com' in url and '.md' in url:
                    ext = '.md'
                    
                filename = sanitize_filename(title) + ext
                
                old_path = os.path.join(papers_dir, filename)
                new_dir = os.path.join(papers_dir, current_category)
                new_path = os.path.join(new_dir, filename)
                
                if os.path.exists(old_path):
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    shutil.move(old_path, new_path)
                    moved_count += 1
                    
    print(f"Migrated {moved_count} papers into category subdirectories.")

if __name__ == "__main__":
    migrate()
