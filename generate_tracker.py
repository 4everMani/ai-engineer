import os
import re

def normalize_name(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    return text.strip('_')

def generate_tracker():
    with open('README.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    tracker_lines = [
        "# 📚 AI Engineering Mastery Tracker\n",
        "Track your progress dissecting papers and building mini-projects here.\n\n",
        "| Category | Paper Title | 🧠 Dissected | 🛠️ Project Built | 📝 Notes |\n",
        "|----------|-------------|--------------|-------------------|----------|\n"
    ]
    
    current_category = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            if current_category is not None:
                tracker_lines.append(f"| {current_category} | **[MILESTONE] Category Synthesis Session** | ⏳ Pending | - | - |\n")
                if current_category == "Core Architecture":
                    tracker_lines.append(f"| {current_category} | **[PROJECT] Build RAG Memory System for Agent** | ⏳ Pending | ⏳ Pending | - |\n")
            current_category = line.replace('## ', '').strip()
        elif line.startswith('- [') and '](' in line:
            match = re.search(r'- \[(.*?)\]\((.*?)\)', line)
            if match:
                title = match.group(1)
                url = match.group(2)
                
                cat_dir = normalize_name(current_category)
                file_name = normalize_name(title)
                
                notes_path = f"notes/{cat_dir}/{file_name}.md"
                code_path = f"projects/{cat_dir}/{file_name}.py"
                
                status_dissect = "✅ Done" if os.path.exists(notes_path) else "⏳ Pending"
                status_project = "✅ Done" if os.path.exists(code_path) else "⏳ Pending"
                notes = f"[Notes]({notes_path})" if os.path.exists(notes_path) else "-"
                
                tracker_lines.append(f"| {current_category} | [{title}]({url}) | {status_dissect} | {status_project} | {notes} |\n")
                
    if current_category is not None:
        tracker_lines.append(f"| {current_category} | **[MILESTONE] Category Synthesis Session** | ⏳ Pending | - | - |\n")
                
    with open('learning_tracker.md', 'w', encoding='utf-8') as f:
        f.writelines(tracker_lines)
        
    print("Tracker generated at learning_tracker.md")

if __name__ == '__main__':
    generate_tracker()
