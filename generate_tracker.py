import re

def generate_tracker():
    with open('README.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    tracker_lines = [
        "# 📚 AI Engineering Mastery Tracker\n",
        "Track your progress dissecting papers and building mini-projects here.\n\n",
        "| Category | Paper Title | 🧠 Dissected | 🛠️ Project Built | 📝 Notes |\n",
        "|----------|-------------|--------------|-------------------|----------|\n"
    ]
    
    current_category = "General"
    
    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            current_category = line.replace('## ', '').strip()
        elif line.startswith('- [') and '](' in line:
            # Extract title and url
            match = re.search(r'- \[(.*?)\]\((.*?)\)', line)
            if match:
                title = match.group(1)
                url = match.group(2)
                
                # Check if it's the BPE paper which we already did
                if "Byte-pair Encoding" in title:
                    status_dissect = "✅ Done"
                    status_project = "⏳ Pending"
                    notes = "[BPE Notes](notes/tokenization/byte_pair_encoding.md)"
                else:
                    status_dissect = "⏳ Pending"
                    status_project = "⏳ Pending"
                    notes = "-"
                    
                tracker_lines.append(f"| {current_category} | [{title}]({url}) | {status_dissect} | {status_project} | {notes} |\n")
                
    with open('learning_tracker.md', 'w', encoding='utf-8') as f:
        f.writelines(tracker_lines)
        
    print("Tracker generated at learning_tracker.md")

if __name__ == '__main__':
    generate_tracker()
