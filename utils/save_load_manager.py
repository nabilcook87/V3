# utils/save_load_manager.py

import json
import os

class SaveLoadManager:
    def __init__(self, save_folder="saved_projects"):
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def save_project(self, project_data, filename):
        """Save project dictionary to JSON file."""
        full_path = os.path.join(self.save_folder, filename + ".json")
        with open(full_path, 'w') as file:
            json.dump(project_data, file, indent=4)
        return f"✅ Project saved: {full_path}"

    def load_project(self, filename):
        """Load project dictionary from JSON file."""
        full_path = os.path.join(self.save_folder, filename + ".json")
        if not os.path.exists(full_path):
            return None, f"❌ File not found: {full_path}"
        with open(full_path, 'r') as file:
            project_data = json.load(file)
        return project_data, f"✅ Project loaded: {full_path}"

    def list_projects(self):
        """List all saved project files."""
        projects = []
        if os.path.exists(self.save_folder):
            for file in os.listdir(self.save_folder):
                if file.endswith(".json"):
                    projects.append(file[:-5])  # Remove ".json"
        return projects