import schedule
import time
import logging
import os
import requests
import subprocess
import json
from datetime import datetime
from flask import Flask, jsonify, request

# Ensure the 'data' directory exists
log_dir = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup logging
logging.basicConfig(
    filename=os.path.join(log_dir, "task_logs.txt"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

class TaskScheduler:
    def __init__(self):
        self.tasks = []  # Store scheduled tasks

    def add_task(self, task_func, interval, unit='seconds', task_name='Unnamed Task'):
        """Schedule a new task"""
        if unit == 'seconds':
            schedule.every(interval).seconds.do(self.run_task, task_func, task_name)
        elif unit == 'minutes':
            schedule.every(interval).minutes.do(self.run_task, task_func, task_name)
        elif unit == 'hours':
            schedule.every(interval).hours.do(self.run_task, task_func, task_name)
        elif unit == 'days':
            schedule.every(interval).days.do(self.run_task, task_func, task_name)
        
        self.tasks.append(task_name)
        logging.info(f"Task '{task_name}' scheduled every {interval} {unit}.")
        print(f"Task '{task_name}' scheduled every {interval} {unit}.")
    
    def run_task(self, task_func, task_name):
        """Execute the scheduled task"""
        logging.info(f"Executing task: {task_name}")
        print(f"Running task: {task_name} at {datetime.now()}...")
        task_func()
    
    def start_scheduler(self, timeout=None):
        """Run the scheduler in a loop with optional timeout"""
        print("Bob's Scheduler is now running...")
        start_time = time.time()
        while True:
            schedule.run_pending()
            time.sleep(1)
            if timeout and time.time() - start_time > timeout:
                print("Stopping scheduler after timeout.")
                break

# Define GitHub trending AI repo tracking
tracked_repos_file = os.path.join(log_dir, "tracked_repos.txt")
search_topics = ["AI", "Artificial Intelligence", "Automation", "Machine Learning", "Deep Learning"]
analysis_results_file = os.path.join(log_dir, "repo_analysis.json")
bob_versions_dir = os.path.join("bob_versions")
approved_versions_file = os.path.join(log_dir, "approved_versions.json")

if not os.path.exists(bob_versions_dir):
    os.makedirs(bob_versions_dir)
if not os.path.exists(approved_versions_file):
    with open(approved_versions_file, "w") as f:
        json.dump([], f)

def load_tracked_repos():
    if not os.path.exists(tracked_repos_file):
        return set()
    with open(tracked_repos_file, "r") as file:
        return set(line.strip() for line in file.readlines())

def save_tracked_repos(repos):
    with open(tracked_repos_file, "w") as file:
        for repo in repos:
            file.write(repo + "\n")

def clone_repository(repo_url):
    repo_name = repo_url.split('/')[-1]
    clone_path = os.path.join("cloned_repos", repo_name)
    if not os.path.exists(clone_path):
        print(f"Cloning repository: {repo_url}")
        logging.info(f"Cloning repository: {repo_url}")
        subprocess.run(["git", "clone", repo_url, clone_path])
        analyze_repository(clone_path, repo_name)
        propose_new_bob_version(repo_name, clone_path)

def analyze_repository(repo_path, repo_name):
    """Analyze the cloned repository for useful AI code."""
    analysis_results = {}
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".ipynb"):
                with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if any(keyword in content for keyword in ["tensorflow", "torch", "scikit-learn", "ml", "ai"]):
                        analysis_results[file] = content[:500]  # Store the first 500 characters as preview
    
    if analysis_results:
        with open(analysis_results_file, "a") as af:
            json.dump({repo_name: analysis_results}, af, indent=4)
        logging.info(f"Analyzed {repo_name} and stored results.")
        print(f"Analysis complete for {repo_name}, results stored.")

def check_github_trends():
    url = "https://api.github.com/search/repositories?q=" + "+".join(search_topics) + "&sort=stars&order=desc"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json().get('items', [])
        repo_names = set(repo['full_name'] for repo in repos)
        
        tracked_repos = load_tracked_repos()
        new_repos = repo_names - tracked_repos
        
        if new_repos:
            print(f"New trending AI/Automation repositories found: {len(new_repos)}")
            logging.info(f"New trending AI/Automation repositories found: {len(new_repos)}")
            for repo in new_repos:
                repo_url = f"https://github.com/{repo}"
                logging.info(f"New Trending Repo: {repo_url}")
                print(f"New Trending Repo: {repo_url}")
                clone_repository(repo_url)
            save_tracked_repos(repo_names)
        else:
            print("No new trending AI/Automation repositories found.")
            logging.info("No new trending AI/Automation repositories found.")
    else:
        print("Failed to fetch GitHub trends.")
        logging.error("Failed to fetch GitHub trends.")

if __name__ == "__main__":
    bob_scheduler = TaskScheduler()
    bob_scheduler.add_task(check_github_trends, 60, 'seconds', 'GitHub AI/Automation Trend Tracker')
    bob_scheduler.start_scheduler()
