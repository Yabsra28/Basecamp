import streamlit as st
import requests
import json
import time
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import socket
import socketserver
import webbrowser
import http.server
import urllib.parse
import re

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(
    filename='basecamp_gemini_app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CONFIG_FILE = "config.json"
BASE_URL = "https://3.basecampapi.com"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 5
RATE_LIMIT_DELAY = 0.2
DEBUG_MODE = True
API_KEY = "AIzaSyArWCID8FdgwcFJpS_mUJNlLy6QJhMvf5w"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

# Load configuration
def load_config() -> Dict:
    try:
        if not os.path.exists(CONFIG_FILE):
            config_data = {
                "CLIENT_ID": "50a101d6c62602fa263cba350201cf7ab27ab618",
                "CLIENT_SECRET": "a92a6a7b7ad2127750dd440b5e54d0b626461b1f",
                "ACCESS_TOKEN": None
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config_data, f)
            logging.info("Created default config.json")
            return config_data
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        logging.info("Loaded config.json successfully")
        return config
    except Exception as e:
        logging.error(f"Config loading error: {str(e)}")
        st.error(f"Config error: {str(e)}")
        return {}

def save_config(config: Dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        logging.info("Saved config.json successfully")
    except Exception as e:
        logging.error(f"Config saving error: {str(e)}")
        st.error(f"Config saving error: {str(e)}")

CONFIG = load_config()
CLIENT_ID = CONFIG.get("CLIENT_ID")
CLIENT_SECRET = CONFIG.get("CLIENT_SECRET")
REDIRECT_URI = None
ACCESS_TOKEN = CONFIG.get("ACCESS_TOKEN")

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                logging.info(f"Found available port: {port}")
                return port
            except OSError:
                port += 1
    logging.error(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")
    raise OSError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

def retry_request(func, *args, max_retries=MAX_RETRIES, backoff_factor=2, **kwargs):
    for attempt in range(max_retries):
        try:
            response = func(*args, **kwargs)
            if response.status_code in [200, 201]:
                return response
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logging.warning(f"Rate limit hit, retrying after {retry_after}s")
                time.sleep(retry_after)
                continue
            elif response.status_code == 403:
                logging.error("403 Forbidden: Check API token permissions")
                st.error("403 Forbidden: Check API token permissions")
                return None
            elif response.status_code == 404:
                logging.warning("404 Not Found: Resource may not exist")
                st.warning("404 Not Found: Resource may not exist")
                return response
            elif response.status_code >= 500:
                logging.warning(f"Server error {response.status_code}, retrying...")
                time.sleep(backoff_factor ** attempt)
                continue
            else:
                logging.error(f"Request failed with status {response.status_code}: {response.text}")
                st.error(f"Request failed with status {response.status_code}")
                return response
        except requests.RequestException as e:
            logging.error(f"Request exception: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
                continue
    logging.error("Max retries exceeded")
    return None

def get_paginated_results(url: str, headers: Dict, params: Optional[Dict] = None) -> List[Dict]:
    results = []
    while url:
        response = retry_request(requests.get, url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        if response and response.ok:
            data = response.json()
            results.extend(data if isinstance(data, list) else [data])
            link_header = response.headers.get('Link', '')
            next_url = None
            for link in link_header.split(','):
                if 'rel="next"' in link:
                    next_url = link[link.find('<')+1:link.find('>')]
                    break
            url = next_url
            time.sleep(RATE_LIMIT_DELAY)
        else:
            break
    logging.info(f"Fetched {len(results)} paginated results")
    return results

def get_access_token():
    global ACCESS_TOKEN, REDIRECT_URI, CONFIG
    if ACCESS_TOKEN:
        logging.info("Using existing access token")
        return ACCESS_TOKEN
    try:
        port = find_available_port()
        REDIRECT_URI = f"http://localhost:{port}/oauth/callback"
        AUTH_URL = f"https://launchpad.37signals.com/authorization/new?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
        
        class OAuthHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/oauth/callback'):
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    code = params.get('code', [None])[0]
                    if code:
                        token_response = retry_request(
                            requests.post,
                            "https://launchpad.37signals.com/authorization/token.json",
                            data={
                                "type": "web_server",
                                "client_id": CLIENT_ID,
                                "client_secret": CLIENT_SECRET,
                                "redirect_uri": REDIRECT_URI,
                                "code": code
                            },
                            timeout=REQUEST_TIMEOUT
                        )
                        if token_response and token_response.ok:
                            token_data = token_response.json()
                            global ACCESS_TOKEN
                            ACCESS_TOKEN = token_data.get("access_token")
                            CONFIG["ACCESS_TOKEN"] = ACCESS_TOKEN
                            save_config(CONFIG)
                            self.respond_with("Success! You can close this tab.")
                        else:
                            self.respond_with("Token exchange failed.")
                    else:
                        self.respond_with("No code found in callback URL")
            
            def respond_with(self, message):
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())
        
        st.info(f"Opening browser for authorization on port {port}...")
        logging.info(f"Starting OAuth flow on port {port}")
        webbrowser.open(AUTH_URL)
        with socketserver.TCPServer(("localhost", port), OAuthHandler) as httpd:
            httpd.handle_request()
        if ACCESS_TOKEN:
            logging.info("Access token obtained and saved successfully")
        else:
            logging.error("Failed to obtain access token during OAuth flow")
        return ACCESS_TOKEN
    except Exception as e:
        logging.error(f"Failed to obtain access token: {str(e)}")
        st.error(f"Failed to obtain access token: {str(e)}")
        return None

def get_account_info(access_token: str) -> Optional[int]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    response = retry_request(
        requests.get,
        "https://launchpad.37signals.com/authorization.json",
        headers=headers,
        timeout=REQUEST_TIMEOUT
    )
    if response and response.ok:
        data = response.json()
        if "accounts" in data and data["accounts"]:
            logging.info("Account ID fetched successfully")
            return data["accounts"][0]["id"]
        else:
            logging.error("No accounts found in authorization response")
            st.error("No accounts found. Verify your API token has correct permissions.")
    else:
        logging.error(f"Failed to fetch account ID: Status {response.status_code if response else 'No response'}")
        st.error(f"Failed to fetch account ID")
    return None

def get_projects(account_id: int, access_token: str) -> List[tuple]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/projects.json"
    projects = get_paginated_results(url, headers)
    project_list = []
    for project in projects:
        todoset_id = None
        for dock in project.get("dock", []):
            if dock["name"] == "todoset" and dock["enabled"]:
                todoset_id = dock["id"]
        project_list.append((project['name'], project['id'], todoset_id))
    logging.info(f"Fetched {len(project_list)} projects")
    return project_list

def get_todoset(account_id: int, project_id: int, todoset_id: int, access_token: str) -> List[tuple]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todosets/{todoset_id}.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response and response.ok:
        todoset = response.json()
        todolists_url = todoset.get("todolists_url", "")
        todolists = get_paginated_results(todolists_url, headers)
        if todolists:
            return [(todolist['title'], todolist['id']) for todolist in todolists]
        else:
            fallback_url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todolists.json"
            todolists = get_paginated_results(fallback_url, headers)
            return [(todolist['title'], todolist['id']) for todolist in todolists]
    logging.warning(f"No todolists found for todoset {todoset_id}")
    return []

def get_tasks(account_id: int, project_id: int, todolist_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/todolists/{todolist_id}/todos.json"
    tasks = get_paginated_results(url, headers)
    task_list = []
    for task in tasks:
        task_id = task.get("id")
        task_response = retry_request(
            requests.get,
            f"{BASE_URL}/{account_id}/buckets/{project_id}/todos/{task_id}.json",
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        if not task_response or task_response.status_code != 200:
            continue
        task_info = {
            "title": task.get("title", "N/A"),
            "status": task.get("status", "N/A"),
            "due_on": task.get("due_on", "N/A"),
            "id": task_id,
            "assignee": task.get("assignee", {}).get("name", "Unassigned"),
            "assignee_id": task.get("assignee", {}).get("id", "N/A"),
            "creator": task.get("creator", {}).get("name", "Unknown"),
            "comments": []
        }
        task_list.append(task_info)
        time.sleep(RATE_LIMIT_DELAY)
    logging.info(f"Fetched {len(task_list)} tasks for todolist {todolist_id}")
    return task_list

def clean_comment_content(content: str) -> tuple[str, List[Dict]]:
    mentions = []
    cleaned_content = content
    attachment_pattern = r'<bc-attachment sgid="[^"]+" content-type="application/vnd.basecamp.mention">.*?<figcaption>\s*([^<]+)\s*</figcaption>\s*</figure></bc-attachment>'
    for match in re.finditer(attachment_pattern, content):
        full_tag = match.group(0)
        person_name = match.group(1).strip()
        sgid = re.search(r'sgid="([^"]+)"', full_tag).group(1) if re.search(r'sgid="([^"]+)"', full_tag) else "N/A"
        person_id = re.search(r'data-avatar-for-person-id="(\d+)"', full_tag).group(1) if re.search(r'data-avatar-for-person-id="(\d+)"', full_tag) else "N/A"
        mentions.append({
            "name": person_name,
            "person_id": person_id,
            "sgid": sgid
        })
    cleaned_content = re.sub(
        attachment_pattern,
        lambda m: f"@{m.group(1).strip()}",
        content
    )
    cleaned_content = re.sub(r'<[^>]+>', '', cleaned_content).strip()
    return cleaned_content, mentions

def get_task_comments(account_id: int, project_id: int, task_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{task_id}/comments.json"
    response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
    comment_list = []
    if response and response.status_code == 200:
        comments = response.json()
        for comment in comments:
            raw_content = comment.get("content", "N/A").strip()
            cleaned_content, mentions = clean_comment_content(raw_content)
            comment_list.append({
                "content": cleaned_content,
                "raw_content": raw_content,
                "mentions": mentions,
                "created_at": comment.get("created_at", "N/A"),
                "id": comment.get("id", "N/A"),
                "creator": comment.get("creator", {}).get("name", "N/A"),
                "creator_id": comment.get("creator", {}).get("id", "N/A")
            })
    logging.info(f"Fetched {len(comment_list)} comments for task {task_id}")
    return comment_list

def fetch_gemini_insights(task: Dict, comments: List[Dict]) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    You are a smart assistant analyzing a Basecamp task and its comments.
    Task: {task['title']}
    Status: {task['status']}
    Due Date: {task['due_on']}
    Assignee: {task['assignee']}
    Creator: {task['creator']}
    Comments: {json.dumps(comments, indent=2)}
    
    Provide insights and automation suggestions based on the task and comments. For example:
    - Summarize the task and comment context.
    - Suggest automation actions (e.g., reminders, status updates).
    - Identify potential issues or delays.
    - Offer recommendations for task completion.
    """
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No insights available")
            logging.info("Gemini insights generated successfully")
            return text
        else:
            logging.error(f"Gemini API error: {response.status_code} - {response.text}")
            return "Failed to fetch insights from Gemini API"
    except Exception as e:
        logging.error(f"Gemini API request failed: {str(e)}")
        return f"Error fetching insights: {str(e)}"

def analyze_company_culture(data: List[Dict]) -> Dict:
    try:
        all_comments = []
        for project in data:
            for todolist in project.get('todolists', []):
                for task in todolist.get('tasks', []):
                    all_comments.extend(task.get('comments', []))
        if not all_comments:
            logging.warning("No comments found in data, using default culture")
            return {
                "formality": "neutral",
                "avg_length": 100,
                "mention_freq": 0.5,
                "common_phrases": ["update", "team", "progress"]
            }
        total_length = 0
        mention_count = 0
        word_counts = {}
        for comment in all_comments:
            content = comment.get('content', '')
            total_length += len(content)
            mention_count += len(comment.get('mentions', []))
            words = content.lower().split()
            for word in words:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
        avg_length = total_length / len(all_comments) if all_comments else 100
        mention_freq = mention_count / len(all_comments) if all_comments else 0.5
        formality = "informal" if avg_length < 100 else "formal"
        if any(word in word_counts for word in ["please", "regards", "thank you", "formally"]):
            formality = "formal"
        common_phrases = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        common_phrases = [word for word, count in common_phrases]
        culture = {
            "formality": formality,
            "avg_length": avg_length,
            "mention_freq": mention_freq,
            "common_phrases": common_phrases
        }
        logging.info(f"Culture analysis: {culture}")
        return culture
    except Exception as e:
        logging.error(f"Culture analysis error: {str(e)}")
        st.error(f"Error analyzing culture: {str(e)}")
        return {
            "formality": "neutral",
            "avg_length": 100,
            "mention_freq": 0.5,
            "common_phrases": ["update", "team", "progress"]
        }

def generate_structured_text(user_input: str, culture: Dict, mentions: List[Dict]) -> str:
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    You are an AI assistant structuring a user's message for a Basecamp comment
    to align with the company culture inferred from existing comments.

    Company Culture:
    - Formality: {culture['formality']}
    - Average comment length: {culture['avg_length']:.0f} characters
    - Mention frequency: {culture['mention_freq']:.2f} mentions per comment
    - Common phrases: {', '.join(culture['common_phrases'])}

    Available Users for Mentions:
    {json.dumps([{'name': m['name'], 'sgid': m['sgid']} for m in mentions], indent=2)}

    User Input:
    {user_input}

    Instructions:
    - Rephrase the user's input to match the company culture (e.g., formality, length).
    - Include mentions (e.g., @username) if the mention frequency is > 0.5, using the provided user names.
    - Keep the tone professional if formal, or conversational if informal.
    - Incorporate common phrases if appropriate.
    - Return only the structured text, ready to be posted as a Basecamp comment.
    """
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        if response.ok:
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Failed to generate structured text")
            logging.info("Structured text generated successfully")
            return text
        else:
            logging.error(f"Gemini API error: {response.status_code} - {response.text}")
            return f"Failed to generate structured text: {response.text}"
    except Exception as e:
        logging.error(f"Gemini API request failed: {str(e)}")
        return f"Error generating structured text: {str(e)}"

def post_comment(account_id: int, project_id: int, task_id: int, content: str, access_token: str) -> bool:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "BasecampAPIClient (mulukenashenafi84@outlook.com)",
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{task_id}/comments.json"
    data = {"content": content}
    try:
        response = retry_request(
            requests.post,
            url,
            headers=headers,
            json=data,
            timeout=REQUEST_TIMEOUT
        )
        if response and response.status_code == 201:
            logging.info(f"Comment posted successfully to task {task_id}")
            return True
        else:
            logging.error(f"Failed to post comment: Status {response.status_code if response else 'No response'}")
            st.error(f"Failed to post comment: {response.text if response else 'Network error'}")
            return False
    except Exception as e:
        logging.error(f"Post comment error: {str(e)}")
        st.error(f"Error posting comment: {str(e)}")
        return False

def save_data(data: List[Dict]):
    try:
        output_file = f"basecamp_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        with open("basecamp_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data saved to {output_file} and basecamp_data.json")
        return output_file
    except Exception as e:
        logging.error(f"Data save error: {str(e)}")
        st.error(f"Error saving data: {str(e)}")
        return None

def load_data() -> List[Dict]:
    try:
        with open("basecamp_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info("Loaded basecamp_data.json successfully")
        return data
    except FileNotFoundError:
        logging.warning("basecamp_data.json not found")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in basecamp_data.json: {str(e)}")
        st.error(f"Invalid basecamp_data.json: {str(e)}")
        return []

# Streamlit UI
st.set_page_config(page_title="Basecamp AI Assistant", layout="wide")

def initialize_session_state():
    try:
        defaults = {
            'data': load_data(),
            'access_token': CONFIG.get("ACCESS_TOKEN"),
            'account_id': None,
            'structured_comment': None
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        logging.info("Session state initialized")
    except Exception as e:
        logging.error(f"Session state initialization error: {str(e)}")
        st.error(f"Error initializing app: {str(e)}")

def main():
    initialize_session_state()

    st.markdown("""
    <style>
    .navbar {
        background-color: #1a73e8;
        padding: 10px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    </style>
    <div class="navbar">Basecamp AI Assistant</div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Dashboard", "Tasks", "Comment Generator", "Settings"])

    if page == "Dashboard":
        st.header("Dashboard")
        st.write("Welcome to the Basecamp AI Assistant. Select a project and task from the Tasks page or generate a comment in the Comment Generator.")
        if st.session_state.data:
            st.subheader("Data Summary")
            project_count = len(st.session_state.data)
            task_count = sum(len(todolist['tasks']) for project in st.session_state.data for todolist in project.get('todolists', []))
            st.write(f"Projects: {project_count}")
            st.write(f"Tasks: {task_count}")
        else:
            st.warning("No data loaded. Fetch data from the Settings page.")

    elif page == "Tasks":
        st.header("Tasks")
        if not st.session_state.data:
            st.warning("No data available. Please fetch data from the Settings page.")
            st.markdown("[Go to Settings to fetch data](#)")
            return

        project_names = [project['project_name'] for project in st.session_state.data]
        selected_project = st.sidebar.selectbox("Select Project", project_names)
        project = next((p for p in st.session_state.data if p['project_name'] == selected_project), None)
        
        if project:
            todolist_names = [todolist['todolist_name'] for todolist in project.get('todolists', [])]
            selected_todolist = st.sidebar.selectbox("Select To-do List", todolist_names)
            todolist = next((t for t in project['todolists'] if t['todolist_name'] == selected_todolist), None)
            
            if todolist:
                tasks = todolist.get('tasks', [])
                if not tasks:
                    st.info("No tasks found in this to-do list.")
                    return
                task_titles = [task['title'] for task in tasks]
                selected_task = st.sidebar.selectbox("Select Task", task_titles)
                task = next((t for t in tasks if t['title'] == selected_task), None)
                
                if task:
                    st.subheader(f"Task: {task['title']}")
                    st.write(f"**Status**: {task['status']}")
                    st.write(f"**Due Date**: {task['due_on']}")
                    assignee = task.get('assignee', 'Unassigned')
                    st.write(f"**Assignee**: {assignee}")
                    creator = task.get('creator', 'Unknown')
                    st.write(f"**Creator**: {creator}")
                    
                    logging.debug(f"Session state - access_token: {'present' if st.session_state.access_token else 'missing'}, account_id: {'present' if st.session_state.account_id else 'missing'}")
                    
                    if st.button("Fetch Comments"):
                        if st.session_state.access_token and st.session_state.account_id:
                            with st.spinner("Fetching comments..."):
                                comments = get_task_comments(
                                    st.session_state.account_id,
                                    project['project_id'],
                                    task['id'],
                                    st.session_state.access_token
                                )
                                task['comments'] = comments
                                save_data(st.session_state.data)
                                st.success("Comments fetched and saved.")
                        else:
                            st.error("Access token or account ID missing. Please fetch data from the Settings page.")
                            st.markdown("[Go to Settings](#)")
                            logging.warning("Attempted to fetch comments without access_token or account_id")
                    
                    if task['comments']:
                        st.subheader("Comments")
                        for comment in task['comments']:
                            st.markdown(f"**Comment ID: {comment['id']}**")
                            st.write(f"**Creator**: {comment['creator']}")
                            st.write(f"**Created**: {comment['created_at']}")
                            st.write(f"**Content**: {comment['content']}")
                            if comment['mentions']:
                                st.write(f"**Mentions**: {', '.join(f'@{m['name']}' for m in comment['mentions'])}")
                            st.markdown("---")
                    else:
                        st.info("No comments available. Click 'Fetch Comments' to retrieve them.")
                    
                    st.subheader("AI Insights")
                    if st.button("Generate Insights"):
                        with st.spinner("Generating insights..."):
                            insights = fetch_gemini_insights(task, task['comments'])
                            st.markdown(insights)

    elif page == "Comment Generator":
        st.header("Generate Structured Comment")
        st.write("Enter your comment or announcement, and the assistant will structure it to match the company culture.")

        # Check for cached data
        if not st.session_state.data:
            st.warning("No cached Basecamp data found. Using default culture. Go to Settings to fetch data.")
            culture = {
                "formality": "neutral",
                "avg_length": 100,
                "mention_freq": 0.5,
                "common_phrases": ["update", "team", "progress"]
            }
            mentions = []
        else:
            try:
                culture = analyze_company_culture(st.session_state.data)
                # Collect unique mentions from data
                all_mentions = []
                for project in st.session_state.data:
                    for todolist in project.get('todolists', []):
                        for task in todolist.get('tasks', []):
                            for comment in task.get('comments', []):
                                all_mentions.extend(comment.get('mentions', []))
                mentions = list({m['person_id']: m for m in all_mentions}.values())
                st.info(f"Using company culture: {culture['formality']} (mention frequency: {culture['mention_freq']:.2f})")
            except Exception as e:
                logging.error(f"Error processing cached data: {str(e)}")
                st.error(f"Error processing cached data: {str(e)}")
                culture = {
                    "formality": "neutral",
                    "avg_length": 100,
                    "mention_freq": 0.5,
                    "common_phrases": ["update", "team", "progress"]
                }
                mentions = []

        # User input
        user_input = st.text_area("Enter your comment or announcement:", height=100)
        if st.button("Generate Structured Comment"):
            if user_input:
                with st.spinner("Generating structured comment..."):
                    try:
                        structured_text = generate_structured_text(user_input, culture, mentions)
                        st.session_state.structured_comment = structured_text
                        st.markdown("**Structured Comment:**")
                        st.write(structured_text)
                        st.code(structured_text, language="text")
                    except Exception as e:
                        logging.error(f"Error generating comment: {str(e)}")
                        st.error(f"Error generating comment: {str(e)}")
            else:
                st.warning("Please enter a comment or announcement.")

        if 'structured_comment' in st.session_state and st.session_state.structured_comment:
            st.markdown("**Preview of Structured Comment:**")
            st.write(st.session_state.structured_comment)
            if st.button("Copy to Clipboard"):
                st.write("Copy the text below:")
                st.code(st.session_state.structured_comment, language="text")
                st.success("Text displayed for manual copying. Use Ctrl+C to copy.")

            st.subheader("Post to Basecamp (Optional)")
            st.write("Provide Basecamp credentials to post the comment directly.")
            account_id = st.text_input("Basecamp Account ID:", placeholder="e.g., 1234567")
            project_id = st.text_input("Project ID:", placeholder="e.g., 8901234")
            task_id = st.text_input("Task ID:", placeholder="e.g., 5678901")
            access_token = st.text_input("Access Token:", type="password", value=st.session_state.access_token or "")

            if st.button("Post Comment to Basecamp"):
                if st.session_state.structured_comment and account_id and project_id and task_id and access_token:
                    try:
                        account_id = int(account_id)
                        project_id = int(project_id)
                        task_id = int(task_id)
                        with st.spinner("Posting comment to Basecamp..."):
                            success = post_comment(
                                account_id,
                                project_id,
                                task_id,
                                st.session_state.structured_comment,
                                access_token
                            )
                            if success:
                                st.success("Comment posted successfully!")
                                del st.session_state.structured_comment
                            else:
                                st.error("Failed to post comment. Check credentials and logs.")
                    except ValueError:
                        logging.error("Invalid numeric input for Basecamp IDs")
                        st.error("Account ID, Project ID, and Task ID must be numbers.")
                    except Exception as e:
                        logging.error(f"Error posting comment: {str(e)}")
                        st.error(f"Error posting comment: {str(e)}")
                else:
                    logging.warning("Attempted to post comment without complete credentials")
                    st.error("Please provide all Basecamp credentials and ensure a comment is generated.")

    elif page == "Settings":
        st.header("Settings")
        st.subheader("Basecamp Data")
        if st.button("Fetch New Data"):
            with st.spinner("Fetching data from Basecamp..."):
                try:
                    access_token = get_access_token()
                    if access_token:
                        st.session_state.access_token = access_token
                        account_id = get_account_info(access_token)
                        if account_id:
                            st.session_state.account_id = account_id
                            projects = get_projects(account_id, access_token)
                            all_data = []
                            for project_name, project_id, todoset_id in tqdm(projects, desc="Processing projects"):
                                project_data = {
                                    "project_name": project_name,
                                    "project_id": project_id,
                                    "todolists": []
                                }
                                if todoset_id:
                                    todolists = get_todoset(account_id, project_id, todoset_id, access_token)
                                    for todolist_name, todolist_id in todolists:
                                        tasks = get_tasks(account_id, project_id, todolist_id, access_token)
                                        project_data["todolists"].append({
                                            "todolist_name": todolist_name,
                                            "todolist_id": todolist_id,
                                            "tasks": tasks
                                        })
                                all_data.append(project_data)
                            st.session_state.data = all_data
                            output_file = save_data(all_data)
                            st.success(f"Data fetched and saved to {output_file}")
                        else:
                            st.error("Failed to fetch account ID")
                            logging.error("Failed to fetch account ID")
                    else:
                        st.error("Failed to obtain access token. Check logs for details.")
                        logging.error("Failed to obtain access token")
                except Exception as e:
                    logging.error(f"Fetch data error: {str(e)}")
                    st.error(f"Error fetching data: {str(e)}")

        st.subheader("Troubleshooting")
        st.write("If the app doesn't load or behaves unexpectedly, check the following:")
        st.markdown("""
        - Ensure all dependencies are installed: `pip install streamlit requests tqdm nltk`
        - Verify the Gemini API key in the code is valid.
        - Check `basecamp_gemini_app.log` for errors.
        - Try a different browser or incognito mode.
        - Run on a different port: `streamlit run basecamp_gemini_app.py --server.port 8503`
        """)

if __name__ == "__main__":
    try:
        logging.info("Starting Streamlit app")
        main()
    except Exception as e:
        logging.error(f"Main app error: {str(e)}")
        st.error(f"Critical error: {str(e)}. Check basecamp_gemini_app.log for details.")