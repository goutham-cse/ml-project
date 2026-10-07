import json
import urllib.request
import urllib.error

def analyze_github_profile(username):
    """
    Fetches real GitHub data via GitHub REST API if a valid username is provided.
    Falls back gracefully to default estimates if invalid or unprovided.
    """
    if not username or username.strip() == "":
        return {
            'repos': 4,
            'stars': 5,
            'commits_year': 120,
            'github_score': 35.0,
            'languages': ['Python', 'HTML/CSS'],
            'profile_found': False,
            'avatar_url': None
        }

    username = username.strip()
    try:
        # 1. Fetch User Data
        user_url = f"https://api.github.com/users/{username}"
        req = urllib.request.Request(user_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            user_data = json.loads(response.read().decode())

        # 2. Fetch Public Repos
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
        req_repos = urllib.request.Request(repos_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_repos, timeout=5) as response:
            repos_data = json.loads(response.read().decode())

        public_repos = user_data.get('public_repos', len(repos_data))
        followers = user_data.get('followers', 0)
        avatar_url = user_data.get('avatar_url')

        total_stars = 0
        languages = set()
        for repo in repos_data:
            total_stars += repo.get('stargazers_count', 0)
            lang = repo.get('language')
            if lang:
                languages.add(lang)

        # Estimate commit volume based on public repos and update activity
        commits_est = min(1200, public_repos * 22 + followers * 5 + len(languages) * 15)

        # Calculate composite GitHub score (0-100)
        github_score = round(
            min(100.0, max(5.0,
                (public_repos / 25 * 35) +
                (commits_est / 600 * 45) +
                (total_stars / 30 * 20)
            )), 1
        )

        return {
            'repos': public_repos,
            'stars': total_stars,
            'commits_year': commits_est,
            'github_score': github_score,
            'languages': list(languages)[:5] if languages else ['Python'],
            'profile_found': True,
            'avatar_url': avatar_url,
            'username': username
        }

    except Exception as e:
        print(f"GitHub API notice for '{username}': {e}. Using estimated metrics.")
        return {
            'repos': 6,
            'stars': 8,
            'commits_year': 180,
            'github_score': 45.0,
            'languages': ['Python', 'JavaScript'],
            'profile_found': False,
            'avatar_url': None,
            'username': username
        }


def analyze_linkedin_profile(connections, certs, posts_freq):
    """
    Calculates LinkedIn Profile Score based on user inputs.
    """
    conn = min(500, max(0, int(connections or 0)))
    cert_count = min(10, max(0, int(certs or 0)))
    freq = min(10, max(1, int(posts_freq or 1)))

    linkedin_score = round(
        min(100.0, max(5.0,
            (conn / 500 * 40) +
            (cert_count / 6 * 40) +
            (freq / 10 * 20)
        )), 1
    )

    return {
        'connections': conn,
        'certs': cert_count,
        'posts_freq': freq,
        'linkedin_score': linkedin_score
    }
