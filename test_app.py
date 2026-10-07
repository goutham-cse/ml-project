import app as flask_app
import unittest
import json
import sqlite3

class PlacementAppTestCase(unittest.TestCase):
    def setUp(self):
        flask_app.app.config['TESTING'] = True
        flask_app.app.config['WTF_CSRF_ENABLED'] = False
        self.app = flask_app.app.test_client()

    def test_full_user_flow(self):
        # 1. Register user
        res_reg = self.app.post('/register', data={
            'username': 'autotestuser',
            'email': 'autotest@university.edu',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(res_reg.status_code, 200)

        # 2. Login user
        res_login = self.app.post('/login', data={
            'username': 'autotestuser',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(res_login.status_code, 200)
        self.assertIn(b'College Placement Readiness Assessor', res_login.data)

        # 3. Predict Endpoint
        res_pred = self.app.post('/predict', data={
            'name': 'Aarav Sharma',
            'dept': 'Computer Science',
            'cgpa': '8.7',
            'leetcode_rating': '1650',
            'leetcode_solved': '320',
            'skills': 'Python, DSA, Web Dev, React, SQL',
            'github_handle': 'torvalds',
            'github_repos': '15',
            'github_commits_year': '450',
            'github_stars': '80',
            'linkedin_connections': '450',
            'linkedin_certs': '4',
            'linkedin_posts_freq': '7',
            'projects_count': '4',
            'project_complexity': 'Advanced'
        }, follow_redirects=True)
        self.assertEqual(res_pred.status_code, 200)
        self.assertIn(b'Readiness Score', res_pred.data)
        self.assertIn(b'Key Strengths', res_pred.data)

        # 4. Analytics page
        res_analytics = self.app.get('/analytics')
        self.assertEqual(res_analytics.status_code, 200)
        self.assertIn(b'94.2', res_analytics.data)

        # 5. File Exports
        res_export_csv = self.app.get('/export/placement_dataset.csv')
        self.assertEqual(res_export_csv.status_code, 200)

        res_export_xlsx = client_get = self.app.get('/export/placement_dataset_info.xlsx')
        self.assertEqual(res_export_xlsx.status_code, 200)

        res_export_md = self.app.get('/export/ml_model_report.md')
        self.assertEqual(res_export_md.status_code, 200)

    def test_github_api_endpoint(self):
        res = self.app.post('/api/fetch_github', data=json.dumps({'github_handle': 'octocat'}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('github_score', data)

if __name__ == '__main__':
    unittest.main()
