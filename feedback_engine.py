def generate_feedback(cgpa, leetcode_rating, leetcode_solved, github_score, github_repos, 
                      linkedin_score, linkedin_certs, projects_count, project_complexity, 
                      skills_count, dept, readiness_score):
    
    strengths = []
    weaknesses = []
    recommendations = []

    # 1. CGPA Analysis
    if cgpa >= 8.5:
        strengths.append(f"Outstanding Academic CGPA of {cgpa:.2f}/10.0, placing you in the top tier for campus eligibility filters.")
    elif cgpa >= 7.5:
        strengths.append(f"Solid Academic CGPA ({cgpa:.2f}/10.0), meeting requirements for almost all hiring drives.")
    else:
        weaknesses.append(f"CGPA of {cgpa:.2f}/10.0 is below 7.5, which may filter you out of tier-1 dream company shortlists.")
        recommendations.append("Target maintaining CGPA above 7.5+ in upcoming semesters to clear initial corporate cutoffs.")

    # 2. LeetCode / Coding DSA Analysis
    if leetcode_rating >= 1800 or leetcode_solved >= 400:
        strengths.append(f"High Problem-Solving capability with LeetCode rating {leetcode_rating} ({leetcode_solved} problems solved). Strong DSA readiness for technical interviews.")
    elif leetcode_rating >= 1400 or leetcode_solved >= 180:
        strengths.append(f"Decent Data Structures & Algorithms foundation with {leetcode_solved} LeetCode problems solved.")
        weaknesses.append(f"LeetCode rating ({leetcode_rating}) needs boost to consistently clear tier-1 product company coding rounds.")
        recommendations.append("Practice Medium/Hard LeetCode problems, focusing on Dynamic Programming, Graphs, and Trees.")
    else:
        weaknesses.append(f"Low coding activity (LeetCode rating: {leetcode_rating}, Solved: {leetcode_solved}). Critical bottleneck for technical rounds.")
        recommendations.append("Commit to solving at least 2-3 DSA problems daily on LeetCode/CodeChef to reach 200+ solved milestone.")

    # 3. GitHub & Open Source Activity
    if github_score >= 70 or github_repos >= 12:
        strengths.append(f"Active GitHub profile with {github_repos} public repos (GitHub score: {github_score}/100), demonstrating version control & open-source discipline.")
    elif github_score >= 40:
        strengths.append(f"Moderate open-source presence on GitHub ({github_repos} repositories).")
        weaknesses.append("GitHub activity could be enhanced with consistent daily commits and starred repositories.")
        recommendations.append("Publish real-world project codebases to GitHub with comprehensive README documentation and unit tests.")
    else:
        weaknesses.append(f"Low GitHub open-source activity (GitHub score: {github_score}/100). Lacks visible proof of practical software development.")
        recommendations.append("Start version controlling all your semester projects on GitHub and contribute to open-source repositories.")

    # 4. LinkedIn & Professional Presence
    if linkedin_score >= 65 or linkedin_certs >= 3:
        strengths.append(f"Strong LinkedIn professional presence (Score: {linkedin_score}/100) with {linkedin_certs} verified domain certifications.")
    else:
        weaknesses.append(f"Under-utilized LinkedIn network (Score: {linkedin_score}/100, Certifications: {linkedin_certs}).")
        recommendations.append("Connect with alumni, recruiters, and tech leads; showcase project accomplishments and pursue industry-recognized certifications.")

    # 5. Real-World Projects
    if projects_count >= 3 and project_complexity in ['Advanced', 'Production-Grade']:
        strengths.append(f"Exceptional project portfolio with {projects_count} projects including '{project_complexity}' complexity level applications.")
    elif projects_count >= 2:
        strengths.append(f"Good portfolio with {projects_count} practical projects.")
        if project_complexity in ['Basic', 'Intermediate']:
            weaknesses.append("Projects are mostly basic/intermediate; missing full-stack deployment or cloud hosting scale.")
            recommendations.append("Upgrade your top project to a production-grade full-stack architecture deployed on AWS/Vercel/Docker.")
    else:
        weaknesses.append("Insufficient real-world projects portfolio (fewer than 2 projects).")
        recommendations.append("Build at least two end-to-end full-stack or ML projects solving real-world domain problems.")

    # Overall Placement Readiness Summary
    if readiness_score >= 83.0:
        overall_summary = "You are in the **Dream Tier / Tier-1 Ready** category! You are well-positioned for top product companies, high packages, and competitive technical interviews."
    elif readiness_score >= 68.0:
        overall_summary = "You are in the **Product Tier Ready** category. With targeted refinement in DSA and system design, you can easily reach Dream Tier!"
    elif readiness_score >= 52.0:
        overall_summary = "You are currently **Service Tier Ready**. Strengthening core coding ratings and practical projects will elevate you to Product Tier."
    else:
        overall_summary = "You are currently **Not Placement Ready** (<52%). Immediate focused effort is required across DSA, projects, and academic consistency."

    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommendations': recommendations,
        'summary': overall_summary
    }
