# Upgraded: GPT-4 Turbo + Multi-Factor Scoring Simulation (Jobscan style)
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_openai_feedback(resume, jd):
    prompt = '''
You are a highly advanced resume screening AI modeled on Jobscan.io.

Compare the following resume and job description and score it using these exact weights:
- 40% Hard Skills Match (tech stacks, APIs, tools)
- 20% Experience Alignment (relevant past roles, project relevance, years of exp)
- 15% Education Relevance (degrees, coursework)
- 10% Soft Skills and Collaboration (communication, leadership, team work)
- 10% Formatting & Structure (length, ATS-friendly layout, clarity)
- 5% Relevance to Work Location & Authorization (remote/hybrid match)

Return a JSON object with:
{
  "score": Final ATS score as int 0–100
  "strengths": List of clear, job-relevant strengths
  "weaknesses": Gaps or mismatches in resume
  "suggestions": Specific changes to increase score
  "matchedSkills": List of important matched skills from JD
  "missingSkills": List of key skills missing or weakly represented
}

Resume:
{resume}

Job Description:
{jd}
'''.replace("{resume}", resume).replace("{jd}", jd)

    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1800,
            stop=["}"]
        )
        text = completion.choices[0].message.content + "}"
        print("OpenAI raw:", text[:300])
        json_block = re.search(r'\{.*\}', text, re.DOTALL)
        parsed = json.loads(json_block.group()) if json_block else {}
        parsed.pop("realismFlags", None)
        return parsed
    except Exception as e:
        print("OpenAI error:", e)
        return {
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "matchedSkills": [],
            "missingSkills": []
        }


@app.route("/score-resume", methods=["POST"])
def score():
    data = request.json
    resume = data.get("resume", "")
    jd = data.get("jobdesc", "")

    if not resume or not jd:
        return jsonify({"error": "Missing input"}), 400

    try:
        feedback = get_openai_feedback(resume, jd)
        return jsonify(feedback)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5055)))
