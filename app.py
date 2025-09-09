import os
import re
import json
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()  # looks for .env file if present
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ No API key found. Please set GEMINI_API_KEY in environment or .env file.")

# 🔑 Configure Gemini (force JSON mode)
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={"response_mime_type": "application/json"}
)

# Flask app
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["REPORT_FOLDER"] = "reports"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)

# Store last uploaded image path
last_uploaded_path = None


# ------------------------------
# Leaf health analysis (OpenCV)
# ------------------------------
def analyze_leaf_health(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Invalid image"}

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Yellow mask
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Brown mask
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        total_pixels = img.shape[0] * img.shape[1]
        yellow_percent = round((np.sum(mask_yellow > 0) / total_pixels) * 100, 2)
        brown_percent = round((np.sum(mask_brown > 0) / total_pixels) * 100, 2)

        # Solidity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = round((area / hull_area) * 100, 2)

        return {"yellow_spots": yellow_percent, "brown_spots": brown_percent, "solidity": solidity}

    except Exception as e:
        return {"error": str(e)}


# ------------------------------
# JSON parser (robust fallback)
# ------------------------------
def parse_json_response(raw_text):
    raw_text = raw_text.strip()

    # Remove Markdown fences if present
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```json", "", raw_text, flags=re.IGNORECASE).strip()
        raw_text = raw_text.strip("`").strip()

    # Extract first {...} block if extra text is around
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        raw_text = match.group(0)

    return json.loads(raw_text)


# ------------------------------
# Gemini analysis
# ------------------------------
def get_gemini_analysis(image_path, yellow, brown, solidity, override_plant=None):
    with open(image_path, "rb") as f:
        image_data = f.read()

    if override_plant:
        plant_info = f"The user confirms this is a {override_plant} leaf."
    else:
        plant_info = "Identify the plant type from the image."

    prompt = f"""
    You are a plant pathology expert.
    {plant_info}
    Leaf stats:
    - Yellow spots: {yellow}%
    - Brown spots: {brown}%
    - Solidity: {solidity}%

    Return a JSON object with:
    - plant (string)
    - suggestion (string, max 3 lines)
    - thresholds (object with yellow_max, brown_max, solidity_min)
    """

    try:
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_data}]
        )
        raw = response.text.strip()
        print("🔎 Gemini raw response:", raw)
        return parse_json_response(raw)

    except Exception as e:
        print("⚠️ Gemini failed:", e)
        return {
            "plant": override_plant if override_plant else "Unknown Plant",
            "suggestion": "Could not analyze properly.",
            "thresholds": {"yellow_max": 10, "brown_max": 5, "solidity_min": 85}
        }


# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        global last_uploaded_path
        last_uploaded_path = filepath

        result = analyze_leaf_health(filepath)
        if "error" in result:
            return jsonify(result), 400

        ai_result = get_gemini_analysis(
            filepath,
            result["yellow_spots"],
            result["brown_spots"],
            result["solidity"]
        )

        result.update(ai_result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/correct-plant", methods=["POST"])
def correct_plant():
    try:
        data = request.json
        plant = data.get("plant", "Unknown Plant")
        yellow = data.get("yellow_spots")
        brown = data.get("brown_spots")
        solidity = data.get("solidity")

        global last_uploaded_path
        if not last_uploaded_path:
            return jsonify({"error": "No uploaded image found"}), 400

        ai_result = get_gemini_analysis(
            last_uploaded_path,
            yellow, brown, solidity,
            override_plant=plant
        )

        return jsonify(ai_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-report", methods=["POST"])
def download_report():
    try:
        data = request.json
        filename = "report.pdf"
        filepath = os.path.join(app.config["REPORT_FOLDER"], filename)

        c = canvas.Canvas(filepath, pagesize=A4)
        c.drawString(100, 800, "🌱 Leaf Analysis Report")
        c.drawString(100, 770, f"Plant: {data['plant']}")
        c.drawString(100, 750, f"Yellow spots: {data['yellow_spots']}%")
        c.drawString(100, 730, f"Brown spots: {data['brown_spots']}%")
        c.drawString(100, 710, f"Solidity: {data['solidity']}%")
        c.drawString(100, 690, f"Suggestion: {data['suggestion']}")
        c.save()

        return jsonify({"file": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-report/<filename>")
def get_report(filename):
    return send_from_directory(app.config["REPORT_FOLDER"], filename)


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
