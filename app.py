from flask import Flask
import threading
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!"

def run_bot():
    import wingo_MLprediction_bot  # just import your bot file

if __name__ == "__main__":
    # Start bot in background thread
    threading.Thread(target=run_bot).start()

    # Start web server (required for Render free)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

