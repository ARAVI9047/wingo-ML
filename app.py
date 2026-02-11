from flask import Flask
import threading
from wingo last 10 prediction-bot import start_bot  # if your bot has a function

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is running!"

def run_bot():
    start_bot()

threading.Thread(target=run_bot).start()

if __name__ == "__main__":
   port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    
