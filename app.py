from flask import Flask, render_template, request
from models.GPT2TextGenerator import GPT2TextGenerator
import asyncio

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def home():
    prompt = ""
    model = GPT2TextGenerator(300, 0.7, 50)
    if request.method == "POST":
        text = request.form["prompt"]
        if text:
            prompt = asyncio.run(generate_text_async(text))

    return render_template("index.html", prompt=prompt)


# Invalid URL
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


# Internal Server Error
@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500


async def generate_text_async(prompt):
    model = GPT2TextGenerator(300, 0.7, 50)
    return await model.generate_text(prompt)


if __name__ == '__main__':
    app.run()
