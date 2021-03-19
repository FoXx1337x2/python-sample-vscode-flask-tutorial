from datetime import datetime
from flask import Flask, flash, request, redirect, render_template, send_from_directory
from . import app
import os
from werkzeug.utils import secure_filename
import pathlib

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = str(pathlib.Path().absolute()) +'\\uploaded_files'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    print(UPLOAD_FOLDER)
    return render_template("home.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/contact/")
def contact():
    return render_template("contact.html")

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")

@app.route("/get_file/<file>")
def get_file(file):
    """Download a file."""
    return send_from_directory(UPLOAD_FOLDER, file, as_attachment=True)


@app.route('/files')
def get_files():
    onlyfiles = os.listdir(UPLOAD_FOLDER)
    for file in onlyfiles:
        flash(file,'files')

    return redirect('/')


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'response')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading', 'response')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            flash('File successfully uploaded', 'response')
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif', 'response')
            return redirect(request.url)

        #krasne :)
        #ADIOJOIJDWIOWDQOIWEQIO
