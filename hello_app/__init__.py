from flask import Flask  # Import the Flask class
app = Flask(__name__)    # Create an instance of the class for our use
app.secret_key = "super secret key" # super secret key
app.debug = True
app.env = "development"