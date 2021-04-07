from datetime import datetime, date

import numpy as np
from flask import Flask, flash, request, redirect, render_template, send_from_directory
from . import app
import os
from werkzeug.utils import secure_filename


from turtle import pd, distance

# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
# UPLOAD_FOLDER = str(pathlib.Path().absolute()) +'\\uploaded_files'
# from .analysis.log_file_analyzer import analyze_laps, put_laps_to_json, segment, intersection
from .analysis.log_file_analyzer import *
from .analysis.lap_difference_analyzer import *

UPLOAD_FOLDER = './UPLOADS'
# logger = logging.getLogger('werkzeug') # grabs underlying WSGI logger
# handler = logging.FileHandler('test.log') # creates handler for the log file
# logger.addHandler(handler) # adds handler to the werkzeug WSGI logger

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

######################CarComp##########################
def log_to_dataFrame(file_path):
    """
        Converts a log file of a ride to a Pandas dataframe.
        Parameters
        --------
            file_path : str
                A path to a log file.
        Example of a log file
        --------
            2020-06-29 13:06:24,595 - INFO - ;LAT;480492306;LON;175678507;UTMX;69136106;UTMY;532496222;HMSL;126112;GSPEED;0;CRS;0;HACC;66720;NXPT;1139
            2020-06-29 13:06:24,648 - INFO - ;LAT;480492313;LON;175678494;UTMX;69136096;UTMY;532496230;HMSL;126121;GSPEED;4;CRS;0;HACC;52510;NXPT;1139
            2020-06-29 13:06:24,698 - INFO - ;LAT;480492305;LON;175678495;UTMX;69136097;UTMY;532496221;HMSL;126146;GSPEED;1;CRS;0;HACC;49421;NXPT;1140
        Returns
        --------
        A dataframe with all the logs.
    """

    logs = pd.read_csv(file_path, header=None, sep=';', names=['TIME', '1', 'LAT', '3', 'LON', '5', 'UTMX', '7', 'UTMY',
                                                               '9', 'HMSL', '11', 'GSPEED', '13', 'CRS', '15', 'HACC',
                                                               '17', 'NXPT'])

    logs = logs.drop(columns=['1', '3', '5', '7', '9', '11', '13', '15', '17'])
    logs = logs.dropna()
    return logs


def normalize_logs(logs):
    """
        Normalizes data of the logs dataframe.
        In particular, the 'LAT' and 'LON' columns is divided by 10 000 000.
        The 'GSPEED' column is divided by 100.
        The CRS column is divided by 100 000.
        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """
    logs['TIME'] = logs['TIME'].apply(lambda x: x.split(' ')[1])
    logs['TIME'] = pd.to_datetime(logs['TIME'], format='%H:%M:%S,%f').dt.time
    logs['TIME'] = logs['TIME'].apply(lambda x: datetime.combine(date.today(), x) - datetime.combine(date.today(), logs['TIME'][0]))
    logs['TIME'] = logs['TIME'].apply(lambda x: x.total_seconds())

    logs['LAT'] = logs['LAT'].apply(lambda x: x * 0.0000001)
    logs['LON'] = logs['LON'].apply(lambda x: x * 0.0000001)
    logs['GSPEED'] = logs['GSPEED'].apply(lambda x: x * 0.01)
    logs['CRS'] = logs['CRS'].apply(lambda x: x * 0.00001)


def separate_laps(traces, ref_lap=None):
    """
        Separate all the log dataframe into several laps.
        Parameters
        --------
            traces : DataFrame
                A dataframe with logs of a ride.
            ref_lap : DataFrame
                A dataframe with logs of a reference ride.
                It is used to define finish line.
                It is and optional parameter. Default value is None.
    """

    ref_lap = traces if ref_lap is None else ref_lap
    points = traces[['LON', 'LAT']].values.tolist()

    # use last points to determine normal vector
    last_point1 = [ref_lap['LON'].iloc[-1], ref_lap['LAT'].iloc[-1]]
    last_point2 = [ref_lap['LON'].iloc[-2], ref_lap['LAT'].iloc[-2]]

    a = last_point2[0] - last_point1[0]
    b = last_point2[1] - last_point1[1]

    dst = distance.euclidean(last_point1, last_point2)
    distance_multiplier = math.ceil(0.0001 / (2 * dst))

    v_normal = np.array([-b, a])
    start_point = np.array(last_point1)

    point_top = start_point + distance_multiplier * v_normal
    point_bottom = start_point - distance_multiplier * v_normal
    start_segment = segment(point_top, point_bottom)

    laps = [0]
    for i in range(len(points) - 1):
        if points[i] == points[i + 1]:
            continue

        # segment between point1 and point2
        seg = segment(points[i], points[i + 1])
        has_intersection = intersection(seg, start_segment)

        # add start of a new lap
        if has_intersection:
            intersection(seg, start_segment)
            laps.append(i + 1)
            print('Lap ending at index: {}'.format(i))
            print(seg, start_segment)

    return laps

def get_calc(reference,traces):
    reference_file_path = str(os.path.join(UPLOAD_FOLDER) + "/" + reference)
    traces_file_path = str(os.path.join(UPLOAD_FOLDER) + "/" + traces)

    reference_df = log_to_dataFrame(reference_file_path)
    normalize_logs(reference_df)

    traces_df = log_to_dataFrame(traces_file_path)
    normalize_logs(traces_df)

    laps = separate_laps(traces_df, reference_df)
    analyzed_laps = analyze_laps(traces_df, reference_df, laps)

    json = put_laps_to_json(analyzed_laps)
    json = json.encode()
    # print(json)
    return json
######################CarComp##########################


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
    absolute_path = os.path.abspath("./UPLOADS")
    return send_from_directory(absolute_path, file, as_attachment=True)


@app.route('/files')
def get_files():
    try:
        onlyfiles = os.listdir(UPLOAD_FOLDER)
        for file in onlyfiles:
            flash(file,'files')

        return redirect('/')
    except Exception as ue:
        # logger.error("Unexpected Error: malformed JSON in POST request, check key/value pair at: ")
        # logger.error(ue)
        return redirect('/')


@app.route('/uploader', methods=['POST', 'GET'])
def upload_file():
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'files[]' not in request.files:
                flash('No file part', 'response')
                return redirect(request.url)
            files = request.files.getlist('files[]')
            filenames = []
            for file in files:
                if file:
                    filename = secure_filename(file.filename)
                    filenames.append(filename)
                    file.save(os.path.join(UPLOAD_FOLDER, filename))
                    flash('File successfully uploaded', 'response')
                    # return redirect('/')
                else:
                    flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif', 'response')
                    return redirect(request.url)

            return get_calc(filenames[0], filenames[1])
    except:
        return redirect('/')


        #krasne :)
        #ADIOJOIJDWIOWDQOIWEQIO
