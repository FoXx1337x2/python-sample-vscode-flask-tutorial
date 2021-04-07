import math
import os
from os import listdir
from os.path import isfile, join
# import magic
import urllib.request
from turtle import pd
import tk
import json
import numpy as np
import math

from . import app
from pandas import DataFrame
from scipy.spatial import distance
import json
from datetime import datetime, date
import pandas as pd
import math
from math import sqrt
from math import atan2
from numpy.linalg import norm, det
from numpy import cross, dot
from numpy import radians
from numpy import array, zeros
from numpy import cos, sin, arcsin
from similaritymeasures import curve_length_measure, frechet_dist
from obspy.geodetics import degrees2kilometers

import numpy as np

from flask import Flask, flash, request, redirect, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime, date
import json

UPLOAD_FOLDER = './UPLOADS'

######################CarComp##########################
firstx = 0
firsty = 0

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


def read_csv_ref_lap(file_path):
    """
        Creates a dataframe of a reference lap from a csv file.
        Parameters
        --------
            file_path : str
                A path to a csv file.
        Example of a log file
        --------
            LAT,LON,GSPEED,CRS,NLAT,NLON,NCRS
            48.049214299999996,17.5678361,1.08,219.10375000000002,48.0492134,17.567835199999998,215.70312
            48.0492134,17.567835199999998,1.03,215.70312,48.0492127,17.567834299999998,215.56731000000002
            48.0492127,17.567834299999998,1.11,215.56731000000002,48.049211899999996,17.567833399999998,216.61797
        Returns
        --------
        A dataframe with a reference lap.
    """

    logs = pd.read_csv(file_path)
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


def drop_unnecessary_columns(logs):
    """
        Drops the columns 'UTMX', 'UTMY', 'HMSL', 'HACC' and 'NXPT' of the logs dataframe.
        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """

    logs.drop(columns=['UTMX', 'UTMY', 'HMSL', 'HACC', 'NXPT'], inplace=True)


def drop_logs_where_car_stayed(logs: DataFrame):
    """
        Drops rows from the logs dataframe where the LAT and LON are not changing.
        Resets indices of a dataframe in the end.
        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """

    last_lat = None
    last_lon = None
    dropped_rows = list()

    for index, row in logs.iterrows():
        if row['LAT'] == last_lat and row['LON'] == last_lon:
            dropped_rows.append(index)
        else:
            last_lat = row['LAT']
            last_lon = row['LON']

    logs.drop(dropped_rows, inplace=True)
    logs.reset_index(drop=True, inplace=True)


def create_columns_with_future_position(logs):
    """
        Creates columns NLAT, NLON and NCRS which are the next position of a car.
        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """

    next_lat = logs['LAT']
    next_lat = next_lat.append(pd.Series([np.nan]), ignore_index=True)
    next_lat = next_lat.iloc[1:]
    next_lat = next_lat.reset_index(drop=True)

    next_lon = logs['LON']
    next_lon = next_lon.append(pd.Series([np.nan]), ignore_index=True)
    next_lon = next_lon.iloc[1:]
    next_lon = next_lon.reset_index(drop=True)

    next_crs = logs['CRS']
    next_crs = next_crs.append(pd.Series([np.nan]), ignore_index=True)
    next_crs = next_crs.iloc[1:]
    next_crs = next_crs.reset_index(drop=True)

    logs['NLAT'] = next_lat
    logs['NLON'] = next_lon
    logs['NCRS'] = next_crs

    logs = logs.dropna()  # Drop the last row which contains NaN values.


def segment(p1, p2):
    """
        Parameters
        ===========
        p1 : list
            The first point.
        p2 : list
            The second point.
        Returns
        ==========
            A line segment of points represented in a quadruple.
    """

    return (p1[0], p1[1], p2[0], p2[1])


def ccw(a, b, c):
    '''
        Determines whether three points are located in a counterclockwise way.
    '''

    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersection(s1, s2):
    a = (s1[0], s1[1])
    b = (s1[2], s1[3])
    c = (s2[0], s2[1])
    d = (s2[2], s2[3])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


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


def normalize_for_graph(logs):
    """
        Drops all columns except LAT, and LON
        Parameters
        --------
            logs : DataFrame
                A dataframe with logs of a ride.
    """
    logs.drop(columns=['UTMX', 'UTMY', 'HMSL', 'GSPEED', 'CRS', 'HACC', 'NXPT'], inplace=True)
    logs.rename(columns={"LAT": "y", "LON": "x"}, inplace=True)


def get_raw_data(file_path) -> DataFrame:
    log_df = log_to_dataFrame(file_path)
    normalize_logs(log_df)
    return log_df


def get_essential_data(file_path) -> DataFrame:
    log_df = log_to_dataFrame(file_path)
    normalize_logs(log_df)
    drop_unnecessary_columns(log_df)
    drop_logs_where_car_stayed(log_df)
    return log_df


def get_graph_data(file_path) -> DataFrame:
    log_df = log_to_dataFrame(file_path)
    normalize_logs(log_df)
    normalize_for_graph(log_df)
    # get_laps_json(log_df)
    return log_df


def get_lap_data(reference_file_path, traces_file_path):
    reference_df = log_to_dataFrame(reference_file_path)
    normalize_logs(reference_df)

    traces_df = log_to_dataFrame(traces_file_path)
    normalize_logs(traces_df)

    laps = separate_laps(traces_df, reference_df)
    analyzed_laps = analyze_laps(traces_df, reference_df, laps)
    return analyzed_laps


def get_raw_data_json(file_path) -> str:
    data = get_raw_data(file_path)
    return data.to_json(orient="records")


def get_essential_data_json(file_path) -> str:
    data = get_essential_data(file_path)
    return data.to_json(orient="records")


def get_track_graph_data(file_path) -> str:
    data = get_graph_data(file_path)
    data.x = data.x.apply(lambda deg: degrees2kilometers(deg) * 1000)
    data.y = data.y.apply(lambda deg: degrees2kilometers(deg) * 1000)
    global firsty
    global firstx
    firsty = data.x[0]
    firstx = data.y[0]
    data.x -= data.x[0]
    data.y -= data.y[0]
    return data.to_json(orient="records")


def get_reference_xy(data) -> str:
    data.drop(columns=['TIME', 'CRS', 'GSPEED'], inplace=True)
    return data.to_json(orient="records")


def get_reference_crs(data) -> str:
    data.drop(columns=['x', 'y', 'GSPEED'], inplace=True)
    data.rename(columns={"TIME": "x", "CRS": "y"}, inplace=True)
    return data.to_json(orient="records")


def get_data_xy(data) -> str:
    data.drop(columns=['TIME', 'CRS'], inplace=True)
    return data.to_json(orient="records")


def get_data_crs(data) -> str:
    data.drop(columns=['x', 'y'], inplace=True)
    data.rename(columns={"TIME": "x", "CRS": "y"}, inplace=True)
    return data.to_json(orient="records")


def average(lst):
    return sum(lst) / len(lst)


def analyze_laps(traces, reference_lap, laps):
    data_dict = {
        'lapNumber': [],
        'pointsPerLap': [],
        'curveLength': [],
        'averagePerpendicularDistance': [],
        'lapData': []
    }

    for i in range(len(laps) - 1):
        lap_data = traces.iloc[laps[i]: laps[i + 1]]

        drop_unnecessary_columns(lap_data)
        perpendicular_distance = find_out_difference_perpendiculars(lap_data, reference_lap)
        average_dist = round(perpendicular_distance / 100.0, 3)

        data_dict['lapNumber'].append(i)
        data_dict['pointsPerLap'].append(len(lap_data))
        data_dict['curveLength'].append(0)
        data_dict['averagePerpendicularDistance'].append(average_dist)
        lap_data.LAT = lap_data.LAT.apply(lambda deg: degrees2kilometers(deg) * 1000)
        lap_data.LON = lap_data.LON.apply(lambda deg: degrees2kilometers(deg) * 1000)
        lap_data.LAT -= firstx
        lap_data.LON -= firsty
        data_dict['lapData'].append(json.loads(lap_data.to_json(orient="records")))

    # tha last circuit (lap) was not saved yet so save that one
    lap_data = traces.iloc[laps[-1:]]

    drop_unnecessary_columns(lap_data)
    perpendicular_distance = find_out_difference_perpendiculars(lap_data, reference_lap)
    average_dist = round(perpendicular_distance / 100.0, 3)

    data_dict['lapNumber'].append(len(laps))
    data_dict['pointsPerLap'].append(len(lap_data))
    data_dict['curveLength'].append(0)
    data_dict['averagePerpendicularDistance'].append(average_dist)
    lap_data.LAT = lap_data.LAT.apply(lambda deg: degrees2kilometers(deg) * 1000)
    lap_data.LON = lap_data.LON.apply(lambda deg: degrees2kilometers(deg) * 1000)
    lap_data.LAT -= firstx
    lap_data.LON -= firsty
    data_dict['lapData'].append(json.loads(lap_data.to_json(orient="records")))

    data_frame = pd.DataFrame(data=data_dict)
    return data_frame


def save_laps_to_files(file_path, file_name, laps):
    laps.sort_values(by=['averagePerpendicularDistance'], inplace=True)
    laps.to_csv('{}/{}_lap-stats.csv'.format(file_path, file_name),
                index=False,
                header=['Lap number', 'Points per lap', 'Avg. perp. diff. (cm)'],
                columns=['lapNumber', 'pointsPerLap', 'averagePerpendicularDistance'])
    laps.to_csv('{}/{}_lap-data.csv'.format(file_path, file_name),
                index=False,
                header=['Lap number', 'Lap data'],
                columns=['lapNumber', 'lapData'])


def put_laps_to_json(laps):
    return laps.to_json(orient="records")


def get_number_of_lines(file_path):
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def create_curve(dataframe):
    curve = zeros((dataframe.shape[0], 2))
    curve[:, 0] = dataframe.LON
    curve[:, 1] = dataframe.LAT
    return curve


def earth_distance(point1, point2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(radians, [point1[1], point1[0], point2[1], point2[0]])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2

    c = 2 * arcsin(sqrt(a))
    km = 6367 * c
    return km


def distance_of_curve(lap):
    return sum(earth_distance(pt1, pt2)
               for pt1, pt2 in zip(lap, lap[1:]))


def find_out_difference(ref_lap, laps):
    """
        With the usage of several curve metrics, finds out differences between
        a referrence lap and laps of a ride

        Parameters
        --------
            ref_lap : DataFrame
                A dataframe of with logs of a a reference ride.
            laps : list
                A list of dataframes.
                Each dataframe represents one lap of a ride.

        Returns
        --------
            A dataframe object with three columns: a Measurements count, a Frechet distance and a Curve length measure.
    """

    ref_curve = create_curve(ref_lap)

    measurement_column = 'Measurements count'
    frechet_column = 'Frechet distance'
    curve_len_column = 'Curve length measure'
    data_structure = {measurement_column: [],
                      frechet_column: [],
                      curve_len_column: []}

    differences_df = pd.DataFrame(data=data_structure)

    for lap in laps:
        experimental_curve = create_curve(lap)

        m_count = len(lap)
        fd = frechet_dist(experimental_curve, ref_curve)
        cl = curve_length_measure(experimental_curve, ref_curve)

        difference = {measurement_column: m_count,
                      frechet_column: fd,
                      curve_len_column: cl, }

        differences_df = differences_df.append(difference, ignore_index=True)

    return differences_df


def line_length(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_point(point, lap, locality=None):
    OFFSET = 10
    minIndex = 0
    minLength = math.inf
    index_range = range(len(lap)) if locality == None \
        else range(locality - OFFSET, locality + OFFSET)

    for i in index_range:
        if i >= len(lap):
            i -= len(lap)
        elif i < 0:
            i += len(lap)

        lat = lap[i][0]
        lon = lap[i][1]

        length = line_length(lat, lon, point[0], point[1])
        if length < minLength:
            minIndex = i
            minLength = length

    return minIndex


def find_angle_between_vectors(v1, v2):
    return atan2(det([v1, v2]), dot(v1, v2))  # return angle in radians


def create_vector(point_A, point_B):
    return [point_B[0] - point_A[0], point_B[1] - point_A[1]]


# Perdendicular from p1 to line (p2,p3)
def shortest_distance(p1, p2, p3):
    dist = norm(cross(p2 - p3, p3 - p1)) / norm(p3 - p2)
    return dist


def find_shortest_distance(p1, p2, p3):
    x = array((p1[0], p1[1]))
    y = array((p2[0], p2[1]))
    z = array((p3[0], p3[1]))
    return shortest_distance(x, y, z)


def find_out_difference_perpendiculars(lap: pd.DataFrame, ref_lap: pd.DataFrame):
    """
        Calculates average perpendicular distance from every point of a lap to a ref_lap.

        Parameters
        --------
            lap : DataFrame
                A dataframe with a lap from which perpendiculars are calculated.
            ref_lap : DataFrame
                A dataframe with a lap to which perpenduculars are calculated.

        Returns
        --------
            A list of perpendiculars from lap to ref_lap.
    """

    lap_list = lap[["LAT", "LON"]].values.tolist()
    ref_lap_list = ref_lap[["LAT", "LON"]].values.tolist()
    distances = 0
    distances_count = 0
    prev_i = -1
    for i in range(len(lap_list)):
        point = lap_list[i]

        closest_index = find_closest_point(point, ref_lap_list, prev_i)
        closest_point = ref_lap_list[closest_index]
        prev_i = closest_index

        neighbor_i = len(ref_lap) - 1 if closest_index == 0 else closest_index - 1
        neighbor1 = ref_lap_list[neighbor_i]
        neighbor_i = 0 if len(ref_lap) == closest_index + 1 else closest_index + 1
        neighbor2 = ref_lap_list[neighbor_i]

        v1 = create_vector(closest_point, point)
        v2 = create_vector(closest_point, neighbor1)
        v3 = create_vector(closest_point, neighbor2)

        angle1 = find_angle_between_vectors(v1, v2)
        angle2 = find_angle_between_vectors(v1, v3)

        degrees90 = math.pi / 2
        min_dist = -1
        if angle1 > degrees90 and angle2 > degrees90:
            min_dist = line_length(point[0], point[1], closest_point[0], closest_point[1])
        elif angle1 < degrees90 and angle2 < degrees90:
            dist1 = find_shortest_distance(point, closest_point, neighbor1)
            dist2 = find_shortest_distance(point, closest_point, neighbor2)
            min_dist = dist1 if dist1 <= dist2 else dist2
        elif angle1 <= degrees90:
            min_dist = find_shortest_distance(point, closest_point, neighbor1)
        elif angle2 <= degrees90:
            min_dist = find_shortest_distance(point, closest_point, neighbor2)

        if min_dist == -1:
            print('ERROR: Could not find distance')
            print("Indices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        elif math.isnan(min_dist):
            print("NAN value!!!\nIndices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        elif min_dist < 0:
            print("Negative value!!!\nIndices: {} {}\nAngles: {} {}".format(i, closest_index, angle1, angle2))
        else:
            min_dist = degrees2kilometers(min_dist) * 100000  # in centimeters
            distances += min_dist
            distances_count += 1

    return distances / distances_count


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