# import pandas as pd
from pandas import DataFrame
import numpy as np
import math
from scipy.spatial import distance
import json
from datetime import datetime, date
# from analysis.lap_difference_analyzer import *
from .lap_difference_analyzer import *

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