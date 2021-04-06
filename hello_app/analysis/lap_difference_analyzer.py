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
