from django.shortcuts import render
from pyproj import Transformer
import pandas as pd
import numpy as np
import json
import os

from math import sqrt
import folium
from shapely.geometry import Point, Polygon

class RainIntensity:
    rund = 2 # round the number
    x = 222379
    y = 631702

    def __init__(self):
        self.P_boundary = Polygon([
            (277739, 797335),
            (263779, 793934),
            (258257, 776165),
            (269541, 760688),
            (261185, 746232),
            (252878, 709602),
            (250193, 665236),
            (257838, 640866),
            (233311, 580928),
            (209471, 534357),
            (194799, 385830),
            (159007, 398159),
            (147103, 501334),
            (136098, 565725),
            (146955, 618871),
            (177500, 666385),
            (198432, 743948),
            (216242, 780525),
            (277739, 797335),
        ])
        # # Transformer: ITM (EPSG:2039) → WGS84 (EPSG:4326)
        self.itm_to_wgs84 = Transformer.from_crs("EPSG:2039", "EPSG:4326", always_xy=True)

    def Gumbel_Cal(self, d, p, s):
        try:
            station_1_name = s['station_1_name']
            station_2_name = s['station_2_name']
            station_3_name = s['station_3_name']

            # Try to read the CSV files
            Gumble_Station_1 = pd.read_csv(f'necessary_files/Gumble_IDF/{station_1_name}_IDF_table.csv')
            Gumble_Station_2 = pd.read_csv(f'necessary_files/Gumble_IDF/{station_2_name}_IDF_table.csv')
            Gumble_Station_3 = pd.read_csv(f'necessary_files/Gumble_IDF/{station_3_name}_IDF_table.csv')
        
            # Try to access the values in the dataframes
            valu1 = round(Gumble_Station_1.iloc[d, p], self.rund)
            valu2  = round(Gumble_Station_2.iloc[d, p], self.rund)
            valu3 = round(Gumble_Station_3.iloc[d, p], self.rund)

            return [valu1, valu2, valu3]

        except KeyError as e:
            # Handle missing station names in the dictionary
            print(f"Error: Missing key {e} in the station dictionary.")
            return [None, None, None]
        except FileNotFoundError as e:
            # Handle file not found error when reading CSV files
            print(f"Error: File not found - {e.filename}")
            return [None, None, None]
        except IndexError as e:
            # Handle errors when accessing data in the dataframe
            print(f"Error: Invalid index {e}. Please check the indices p and d.")
            return [None, None, None]
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {str(e)}")
            return [None, None, None]
    
    def Log_Pearson_Cal(self, d, p, s):
        try:
            station_1_name = s['station_1_name']
            station_2_name = s['station_2_name']
            station_3_name = s['station_3_name']
        
            Station_1 = pd.read_excel(f'necessary_files/idf_tabels_log_pearson/{station_1_name}_Pearson_type3_IDF.xls', engine='xlrd')
            Station_2 = pd.read_excel(f'necessary_files/idf_tabels_log_pearson/{station_2_name}_Pearson_type3_IDF.xls', engine='xlrd')
            Station_3 = pd.read_excel(f'necessary_files/idf_tabels_log_pearson/{station_3_name}_Pearson_type3_IDF.xls', engine='xlrd')

            valu1 = round(Station_1.iloc[d, p],self.rund)
            valu2 = round(Station_2.iloc[d, p],self.rund)
            valu3 = round(Station_3.iloc[d, p],self.rund)

            return [valu1,valu2,valu3]
        except KeyError as e:
            # Handle missing station names in the dictionary
            print(f"Error: Missing key {e} in the station dictionary.")
            return [None, None, None]
        except FileNotFoundError as e:
            # Handle file not found error when reading CSV files
            print(f"Error: File not found - {e.filename}")
            return [None, None, None]
        except IndexError as e:
            # Handle errors when accessing data in the dataframe
            print(f"Error: Invalid index {e}. Please check the indices p and d.")
            return [None, None, None]
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {str(e)}")
            return [None, None, None]

    def Log_Normal_Cal(self, d, p, s):
        """Calculate Log-Normal values for three stations."""
        try:
            station_1_name = s['station_1_name']
            station_2_name = s['station_2_name']
            station_3_name = s['station_3_name']

            Station_1 = pd.read_csv(f'necessary_files/log_normal_idf_tables/{station_1_name}_z_normal.csv')
            Station_2 = pd.read_csv(f'necessary_files/log_normal_idf_tables/{station_2_name}_z_normal.csv')
            Station_3 = pd.read_csv(f'necessary_files/log_normal_idf_tables/{station_3_name}_z_normal.csv')

            valu1 = round(Station_1.iloc[d, p],self.rund)
            valu2 = round(Station_2.iloc[d, p],self.rund)
            valu3 = round(Station_3.iloc[d, p],self.rund)

            return [valu1,valu2,valu3]

        except KeyError as e:
            # Handle missing station names in the dictionary
            print(f"Error: Missing key {e} in the station dictionary.")
            return [None, None, None]
        except FileNotFoundError as e:
            # Handle file not found error when reading CSV files
            print(f"Error: File not found - {e.filename}")
            return [None, None, None]
        except IndexError as e:
            # Handle errors when accessing data in the dataframe
            print(f"Error: Invalid index {e}. Please check the indices p and d.")
            return [None, None, None]
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {str(e)}")
            return [None, None, None]


    def Plotting_Position_Cal(self, duration_H, period_Y, station_names):
        """
        Computes rainfall intensity for a given return period using plotting positions.

        Args:
            duration_H (str): Rainfall duration.
            period_Y (float): Return period.
            station_names (list): List of station names.

        Returns:
            dict: Dictionary containing interpolated values per station.
        """
        duration_map = {
            '10 min': 'x10_min', '20 min': 'x20_min', '30 min': 'x30_min',
            '40 min': 'x40_min', '50 min': 'x50_min', '1 hr': 'x1_hr',
            '1.5 hr': 'x1_5_hr', '2 hr': 'x2_hr', '2.5 hr': 'x2_5_hr',
            '3 hr': 'x3_hr', '3.5 hr': 'x3_5_hr', '4 hr': 'x4_hr',
            '4.5 hr': 'x4_5_hr', '5 hr': 'x5_hr', '6 hr': 'x6_hr'
        }
        
        prefix = duration_map.get(duration_H)
        if prefix is None or period_Y is None:
            print("Invalid duration or return period selected.")
            return None

        folder_path = 'necessary_files/csv_frequency_analysis_files/'
        results = {}

        for i, station_name in enumerate(station_names):
            file_name = f'{prefix}_{station_name}.csv'
            file_path = os.path.join(folder_path, file_name)

            try:
                data = pd.read_csv(file_path)
                return_periods = data.iloc[:, 4].values
                rain_intensities = data.iloc[:, 1].values
                
                sorted_indices = np.argsort(return_periods)
                return_periods_sorted = return_periods[sorted_indices]
                rain_intensities_sorted = rain_intensities[sorted_indices]
                
                if period_Y in return_periods:
                    valuePP = rain_intensities[np.where(return_periods == period_Y)[0][0]]
                else:
                    if period_Y > max(return_periods):
                        print(f"Return period {period_Y} years is out of range for station {station_name}.")
                        valuePP = None
                    else:
                        valuePP = np.interp(period_Y, return_periods_sorted, rain_intensities_sorted)
                        
                results[i] = round(valuePP, self.rund) if valuePP is not None else None

            except FileNotFoundError:
                print(f"File not found: {file_name}")
                results[i] = None
            except Exception as e:
                print(f"Error processing {station_name}: {e}")
                results[i] = None

        return results
    
    def haversine(self, itm_x1, itm_y1, itm_x2, itm_y2):
        """Calculate distance between two ITM points"""
        ss = sqrt(pow(itm_x2 - itm_x1,2)+pow(itm_y2 - itm_y1,2))/1000
        return ss

    def stations_coordinates_file_reader(self, itm_x, itm_y):
        """Read station coordinates and calculate the distance to the given ITM point."""
        try:
            # Read the Excel file containing station coordinates
            P_stations = pd.read_excel('necessary_files/Stations/stations_coordinations.xlsx')

            if P_stations.empty:
                print("Error: The stations coordination file is empty.")
                return False

            # Calculate Haversine distance for each station by converting ITM to WGS84
            P_stations['distance'] = P_stations.apply(
                lambda row: self.haversine(itm_x, itm_y, row['x'], row['y']), axis=1
            )

            # Find the indices of the three shortest distances
            sorted_stations = P_stations.nsmallest(3, 'distance')
            return sorted_stations

        except FileNotFoundError:
            print("Error: The stations coordination file was not found. Ensure the file path is correct.")
            return None
        except KeyError as e:
            print(f"Error: Missing column in the file. {e}")
            return None
        except pd.errors.EmptyDataError:
            print("Error: The file is empty or invalid.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def process_coordinates(self, itm_x, itm_y):
        """Check if given ITM coordinates are inside the defined ITM boundary."""
        try:
            itm_x, itm_y = float(itm_x), float(itm_y)
            point = Point(itm_x, itm_y)
            return self.P_boundary.contains(point)
        except ValueError:
            return False

    def map_show(self, itm_x, itm_y, zoom):
        """Generate a map centered on ITM coordinates by converting them to WGS84."""
        lon, lat = self.itm_to_wgs84.transform(itm_x, itm_y)
        # lon, lat = itm_x, itm_y
        f = folium.Figure(width=1000, height=500)
        m = folium.Map(
            location=[lat, lon],  # Use converted WGS84 coordinates
            zoom_start=zoom,
            min_lat=29.45,
            min_lon=34.1,
            max_lat=33.30,
            max_lon=35.9
        ).add_to(f)
        
        folium.Marker((lat,lon)).add_to(m)

        return m

    def Average_Intensity_Cal(self, total_sum, d):
        try:
            average = total_sum / d
            return average
        except ZeroDivisionError:
            return 0
        except TypeError:
            return None
    
    def idw_weighted_average(self, distances, gumbel_vals, pearson_vals, normal_vals, selected_distributions):
        try:
            # Add a small epsilon to avoid division by zero
            epsilon = np.finfo(float).eps
            d1, d2, d3 = distances[0] + epsilon, distances[1] + epsilon, distances[2] + epsilon

            # Calculate weights
            weights = np.array([1 / d1**2, 1 / d2**2, 1 / d3**2])
            weights /= weights.sum()

            # Calculate weighted averages
            I_gumble = np.dot(gumbel_vals if gumbel_vals else [0, 0, 0], weights)
            I_pearson = np.dot(pearson_vals if pearson_vals else [0, 0, 0], weights)
            I_normal = np.dot(normal_vals if normal_vals else [0, 0, 0], weights)

            # Select the required distributions
            selected_values = []
            if 'gumble' in selected_distributions:
                selected_values.append(I_gumble)
            if 'log-pearson' in selected_distributions:
                selected_values.append(I_pearson)
            if 'log-normal' in selected_distributions:
                selected_values.append(I_normal)

            # Return the mean of selected values or 0 if no distributions are selected
            return np.mean(selected_values) if selected_values else 0

        except ZeroDivisionError:
            print("Error: Division by zero occurred due to invalid distances.")
            return None
        except TypeError:
            print("Error: Invalid input type. Ensure all inputs are of the correct type.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    
    def linear_interpolation(self, distances, gumbles, log_pearsons, log_normals, selected_distributions):
        """
        Performs inverse distance weighted interpolation.
        
        Args:
            distances (list): Distances to the 3 closest stations.
            gumbles (list): Gumbel distribution data.
            log_pearsons (list): Log-Pearson distribution data.
            log_normals (list): Log-Normal distribution data.
            selected_distributions (list): Selected distributions for interpolation.

        Returns:
            float: Interpolated value.
        """

        # Convert lists to numpy arrays
        distances = np.array(distances)
        gumbles = np.array(gumbles) if gumbles else 0
        log_pearsons = np.array(log_pearsons) if log_pearsons else 0 
        log_normals = np.array(log_normals)if log_normals else 0

        # Compute weights as inverse distances
        weights = 1 / distances

        arr=[1/distances[0]**2/(1/distances[0]**2+1/distances[1]**2+1/distances[2]**2), 
             1/distances[1]**2/(1/distances[0]**2+1/distances[1]**2+1/distances[2]**2),
             1/distances[2]**2/(1/distances[0]**2+1/distances[1]**2+1/distances[2]**2),
            ]

        # Compute weighted averages for each distribution
        Ig = np.sum(arr * gumbles) / np.sum(arr)
        Ip = np.sum(arr * log_pearsons) / np.sum(arr)
        In = np.sum(arr * log_normals) / np.sum(arr)

        # Select the requested distributions
        selected_interp_values = []
        
        if 'gumble' in selected_distributions:
            selected_interp_values.append(Ig)

        if 'log-pearson' in selected_distributions:
            selected_interp_values.append(Ip)

        if 'log-normal' in selected_distributions:
            selected_interp_values.append(In)

        # Compute the final interpolated value
        interp_value = np.mean(selected_interp_values) if selected_interp_values else 0

        return interp_value if interp_value else 0
    
    def handle_request(self, request):
        """Handle POST request for the Rain Intensity page."""
        map_html = self.map_show(self.x, self.y, 8)._repr_html_()
        durations= [
            "10 minutes", "20 minutes", "30 minutes", "40 minutes", "50 minutes", 
            "1 hr", "1.5 hr", "2 hours", "2.5 hours", "3 hours", "3.5 hours", 
            "4 hr", "4.5 hours", "5 hours", "6 hours"
        ]
        
        average_table=  np.zeros((15,9))

        table_data = list(zip(durations, average_table))
        context = {
            "average_table":table_data,
            "headers": ["2 Years", "5 Years", "10 Years", "25 Years", "50 Years", "100 Years", "200 Years", "500 Years", "1000 Years"],
            "return_periods":[2, 5, 10, 25, 50, 100, 200, 500, 1000],
            "value": [[0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,]],
            "Map": map_html
            }

        if request.method == "POST":
            action = request.POST.get("action")

            if action == "calculate":
                # Use the formula_view to calculate the average TC
                context = self.RI_formula_view(request)

            if action == "calculateRunoff":
                pass

        return render(request, "RI.html", context)
    
    def calculate_idf_average_table(self, selected_methods, stations):
        """
        Calculate the IDF average table based on the selected methods and stations.

        Args:
            selected_methods (list): List of selected calculation methods (e.g., 'gumbel', 'log-pearson', 'log-normal').
            stations (list): List of station data.

        Returns:
            list: 2D array where rows are durations and columns are return periods with average intensity values.
        """

        # Map return periods and durations to indices
        return_period_mapping = {
            "2 Years": 1, "5 Years": 2, "10 Years": 3, "25 Years": 4,
            "50 Years": 5, "100 Years": 6, "200 Years": 7, "500 Years": 8, "1000 Years": 9,
        }

        duration_mapping = {
            "10 min": 0, "20 min": 1, "30 min": 2, "40 min": 3, "50 min": 4, "1 hr": 5,
            "1.5 hr": 6, "2 hr": 7, "2.5 hr": 8, "3 hr": 9, "3.5 hr": 10, "4 hr": 11,
            "4.5 hr": 12, "5 hr": 13, "6 hr": 14,
        }

        # Initialize 2D array
        num_durations = len(duration_mapping)
        num_return_periods = len(return_period_mapping)
        average_table = [[0 for _ in range(num_return_periods)] for _ in range(num_durations)]
        gumbel_value = []
        
        
        # Iterate over durations and return periods
        for duration_index, duration in enumerate(duration_mapping.values()):
            for return_period_index, return_period in enumerate(return_period_mapping.values()):
                method_sums = []
                # Calculate using selected methods
                if 'gumble' in selected_methods:
                    gumbel_value = self.Gumbel_Cal(duration, return_period, stations)
                    method_sums.append(gumbel_value)
            
                if 'log-pearson' in selected_methods:
                    log_pearson_value = self.Log_Pearson_Cal(duration, return_period, stations)
                    method_sums.append(log_pearson_value)
            
                if 'log-normal' in selected_methods:
                    log_normal_value = self.Log_Normal_Cal(duration, return_period, stations)
                    method_sums.append(log_normal_value)

                # Calculate the average intensity
                if method_sums:
                    flat_method_sums = [value for sublist in method_sums for value in sublist]
                    
                    total_sum = sum(flat_method_sums)
                    average_intensity = total_sum / len(flat_method_sums)
                    average_table[duration_index][return_period_index] = round(average_intensity, self.rund)

        return average_table

    def RI_formula_view(self, request):
        """Calculate rain intensity based on user inputs."""
        selected_checkbox = request.POST.getlist("distribution") + request.POST.getlist("adjusted") + request.POST.getlist("plotting")
        x_coordinate = request.POST.get("x_coordinate") 
        y_coordinate = request.POST.get("y_coordinate")
        period = request.POST.get("return_period")
        duration = request.POST.get("duration")
        A_dunam = request.POST.get("A_dunam") 
        Runoff_Coefficient = request.POST.get("Runoff_Coefficient")
        Qp = 0

        # Map values to indices
        return_period_mapping = {
            "2 Years": 1,
            "5 Years": 2,
            "10 Years": 3,
            "25 Years": 4,
            "50 Years": 5,
            "100 Years": 6,
            "200 Years": 7,
            "500 Years": 8,
            "1000 Years": 9,
        }
        return_periods = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
        
        duration_mapping = {
            "10 min": 0,
            "20 min": 1,
            "30 min": 2,
            "40 min": 3,
            "50 min": 4,
            "1 hr": 5,
            "1.5 hr": 6,
            "2 hr": 7,
            "2.5 hr": 8,
            "3 hr": 9,
            "3.5 hr": 10,
            "4 hr": 11,
            "4.5 hr": 12,
            "5 hr": 13,
            "6 hr": 14,
        }

        station_data = {
            "station_1_name": '',
            "station_1_km": '',
            "station_2_name": '',
            "station_2_km": '',
            "station_3_name": '',
            "station_3_km": '',
        }
        durations= ["10 minutes", "20 minutes", "30 minutes", "40 minutes", "50 minutes", 
                    "1 hr", "1.5 hr", "2 hours", "2.5 hours", "3 hours", "3.5 hours", 
                    "4 hr", "4.5 hours", "5 hours", "6 hours"
                    ]
        gumbles = []
        log_pearsons = []
        log_normals = []
        plotting_positions = []
        selected_distributions = []
        total_sum = 0
        average_intensity, average_idw, average_linear = 0, 0, 0
        
        distances = []
        average_table = []
        table_data = []
        final_intensity_division = 0
        
        d = 1
        m = self.map_show(self.x, self.y, 8)

        period_index = return_period_mapping.get(period, None)
        duration_index = duration_mapping.get(duration, None)
        
        if self.process_coordinates(x_coordinate, y_coordinate) == True:
            m = self.map_show(float(x_coordinate), float(y_coordinate), 13)
            closest_stations = self.stations_coordinates_file_reader(float(x_coordinate), float(y_coordinate))
            station_data = {
                "station_1_name": closest_stations.iloc[0]['Name'],
                "station_1_km": round(closest_stations.iloc[0]['distance'], 2),
                "station_2_name": closest_stations.iloc[1]['Name'],
                "station_2_km": round(closest_stations.iloc[1]['distance'], 2),
                "station_3_name": closest_stations.iloc[2]['Name'],
                "station_3_km": round(closest_stations.iloc[2]['distance'], 2),
            }
            distances = [station_data['station_1_km'], station_data['station_2_km'], station_data['station_3_km']]
            stations_names = [station_data['station_1_name'], station_data['station_2_name'], station_data['station_3_name']]

            folium.Marker([float(x_coordinate), float(y_coordinate)], popup="Valid Point").add_to(m)
            
            """ Start Calculate the equations """
            if 'gumble' in selected_checkbox:
                gumbles = self.Gumbel_Cal(duration_index, period_index, station_data)
                selected_distributions.append('gumble')
            
            if 'log-pearson' in selected_checkbox:
                log_pearsons = self.Log_Pearson_Cal(duration_index, period_index, station_data)
                selected_distributions.append('log-pearson')
            
            if 'log-normal' in selected_checkbox:
                log_normals = self.Log_Normal_Cal(duration_index, period_index, station_data)
                selected_distributions.append('log-normal')

            if 'plotting-position' in selected_checkbox:
                plotting_positions = self.Plotting_Position_Cal(duration, return_periods[period_index-1], stations_names)
            
            if 'intensity' in selected_checkbox:
                total_sum = sum(gumbles) + sum(log_pearsons) + sum(log_normals)
                d = len(gumbles) + len(log_pearsons) + len(log_normals)
                average_intensity = self.Average_Intensity_Cal(total_sum,d)
                final_intensity_division += 1
            
            if 'idw' in selected_checkbox:
                average_idw = self.idw_weighted_average(distances, gumbles, log_pearsons, log_normals, selected_distributions)
                final_intensity_division += 1
            
            if 'linear' in selected_checkbox:
               average_linear = self.linear_interpolation(distances, gumbles, log_pearsons, log_normals, selected_distributions)
               final_intensity_division += 1
            
            final_intensity = 0 if final_intensity_division == 0 else (average_intensity + average_idw + average_linear) / final_intensity_division
            Qp = 0 if A_dunam == '' or Runoff_Coefficient == '' or final_intensity == 0 else float(A_dunam) * float(Runoff_Coefficient) * float(final_intensity)
                
            average_table = self.calculate_idf_average_table(selected_distributions, station_data)
            table_data = list(zip(durations, average_table))

        else:
            print("out of range!!!!!")
            station_data = {
                "station_1_name": '',
                "station_1_km": '',
                "station_2_name": '',
                "station_2_km": '',
                "station_3_name": '',
                "station_3_km": '',
            }
            # average_table=  np.zeros((15,9))
            table_data = list(zip(durations, average_table))
            average_table = [[0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,], 
                      [0, 0, 0, 0, 0, 0, 0, 0, 0,]]
            final_intensity = 0
            table_data = list(zip(durations, average_table))
        # Render the map as an HTML string
        map_html = m._repr_html_()
        return {
            "x_coordinate": x_coordinate,
            "y_coordinate": y_coordinate,
            "return_period_index": period_index,
            "return_period": json.dumps(period),
            'duration': json.dumps(duration),
            "selected_checkbox": selected_checkbox,
            "gumble_1": gumbles[0] if gumbles else '',
            "gumble_2": gumbles[1] if gumbles else '',
            "gumble_3": gumbles[2] if gumbles else '',
            "log_pearson_1": log_pearsons[0] if log_pearsons else '',
            "log_pearson_2": log_pearsons[1] if log_pearsons else '',
            "log_pearson_3": log_pearsons[2] if log_pearsons else '',
            "log_normal_1": log_normals[0] if log_normals else '',
            "log_normal_2": log_normals[1] if log_normals else '',
            "log_normal_3": log_normals[2] if log_normals else '',
            "plotting_position_1": plotting_positions[0] if plotting_positions else '',
            "plotting_position_2": plotting_positions[1] if plotting_positions else '',
            "plotting_position_3": plotting_positions[2] if plotting_positions else '',
            "average_intensity": round(average_intensity,self.rund) if average_intensity else '',
            "average_idw": round(average_idw,self.rund) if average_idw else '',
            "linear_interpolation": round(average_linear,self.rund) if average_linear else '',
            **station_data,
            "final_intensity":json.dumps(round(final_intensity,self.rund)),
            "Qp": round(Qp, self.rund),
            "A_dunam":A_dunam,
            "Runoff_Coefficient":Runoff_Coefficient,
            "average_table": table_data,
            "return_periods": return_periods,
            "value": json.dumps(average_table),
            "headers": ["2 Years", "5 Years", "10 Years", "25 Years", "50 Years", "100 Years", "200 Years", "500 Years", "1000 Years"],
            "Map": map_html,
        }

def RI(request):
    """
    View function for the 'Rain Intensity' page.
    """
    view = RainIntensity()
    return view.handle_request(request)
