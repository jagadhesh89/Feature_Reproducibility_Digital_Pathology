import numpy as np
from annoy import AnnoyIndex
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import joblib
import pandas as pd
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon
from scipy.spatial import distance_matrix
from collections import Counter
import math
import os
import argparse
from annoy import AnnoyIndex


def distance(centroid1, centroid2):
    return np.linalg.norm(np.array(centroid1) - np.array(centroid2))
	
def get_centroid_types(pred):
    centroids = []
    cell_types = []
    box = []
    contour = []
    prob = []
    pred_id = []
    for eachpred in pred:
        centroids.append(pred[eachpred]["centroid"])
        if pred[eachpred]["type"] == 4 or pred[eachpred]["type"] == 5:
            cell_types.append(0)
        else:
            if pred[eachpred]["type"] == 6 or pred[eachpred]["type"] == 8:
                cell_types.append(1)
            else:
                cell_types.append(pred[eachpred]["type"])
        box.append(pred[eachpred]["box"])
        contour.append(pred[eachpred]["contour"])
        prob.append(pred[eachpred]["prob"])
        pred_id.append(eachpred)
    return centroids, cell_types, box, contour, prob, pred_id
	
def find_non_overlapping_centroids(centroids1, centroids2, radius_threshold=10.0):
    non_overlapping_method1 = []
    non_overlapping_method2 = []

    for i in range(0,len(centroids1)):
        centroid1 = centroids1[i]
        overlaps = False
        for centroid2 in centroids2:
            if distance(centroid1, centroid2) <= radius_threshold:
                overlaps = True
                break

        if not overlaps:
            non_overlapping_method1.append(i)

    for i in range(0,len(centroids2)):
        overlaps = False
        centroid2 = centroids2[i]
        for centroid1 in centroids1:
            if distance(centroid2, centroid1) <= radius_threshold:
                overlaps = True
                break

        if not overlaps:
            non_overlapping_method2.append(i)

    return non_overlapping_method1, non_overlapping_method2
	
def merge_coordinates(coords_method1, coords_method2, pan_types, mon_types,id):
    merged_coords = []
    dim = len(coords_method1[0])
    
    # Build Annoy index with the coordinates from the Monusac
    t = AnnoyIndex(dim, 'euclidean')
    for i, coord in enumerate(coords_method2):
        t.add_item(i, coord)
    t.build(10)  
    
    distance_dict = {
        '0': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '1': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '2': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '3': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '6': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '7': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '8': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    }

    neighbor_count_dict = {
        '0': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '1': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '2': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '3': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '6': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '7': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0},
        '8': {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    }
    # For each coordinate in Pannuke, find coordinates in Pannuke within the radius threshold
    for i, coord1 in enumerate(coords_method1):
        #print(len(coords_method1))
        indices = t.get_nns_by_vector(coord1, n=2, search_k=-1, include_distances=True)
        #print(len(indices))
        for j, distance in zip(*indices):
            #print(j, distance)
            if distance == 0:
                continue
            pred1_type = str(pan_types[i])
            pred2_type = str(mon_types[j])
            pred1_id = id[i]
            pred2_id = id[j]
            #print(pred1_id,pred2_id)
            if pred1_id == pred2_id:
                continue
            #if pred1_type == '7' and pred2_type == '0':
            #    print(distance)
            distance_dict[pred1_type][pred2_type] =  float((distance_dict[pred1_type][pred2_type] + distance))/float(2)
            neighbor_count_dict[pred1_type][pred2_type] += 1

    return distance_dict, neighbor_count_dict


def calculate_polygon_area(region, vertices):
    polygon = [vertices[i] for i in region]
    return Polygon(polygon).area

def voronoi_region_areas(vor):
    areas = []
    for region_idx in vor.regions:
        if not region_idx or -1 in region_idx:
            continue
        area = calculate_polygon_area(region_idx, vor.vertices)
        areas.append(area)
    
    return np.array(areas)


def delaunay_edge_lengths(delaunay):
    simplices = delaunay.simplices
    edges = set()
    for simplex in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    edge_lengths = [np.linalg.norm(centroids[e[0]] - centroids[e[1]]) for e in edges]
    return np.array(edge_lengths)

def get_types_by_annoy(coord1, tree, mon_types, radius):    
    neighbor_types = []
    
    indices = tree.get_nns_by_vector(coord1, n=50, search_k=-1, include_distances=True)
    for j, distance in zip(*indices):
        #print(j, distance)
        if distance == 0:
            continue
        if distance >int(radius):
            continue
        pred2_type = str(mon_types[j])
        neighbor_types.append(pred2_type)

    return neighbor_types


def shannon_diversity_index(neighbors_types):
    type_counts = Counter(neighbors_types)
    total = sum(type_counts.values())
    shannon_index = -sum((count / total) * math.log(count / total) for count in type_counts.values() if count > 0)
    return shannon_index


def calculate_cellular_diversity(centroids, types, radius):
    
    diversity = []
    
    dim = len(centroids[0])
    
    # Build Annoy index with the coordinates from the Monusac
    t = AnnoyIndex(dim, 'euclidean')
    for i, coord in enumerate(centroids):
        t.add_item(i, coord)
    t.build(10)  

    temp_counter = 0
    for each_centroid in centroids:
        temp_counter += 1
        
        neighbors_types = get_types_by_annoy(each_centroid,t,types,radius)
        if temp_counter <10:
            print(types[temp_counter-1],neighbors_types)
        diversity_index = shannon_diversity_index(neighbors_types)
        diversity.append(diversity_index)
    
    return diversity
    
def remove_outliers_iqr(data):

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (data >= lower_bound) & (data <= upper_bound)

    return data[mask]

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Extract spatial features from centroids')
    #REQUIRED PARAMS
    parser.add_argument("-i", "--prediction_file",
                        dest="datfile",
                        help="The path to dat file",
                        required=True)
						
    inpArgs = parser.parse_args()

    wsi_pred_combined = joblib.load(inpArgs.datfile)
    slidename = os.path.basename(inpArgs.datfile).split(".")[0].replace("_HE_results.dat","")
    centroids, types, box, countour, prob, id = get_centroid_types(wsi_pred_combined)
    
    #Annoy to identify neighborhood nuclei
    type_dict = {
        0: "neoplastic epithelial",
        1: "Inflammatory",
        2: "Connective", 
        3: "Dead", 
        4: "non-neoplastic epithelial",
        5: "epithelial",
        6: "lymphocyte",
        7: "macrophage",
        8: "neutrophils"
    }
    
    radius_threshold = 10
    distance_dict, neighbor_count_dict = merge_coordinates(centroids, centroids, types, types,id)
    neighbor_df = pd.DataFrame(neighbor_count_dict)
    neighbor_df = neighbor_df[(neighbor_df.T != 0).any()]
    neighbor_df = neighbor_df.loc[:, (neighbor_df != 0).any(axis=0)]
    flat_data = neighbor_df.values.flatten()
    new_df = pd.DataFrame([flat_data], columns=[f'{col}_{i}' for col in neighbor_df.columns for i in range(neighbor_df.shape[0])])
    
    cell_neighbor_count_df = new_df

    centroids = np.array(centroids)
    types = np.array(types)
	
	
	## VORNOI AND DELAUNAY
    all_features = []

    for cell_type in np.unique(types):
        type_indices = np.where(types == cell_type)[0]
        type_centroids = centroids[type_indices]
        num_points = len(type_centroids)
        if num_points < 4:
            features['type'] = cell_type
            features['voronoi_area'] = 0
            features['cv_voronoi_area'] = 0
        else:
            vor = Voronoi(type_centroids)
            voronoi_areas = voronoi_region_areas(vor)
            valid_voronoi_areas = voronoi_areas[~np.isnan(voronoi_areas)]
            valid_voronoi_areas = remove_outliers_iqr(valid_voronoi_areas)
            
            mean_area = np.mean(valid_voronoi_areas)
            std_area = np.std(valid_voronoi_areas)
            cv_area = std_area / mean_area if mean_area != 0 else np.nan
            features = pd.DataFrame(type_centroids, columns=['x', 'y'])
            features['type'] = cell_type
            #features['voronoi_area'] = valid_voronoi_areas
            features['cv_voronoi_area'] = [cv_area] * len(type_centroids)
        
        delaunay = Delaunay(type_centroids)
        delaunay_edges = delaunay_edge_lengths(delaunay)
        features['delaunay_edge_length_mean'] = [np.mean(delaunay_edges)] * len(type_centroids)
        features['delaunay_edge_length_std'] = [np.std(delaunay_edges)] * len(type_centroids)
        
        all_features.append(features)

    final_features = pd.concat(all_features, ignore_index=True)
	
    voronoi_delaunay_dict = {}

    for eachtype in final_features['type'].unique():
        type_df = final_features[final_features['type'] == eachtype]
        feature_name=str(eachtype) + "_cv_voronoi_area"
        voronoi_delaunay_dict[feature_name] = np.mean(type_df["cv_voronoi_area"])
        feature_name=str(eachtype) + "_delaunay_edge_length_mean"
        voronoi_delaunay_dict[feature_name] = np.mean(type_df["delaunay_edge_length_mean"])
        feature_name=str(eachtype) + "_delaunay_edge_length_std"
        voronoi_delaunay_dict[feature_name] = np.mean(type_df["delaunay_edge_length_std"])
        

    voronoi_delaunay_df = pd.DataFrame([voronoi_delaunay_dict])
	
	
	## SHANNON DIVERSITY INDEX
    radius = 100

    diversity = calculate_cellular_diversity(centroids, types, radius)

    results_df = pd.DataFrame(centroids, columns=['x', 'y'])
    results_df['type'] = types
    results_df['diversity'] = diversity
    diversity_dict = {}
    for eachtype in results_df['type'].unique():
        type_df = results_df[results_df['type'] == eachtype]
        feature_name= str(eachtype) + "_shannon_div_index"
        diversity_dict[feature_name] = np.mean(type_df["diversity"])

    diversity_df = pd.DataFrame([diversity_dict])
	
	
	##COMBINE ALL FEATURES TO A DATAFRAME
    concatenated_df = pd.concat([cell_neighbor_count_df, voronoi_delaunay_df, diversity_df], axis=1)
    concatenated_df.to_csv("spatial_features/"+slidename+"_spatial_features.csv",index=False)


