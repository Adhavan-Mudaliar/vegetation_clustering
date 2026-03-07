import os
import io
import json
import base64
import datetime
import numpy as np
import pandas as pd
import osmnx as ox
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from sklearn.cluster import MiniBatchKMeans
from shapely.geometry import mapping, box
import geopandas as gpd

import requests
import zipfile

from sentinelhub import SHConfig, SentinelHubRequest, SentinelHubDownloadClient, BBox, CRS, DataCollection, MimeType, bbox_to_dimensions

from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Simple in-memory cache for API responses
response_cache = {}

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def array_to_base64_png(rgba_array):
    img = Image.fromarray(rgba_array, 'RGBA')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return "data:image/png;base64," + b64

def get_boundary(district, country):
    place_query = f"{district}, {country}"
    print(f"Fetching boundary for {place_query}...", flush=True)
    try:
        gdf = ox.geocode_to_gdf(place_query)
        geom = gdf.iloc[0].geometry
        return geom, gdf
    except Exception as e:
        print(f"Error fetching boundary: {e}", flush=True)
        raise

def download_sentinel(boundary_geom, district, country, client_id=None, client_secret=None):
    safe_name = f"{district.replace(' ', '_')}_{country.replace(' ', '_')}".lower()
    output_filename = f"sentinel_median_{safe_name}.tif"
    print(f"Checking for existing Sentinel-2 image at {output_filename}...", flush=True)
    if os.path.exists(output_filename):
        print(f"Found existing {output_filename}, skipping download.", flush=True)
        return output_filename

    import rasterio.transform

    config = SHConfig()
    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
    
    print("Fetching Sentinel-2 median composite from SentinelHub...", flush=True)
    
    minx, miny, maxx, maxy = boundary_geom.bounds
    bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "dataMask"],
            output: {
                bands: 8,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        if (sample.dataMask === 0) {
            return [0, 0, 0, 0, 0, 0, 0, 0];
        }
        return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B11];
    }
    """
    
    size_x, size_y = bbox_to_dimensions(bbox, resolution=20)
    
    # Cap dimensions to 512 to vastly improve performance
    if size_x > 512:
        ratio = 512 / size_x
        size_x = 512
        size_y = int(size_y * ratio)
    if size_y > 512:
        ratio = 512 / size_y
        size_y = 512
        size_x = int(size_x * ratio)

    request_sh = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                maxcc=0.2,
                mosaicking_order="leastCC"
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=[size_x, size_y],
        config=config
    )
    
    data = request_sh.get_data()
    
    if len(data) == 0:
        raise ValueError("No data returned from SentinelHub.")
        
    image_array = data[0]
    
    print(f"Saving downloaded image to {output_filename}... dimensions: {image_array.shape}", flush=True)
    
    bands = image_array.shape[-1]
    height = image_array.shape[0]
    width = image_array.shape[1]
    
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=image_array.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        if image_array.shape[-1] == bands:
            image_array = np.transpose(image_array, (2, 0, 1))
            
        for i in range(bands):
            dst.write(image_array[i], i + 1)
            
    return output_filename

def read_bands(image_path):
    print(f"Reading spectral bands from {image_path}...", flush=True)
    with rasterio.open(image_path) as src:
        bands = src.read()
        profile = src.profile
    
    b2 = bands[0].astype('float32')
    b3 = bands[1].astype('float32')
    b4 = bands[2].astype('float32')
    b5 = bands[3].astype('float32')
    b6 = bands[4].astype('float32')
    b7 = bands[5].astype('float32')
    b8 = bands[6].astype('float32')
    b11 = bands[7].astype('float32')
    
    return b2, b3, b4, b5, b6, b7, b8, b11, profile

def compute_ndvi(b4, b8):
    print("Computing NDVI...", flush=True)
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.where((b8 + b4) == 0., 0, (b8 - b4) / (b8 + b4))
    return ndvi

def compute_ndwi(b3, b8):
    print("Computing NDWI...", flush=True)
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.where((b3 + b8) == 0., 0, (b3 - b8) / (b3 + b8))
    return ndwi

def compute_red_edge_index(b5, b6, b7):
    print("Computing Red-Edge Vegetation Index...", flush=True)
    return (b5 + b6 + b7) / 3.0

def prepare_features(b2, b3, b4, b8, b11, ndvi, ndwi):
    print("Preparing feature stack for clustering...", flush=True)
    shape = b2.shape
    
    b2_flat = b2.flatten()
    b3_flat = b3.flatten()
    b4_flat = b4.flatten()
    b8_flat = b8.flatten()
    b11_flat = b11.flatten()
    ndvi_flat = ndvi.flatten()
    ndwi_flat = ndwi.flatten()
    
    features = np.column_stack([b2_flat, b3_flat, b4_flat, b8_flat, b11_flat, ndvi_flat, ndwi_flat])
    
    valid_mask = ~np.isnan(ndvi_flat) & (b2_flat != 0)
    valid_features = features[valid_mask]
    
    return valid_features, valid_mask, shape

def run_clustering(valid_features, valid_mask, shape):
    print("Running MiniBatchKMeans clustering (k=5)...", flush=True)
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=3, batch_size=2048)
    
    cluster_labels_valid = kmeans.fit_predict(valid_features)
    
    full_clusters_flat = np.full(shape[0] * shape[1], fill_value=-1, dtype=np.int32)
    full_clusters_flat[valid_mask] = cluster_labels_valid
    
    cluster_image = full_clusters_flat.reshape(shape)
    return cluster_image

def calculate_cluster_statistics(cluster_image, ndvi, red_edge):
    print("Calculating cluster statistics...", flush=True)
    stats = []
    # Pixel area = 10m * 10m = 100 m^2 = 0.0001 km^2
    pixel_area_km2 = 0.0001
    
    for c_id in range(5):
        mask = cluster_image == c_id
        pixel_count = np.sum(mask)
        
        if pixel_count == 0:
            stats.append({
                "Cluster": c_id,
                "Mean NDVI": 0,
                "Mean RedEdge": 0,
                "Pixel Count": 0,
                "Area_km2": 0
            })
            continue
            
        mean_ndvi = np.nanmean(ndvi[mask])
        mean_red_edge = np.nanmean(red_edge[mask])
        area_km2 = pixel_count * pixel_area_km2
        
        stats.append({
            "Cluster": c_id,
            "Mean NDVI": float(mean_ndvi),
            "Mean RedEdge": float(mean_red_edge),
            "Pixel Count": int(pixel_count),
            "Area_km2": float(area_km2)
        })
        
    df = pd.DataFrame(stats)
    return df

def assign_cluster_labels(cluster_stats):
    print("Assigning cluster labels based on spectral evidence...", flush=True)
    cluster_labels = {}
    assigned_types = []
    
    for _, row in cluster_stats.iterrows():
        c_id = int(row["Cluster"])
        ndvi_val = row["Mean NDVI"]
        
        if ndvi_val >= 0.65:
            label = "Dense Deciduous Forest (likely teak dominated)"
        elif ndvi_val >= 0.55:
            label = "Mixed Deciduous Forest"
        elif ndvi_val >= 0.40:
            label = "Bamboo / Shrub Vegetation"
        elif ndvi_val >= 0.30:
            label = "Grassland"
        elif ndvi_val >= 0:
            label = "Sparse / Degraded Grassland"
        else:
            label = "Water / Non-vegetation"
            
        cluster_labels[c_id] = label
        assigned_types.append(label)
        
    cluster_stats["Assigned Vegetation Type"] = assigned_types
    
    print("\n--- Spectral Clustering Results ---", flush=True)
    print(cluster_stats[["Cluster", "Mean NDVI", "Mean RedEdge", "Assigned Vegetation Type"]].to_string(index=False), flush=True)
    print("-----------------------------------\n", flush=True)
    
    return cluster_labels, cluster_stats

def generate_cluster_map_data(cluster_image, cluster_labels, profile):
    print("Generating cluster map base64 encoded data...", flush=True)
    h, w = cluster_image.shape
    
    label_colors = {
        "Grassland": "#e5f5f9", 
        "Water / Non-vegetation": "#1e90ff", 
        "Sparse / Degraded Grassland": "#ffd700", 
        "Mixed Deciduous Forest": "#2ca25f", 
        "Bamboo / Shrub Vegetation": "#99d8c9", 
    }
    
    rgba_clusters = np.zeros((h, w, 4), dtype=np.uint8)
    cluster_hex_colors = {}
    
    for i in range(5):
        mask = cluster_image == i
        label = cluster_labels.get(i, "Unknown")
        hex_color = label_colors.get(label, "#808080")
        cluster_hex_colors[i] = hex_color
        
        if len(hex_color) == 7:
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        else:
            r, g, b = 128, 128, 128
        rgba_clusters[mask, :] = [r, g, b, 255]
        
    bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
    lat_min, lat_max = bounds[1], bounds[3]
    lon_min, lon_max = bounds[0], bounds[2]
    
    b64_img = array_to_base64_png(rgba_clusters)
    
    legend = []
    for label, color in label_colors.items():
        legend.append({
            "color": color,
            "label": label
        })
        
    return {
        "image_data": b64_img,
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "opacity": 1.0,
        "legend": legend
    }

@app.route('/api/vegetation', methods=['GET'])
def get_vegetation_clustering():
    district = request.args.get('district')
    country = request.args.get('country')
    
    if not district or not country:
        return jsonify({"error": "Missing district or country parameters"}), 400
        
    cache_key = f"{district}_{country}".lower().strip()
    
    if cache_key in response_cache:
        print(f"Returning in-memory cached response for {district}, {country}", flush=True)
        return jsonify(response_cache[cache_key])
        
    try:
        print(f"--- Processing started for {district}, {country} ---", flush=True)
        # 1. Fetch boundary
        geom, gdf = get_boundary(district, country)
        
        # Convert GeoDataFrame to GeoJSON Dictionary
        district_boundary = json.loads(gdf.to_json())
        
        # Extract center from bbox
        minx, miny, maxx, maxy = geom.bounds
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2
        center = [center_lat, center_lon]
        
        # SENTINELHUB CREDENTIALS
        sh_client_id = "35e5cc9a-7035-4c81-bf32-6c6b3b160d57"
        sh_client_secret = "wXOhYtIKKLuMLUUNX8wwzS5KzQur1N2r"
        
        # 2. Download imagery via SentinelHub
        image_path = download_sentinel(geom, district, country, client_id=sh_client_id, client_secret=sh_client_secret)
        
        # 3 & 4. Load bounds & compute indices
        b2, b3, b4, b5, b6, b7, b8, b11, profile = read_bands(image_path)
        
        ndvi = compute_ndvi(b4, b8)
        ndwi = compute_ndwi(b3, b8)
        red_edge = compute_red_edge_index(b5, b6, b7)
        
        # 5. Prepare Features
        valid_features, valid_mask, shape = prepare_features(b2, b3, b4, b8, b11, ndvi, ndwi)
        
        # 6. Clustering
        cluster_image = run_clustering(valid_features, valid_mask, shape)
        
        # 7. Compute spectral stats & assign labels
        cluster_statistics = calculate_cluster_statistics(cluster_image, ndvi, red_edge)
        cluster_labels, cluster_stats_df = assign_cluster_labels(cluster_statistics)
        
        # Generate Map Data Dictionaries
        cluster_map_data = generate_cluster_map_data(cluster_image, cluster_labels, profile)
        print(f"--- Processing complete for {district}, {country} ---", flush=True)
        
        response_data = {
            "maps": {
                "cluster_map": cluster_map_data
            },
            "district_boundary": district_boundary,
            "center": center,
            "stats": cluster_stats_df.to_dict(orient='records')
        }
        
        response_cache[cache_key] = response_data
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask web server on port 5002...", flush=True)
    print("Test it via: http://127.0.0.1:5002/api/vegetation?district=Dang&country=India", flush=True)
    app.run(host='0.0.0.0', port=5002, debug=True)
