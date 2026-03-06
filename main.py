import os
import io
import datetime
import numpy as np
import pandas as pd
import osmnx as ox
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from sklearn.cluster import KMeans
import folium
from shapely.geometry import mapping, box
import geopandas as gpd

import ee
import requests
import zipfile

from sentinelhub import SHConfig, SentinelHubRequest, SentinelHubDownloadClient, BBox, CRS, DataCollection, MimeType, bbox_to_dimensions

def get_boundary(district, country):
    """
    Use osmnx to fetch the boundary of the given district and country.
    Returns a GeoJSON-like dictionary and a shapely polygon.
    """
    place_query = f"{district}, {country}"
    print(f"Fetching boundary for {place_query}...")
    try:
        # Fetch the geometry
        gdf = ox.geocode_to_gdf(place_query)
        geom = gdf.iloc[0].geometry
        return geom, gdf
    except Exception as e:
        print(f"Error fetching boundary: {e}")
        raise

def download_sentinel(boundary_geom, client_id=None, client_secret=None):
    """
    Download a median composite of Sentinel-2 surface reflectance
    for the given boundary using SentinelHub.
    """
    output_filename = "sentinel_median_8bands.tif"
    if os.path.exists(output_filename):
        print(f"Found existing {output_filename}, skipping download.")
        return output_filename

    import rasterio.transform

    config = SHConfig()
    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
    
    print("Fetching Sentinel-2 median composite from SentinelHub...")
    
    # Get bounding box of the geometry
    minx, miny, maxx, maxy = boundary_geom.bounds
    bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
    
    end_date = datetime.datetime.now()
    # SentinelHub Free Tier struggles with massive temporal medians. 
    # Use 30 days composite instead of 365 days.
    start_date = end_date - datetime.timedelta(days=30)
    time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Find the single least cloudy scene
    # We want B2, B3, B4, B5, B6, B7, B8, B11. 
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
    
    # Let's cap at 1000x1000 pixels to be safe for free tier processing units (PU limits)
    size_x, size_y = bbox_to_dimensions(bbox, resolution=10)
    
    if size_x > 1000:
        ratio = 1000 / size_x
        size_x = 1000
        size_y = int(size_y * ratio)
    if size_y > 1000:
        ratio = 1000 / size_y
        size_y = 1000
        size_x = int(size_x * ratio)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                maxcc=0.2, # Extremely strict cloud cover filter to pick a single clean image
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
    
    data = request.get_data()
    
    if len(data) == 0:
        raise ValueError("No data returned from SentinelHub.")
        
    image_array = data[0] # The median composite
    
    print(f"Saving downloaded image to {output_filename}... dimensions: {image_array.shape}")
    
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
    print(f"Reading spectral bands from {image_path}...")
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
    print("Computing NDVI...")
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.where((b8 + b4) == 0., 0, (b8 - b4) / (b8 + b4))
    return ndvi

def compute_ndwi(b3, b8):
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.where((b3 + b8) == 0., 0, (b3 - b8) / (b3 + b8))
    return ndwi

def compute_red_edge_index(b5, b6, b7):
    print("Computing Red-Edge Vegetation Index...")
    return (b5 + b6 + b7) / 3.0

def prepare_features(b2, b3, b4, b8, b11, ndvi, ndwi):
    print("Preparing feature stack for clustering...")
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
    print("Running KMeans clustering (k=5)...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    
    cluster_labels_valid = kmeans.fit_predict(valid_features)
    
    full_clusters_flat = np.full(shape[0] * shape[1], fill_value=-1, dtype=np.int32)
    full_clusters_flat[valid_mask] = cluster_labels_valid
    
    cluster_image = full_clusters_flat.reshape(shape)
    return cluster_image

def calculate_cluster_statistics(cluster_image, ndvi, red_edge):
    print("Calculating cluster statistics...")
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
    print("Assigning cluster labels based on spectral evidence...")
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
        
    # Add to dataframe for final output
    cluster_stats["Assigned Vegetation Type"] = assigned_types
    
    print("\n--- Spectral Clustering Results ---")
    print(cluster_stats[["Cluster", "Mean NDVI", "Mean RedEdge", "Assigned Vegetation Type"]].to_string(index=False))
    print("-----------------------------------\n")
    
    return cluster_labels, cluster_stats

def download_esa_worldcover(boundary_geom, ee_project=None):
    print("Initializing Earth Engine...")
    try:
        if ee_project:
            ee.Initialize(project=ee_project)
        else:
            ee.Initialize()
    except Exception as e:
        print(f"Earth Engine not initialized: {e}")
        print("Please update 'ee_project' in main.py with your Google Cloud Project ID.")
        raise


    output_filename = "worldcover_reference.tif"
    if os.path.exists(output_filename):
        print(f"Found existing {output_filename}, skipping download.")
        return output_filename
        
    print("Downloading ESA WorldCover from Earth Engine...")
    minx, miny, maxx, maxy = boundary_geom.bounds
    region = ee.Geometry.BBox(minx, miny, maxx, maxy)
    
    dataset = ee.ImageCollection("ESA/WorldCover/v100").first()
    image = dataset.select('Map').clip(region)
    
    url = image.getDownloadURL({
        'dimensions': 2000,
        'region': region,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326',
        'maxPixels': 1e9
    })
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download from EE: {response.text}")
        
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            tif_name = [n for n in z.namelist() if n.endswith('.tif')][0]
            with open(output_filename, 'wb') as f:
                f.write(z.read(tif_name))
    except zipfile.BadZipFile:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
            
    print(f"Saved WorldCover to {output_filename}")
    return output_filename

def process_worldcover_raster(worldcover_path, target_profile):
    print("Resampling WorldCover raster to match cluster resolution...")
    resampled_data = np.zeros((target_profile['height'], target_profile['width']), dtype=np.uint8)
    
    with rasterio.open(worldcover_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=resampled_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile['transform'],
            dst_crs=target_profile['crs'],
            resampling=Resampling.nearest
        )
        
    return resampled_data

def calculate_cluster_overlap(cluster_image, worldcover_image):
    print("Calculating cluster vs WorldCover overlap...")
    
    class_mapping = {
        10: "Tree Cover",
        20: "Shrubland",
        30: "Grassland"
    }
    
    stats = []
    
    for c_id in range(5):
        mask = cluster_image == c_id
        cluster_pixels = np.sum(mask)
        if cluster_pixels == 0:
            continue
            
        wc_pixels = worldcover_image[mask]
        
        unique, counts = np.unique(wc_pixels, return_counts=True)
        overlap_dict = dict(zip(unique, counts))
        
        best_class = None
        best_count = -1
        
        vegetation_found = False
        for wc_class, cnt in overlap_dict.items():
            if wc_class in class_mapping:
                vegetation_found = True
                if cnt > best_count:
                    best_count = cnt
                    best_class = wc_class
                
        if vegetation_found and best_class is not None:
            dominant_type = class_mapping[best_class]
            percentage = (best_count / cluster_pixels) * 100
        else:
            # Not primarily vegetation. Find the actual dominant type.
            max_cnt = max(overlap_dict.values())
            top_cls = [k for k, v in overlap_dict.items() if v == max_cnt][0]
            if top_cls == 40: dominant_type = "Cropland"
            elif top_cls == 50: dominant_type = "Built-up"
            elif top_cls == 60: dominant_type = "Bare / sparse vegetation"
            elif top_cls == 70: dominant_type = "Snow and ice"
            elif top_cls == 80: dominant_type = "Permanent water bodies"
            elif top_cls == 90: dominant_type = "Herbaceous wetland"
            elif top_cls == 95: dominant_type = "Mangroves"
            elif top_cls == 100: dominant_type = "Moss and lichen"
            else: dominant_type = "Other"
            
            percentage = (max_cnt / cluster_pixels) * 100
            
        stats.append({
            "Cluster": c_id,
            "Dominant Land Cover": dominant_type,
            "Overlap %": f"{percentage:.0f}%"
        })
        
    df = pd.DataFrame(stats)
    print("")
    print("--- Validation Statistics ---")
    print(df.to_string(index=False))
    print("-----------------------------")
    print("")
    return df

def generate_named_cluster_map(cluster_image, cluster_labels, profile, boundary_gdf):
    print("Generating named cluster map with folium...")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    h, w = cluster_image.shape
    
    # 1. Dynamic colors for classes
    label_colors = {
        "Grassland": "#e5f5f9", # Very Light Mint "Sparse / Degraded Grassland": "#ffd700", # Gold
        "Water / Non-vegetation": "#1e90ff", # Dodger Blue
        "Sparse/Degraded Grassland": "#00441b", # Dark Green
        "Mixed Deciduous Forest": "#2ca25f", # Medium Green
        "Bamboo / Shrub Vegetation": "#99d8c9", # Light Teal Green
    }
    
    rgba_clusters = np.zeros((h, w, 4), dtype=np.uint8)
    cluster_hex_colors = {}
    for i in range(5):
        mask = cluster_image == i
        label = cluster_labels.get(i, "Unknown")
        hex_color = label_colors.get(label, "#808080")
        cluster_hex_colors[i] = hex_color
        
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        rgba_clusters[mask, :] = [r, g, b, 255]
        
    bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
    lat_min, lat_max = bounds[1], bounds[3]
    lon_min, lon_max = bounds[0], bounds[2]
    
    image_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    
    # --- Map 1: Cluster Map ---
    m_cluster = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    fg_clusters = folium.FeatureGroup(name='Vegetation Clusters', show=True)
    folium.raster_layers.ImageOverlay(
        image=rgba_clusters,
        bounds=image_bounds,
        opacity=1.0,
        name='Vegetation Clusters',
        interactive=True,
        cross_origin=False
    ).add_to(fg_clusters)
    fg_clusters.add_to(m_cluster)
    
    folium.GeoJson(
        boundary_gdf,
        name="District Boundary",
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 2}
    ).add_to(m_cluster)
    
    legend_html_cluster = f'''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 380px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;
     ">
         <b>Vegetation Classes (Based on Spectral Indices)</b><br>
         <i style="background:{cluster_hex_colors.get(0, '#808080')};width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 0 — {cluster_labels.get(0, "C0")}<br>
         <i style="background:{cluster_hex_colors.get(1, '#808080')};width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 1 — {cluster_labels.get(1, "C1")}<br>
         <i style="background:{cluster_hex_colors.get(2, '#808080')};width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 2 — {cluster_labels.get(2, "C2")}<br>
         <i style="background:{cluster_hex_colors.get(3, '#808080')};width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 3 — {cluster_labels.get(3, "C3")}<br>
         <i style="background:{cluster_hex_colors.get(4, '#808080')};width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 4 — {cluster_labels.get(4, "C4")}<br>
     </div>
     '''
    m_cluster.get_root().html.add_child(folium.Element(legend_html_cluster))
    folium.LayerControl().add_to(m_cluster)
    
    return m_cluster

def generate_worldcover_map(worldcover_image, profile, boundary_gdf):
    h, w = worldcover_image.shape
    worldcover_colors = {
        10: [0, 100, 0, 255],     # Tree cover
        20: [255, 187, 34, 255],  # Shrubland
        30: [255, 255, 76, 255]   # Grassland
    }
    rgba_worldcover = np.zeros((h, w, 4), dtype=np.uint8)
    for val, col in worldcover_colors.items():
        mask = worldcover_image == val
        rgba_worldcover[mask, :] = col
        
    bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
    lat_min, lat_max = bounds[1], bounds[3]
    lon_min, lon_max = bounds[0], bounds[2]
    image_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2

    m_wc = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    fg_wc = folium.FeatureGroup(name='ESA WorldCover (Vegetation classes)', show=True)
    folium.raster_layers.ImageOverlay(
        image=rgba_worldcover,
        bounds=image_bounds,
        opacity=1.0,
        name='WorldCover Vegetation',
        interactive=True,
        cross_origin=False
    ).add_to(fg_wc)
    fg_wc.add_to(m_wc)
    
    folium.GeoJson(
        boundary_gdf,
        name="District Boundary",
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 2}
    ).add_to(m_wc)
    
    legend_html_wc = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 120px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;
     ">
         <b>WorldCover</b><br>
         <i style="background:rgb(0,100,0);width:15px;height:15px;float:left;margin-right:5px;"></i> Tree Cover<br>
         <i style="background:rgb(255,187,34);width:15px;height:15px;float:left;margin-right:5px;"></i> Shrubland<br>
         <i style="background:rgb(255,255,76);width:15px;height:15px;float:left;margin-right:5px;"></i> Grassland<br>
     </div>
     '''
    m_wc.get_root().html.add_child(folium.Element(legend_html_wc))
    folium.LayerControl().add_to(m_wc)
    
    return m_wc

def export_results(stats_df, map_cluster, map_wc, cluster_stats):
    print("Exporting validation results...")
    if stats_df is not None:
        stats_df.to_csv("cluster_validation_stats.csv", index=False)
        print("Saved cluster_validation_stats.csv")
    if cluster_stats is not None:
        cluster_stats.to_csv("cluster_spectral_stats.csv", index=False)
        print("Saved cluster_spectral_stats.csv")
    if map_cluster is not None:
        map_cluster.save("cluster_named_map.html")
        print("Saved cluster_named_map.html")
    if map_wc is not None:
        map_wc.save("worldcover_map.html")
        print("Saved worldcover_map.html")

def main():
    district = "Dang"
    country = "India"
    
    # 1. Fetch boundary
    geom, gdf = get_boundary(district, country)
    
    # SENTINELHUB CREDENTIALS
    sh_client_id = "35e5cc9a-7035-4c81-bf32-6c6b3b160d57"
    sh_client_secret = "wXOhYtIKKLuMLUUNX8wwzS5KzQur1N2r"
    
    # 2. Download imagery via SentinelHub
    image_path = download_sentinel(geom, client_id=sh_client_id, client_secret=sh_client_secret)
    
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
    
    # --- ESA WORLDCOVER VALIDATION PIPELINE ---
    ee_project = "forest-monitoring-hackathon" # <-- UPDATE THIS IF INITIALIZE FAILS
    worldcover_path = download_esa_worldcover(geom, ee_project)
    worldcover_image = process_worldcover_raster(worldcover_path, profile)
    
    stats_df = calculate_cluster_overlap(cluster_image, worldcover_image)
    
    map_cluster = generate_named_cluster_map(cluster_image, cluster_labels, profile, gdf)
    map_wc = generate_worldcover_map(worldcover_image, profile, gdf)
    
    export_results(stats_df, map_cluster, map_wc, cluster_stats_df)

if __name__ == "__main__":
    main()
