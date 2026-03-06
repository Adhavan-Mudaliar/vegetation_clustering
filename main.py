import os
import io
import datetime
import numpy as np
import osmnx as ox
import rasterio
from rasterio.mask import mask
from sklearn.cluster import KMeans
import folium
from shapely.geometry import mapping, box
import geopandas as gpd

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
    # We want B2, B3, B4, B8, B11. 
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B02", "B03", "B04", "B08", "B11", "dataMask"],
            output: {
                bands: 5,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        if (sample.dataMask === 0) {
            return [0, 0, 0, 0, 0];
        }
        return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11];
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
    
    # Download data
    data = request.get_data()
    
    if len(data) == 0:
        raise ValueError("No data returned from SentinelHub.")
        
    image_array = data[0] # The median composite
    
    # Output to TIF
    output_filename = "sentinel_median.tif"
    print(f"Saving downloaded image to {output_filename}... dimensions: {image_array.shape}")
    
    # SH Request returns array usually (H, W, Bands)
    bands = image_array.shape[-1]
    height = image_array.shape[0]
    width = image_array.shape[1]
    
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Save the TIFF
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
            # Transpose to (bands, height, width) for saving
            image_array = np.transpose(image_array, (2, 0, 1))
            
        for i in range(bands):
            dst.write(image_array[i], i + 1)
            
    return output_filename

def compute_indices(image_path):
    """
    Compute NDVI, NDWI and return them alongside the raw bands.
    """
    print("Computing vegetation indices (NDVI, NDWI)...")
    with rasterio.open(image_path) as src:
        # Read bands: B2(0), B3(1), B4(2), B8(3), B11(4)
        bands = src.read()
        transform = src.transform
        crs = src.crs
        profile = src.profile

    # SentinelHub Reflectance is already float between 0-1
    b2 = bands[0].astype('float32')
    b3 = bands[1].astype('float32')
    b4 = bands[2].astype('float32')
    b8 = bands[3].astype('float32')
    b11 = bands[4].astype('float32')
    
    # Avoid division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    # NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
    ndvi = np.where((b8 + b4) == 0., 0, (b8 - b4) / (b8 + b4))
    
    # NDWI = (Green - NIR) / (Green + NIR) = (B3 - B8) / (B3 + B8)
    ndwi = np.where((b3 + b8) == 0., 0, (b3 - b8) / (b3 + b8))
    
    return b2, b3, b4, b8, b11, ndvi, ndwi, profile

def prepare_features(b2, b3, b4, b8, b11, ndvi, ndwi):
    """
    Flatten the features into an N x 7 matrix: [B2,B3,B4,B8,B11,NDVI,NDWI].
    Returns the feature matrix and the original shape for reconstruction.
    """
    print("Preparing feature stack for clustering...")
    shape = b2.shape
    
    # Flatten each layer
    b2_flat = b2.flatten()
    b3_flat = b3.flatten()
    b4_flat = b4.flatten()
    b8_flat = b8.flatten()
    b11_flat = b11.flatten()
    ndvi_flat = ndvi.flatten()
    ndwi_flat = ndwi.flatten()
    
    # Stack into columns
    features = np.column_stack([b2_flat, b3_flat, b4_flat, b8_flat, b11_flat, ndvi_flat, ndwi_flat])
    
    # We should exclude NaN values from clustering, but fill them later with a specific value (like -1)
    # Tiff padded areas will have many zeros, we might want to track valid pixels
    # For Sentinel-2 SR, 0 is often no-data over land, but we use nan here
    valid_mask = ~np.isnan(ndvi_flat) & (b2_flat != 0)
    valid_features = features[valid_mask]
    
    return valid_features, valid_mask, shape

def run_clustering(valid_features, valid_mask, shape):
    """
    Execute KMeans clustering with n_clusters=5.
    """
    print("Running KMeans clustering (k=5)...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    
    cluster_labels_valid = kmeans.fit_predict(valid_features)
    
    # Reconstruct the full grid
    full_clusters_flat = np.full(shape[0] * shape[1], fill_value=-1, dtype=np.int32)
    full_clusters_flat[valid_mask] = cluster_labels_valid
    
    cluster_image = full_clusters_flat.reshape(shape)
    return cluster_image

def generate_cluster_map(cluster_image, profile, boundary_gdf):
    """
    Generate an interactive map using folium.
    Outputs the map as map.html.
    """
    print("Generating interactive folium map...")
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Create colormap for 5 clusters (e.g., Greens/YlGn/etc.)
    colors = ['#fde725', '#5dc863', '#21908c', '#3b528b', '#440154'] # Viridis colors for distinct clusters
    cmap = ListedColormap(colors)
    
    # Save the cluster array as a temporary PNG or TIF for mapping, but we can plot straight to folium
    # using ImageOverlay if we colorize it to RGBA
    
    # Mask out the -1 values with transparency
    h, w = cluster_image.shape
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    for i in range(5):
        mask = cluster_image == i
        hex_color = colors[i]
        # hex to rgb
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        rgba_image[mask, 0] = r
        rgba_image[mask, 1] = g
        rgba_image[mask, 2] = b
        rgba_image[mask, 3] = 255 # fully opaque
    
    # Get bounds
    bounds = rasterio.transform.array_bounds(profile['height'], profile['width'], profile['transform'])
    # Output is (minx, miny, maxx, maxy) in EPSG:4326 usually since we download that way, let's verify
    if profile['crs'] and profile['crs'].to_epsg() != 4326:
        # If the image was downloaded in another CRS, we need to handle it.
        # But ee_export_image defaults to 4326 if not specified (or native CRS).
        pass

    # Bounds for folium ImageOverlay: [[lat_min, lon_min], [lat_max, lon_max]]
    lat_min = bounds[1]
    lat_max = bounds[3]
    lon_min = bounds[0]
    lon_max = bounds[2]
    
    image_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    
    # Create folium map centered on the image
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add ImageOverlay
    folium.raster_layers.ImageOverlay(
        image=rgba_image,
        bounds=image_bounds,
        opacity=0.7,
        name='Vegetation Clusters',
        interactive=True,
        cross_origin=False
    ).add_to(m)
    
    # Add the boundary polygon
    folium.GeoJson(
        boundary_gdf,
        name="District Boundary",
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 2}
    ).add_to(m)
    
    # Add a custom categorical legend
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 120px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     padding: 10px;
     ">
     <b>Clusters</b><br>
     <i style="background:#fde725;width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 0<br>
     <i style="background:#5dc863;width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 1<br>
     <i style="background:#21908c;width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 2<br>
     <i style="background:#3b528b;width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 3<br>
     <i style="background:#440154;width:15px;height:15px;float:left;margin-right:5px;"></i> Cluster 4<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl().add_to(m)
    
    m.save("map.html")
    print("Map generated successfully at map.html!")

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
    b2, b3, b4, b8, b11, ndvi, ndwi, profile = compute_indices(image_path)
    
    # 5. Prepare Features
    valid_features, valid_mask, shape = prepare_features(b2, b3, b4, b8, b11, ndvi, ndwi)
    
    # 6. Clustering
    cluster_image = run_clustering(valid_features, valid_mask, shape)
    
    # 7. Map generation
    generate_cluster_map(cluster_image, profile, gdf)

if __name__ == "__main__":
    main()
