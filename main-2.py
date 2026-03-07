import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import datetime
import urllib.request
import base64
import ssl
import os
import json

# Attempt to initialize Earth Engine
try:
    ee.Initialize(project='forest-monitoring-hackathon')
except Exception as e:
    print(f"Earth Engine initialization failed with error: {e}")
    print("Please run `earthengine authenticate` in your terminal or check your project setup.")

app = FastAPI(
    title="Forest Fire Risk API",
    description="Calculates forest fire risk based on Sentinel-2 and ERA5 data using Google Earth Engine.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FireRiskRequest(BaseModel):
    district: str
    state: str

class VegetationMapRequest(BaseModel):
    district: str
    state: str

class TreeCountRequest(BaseModel):
    district: str
    state: str

CACHE_DIR = "api_cache_dir"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_response(cache_key):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading cache for {cache_key}: {e}")
    return None

def set_cached_response(cache_key, data):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error writing cache for {cache_key}: {e}")


@app.post("/api/fire-risk")
async def get_fire_risk(request: FireRiskRequest):
    cache_key = f"fire_risk_{request.district}_{request.state}".lower()
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    try:
        # Initialize GEE just in case it wasn't authenticated at startup but was authenticated later
        try:
            ee.Initialize()
        except:
             raise HTTPException(status_code=500, detail="Google Earth Engine is not authenticated. Please run `earthengine authenticate`.")

        # 1. Get region boundary (FAO GAUL dataset)
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
        
        # Filter by district name (ADM2_NAME) and state (ADM1_NAME)
        region = gaul.filter(ee.Filter.And(
            ee.Filter.eq('ADM2_NAME', request.district),
            ee.Filter.eq('ADM1_NAME', request.state)
        )).first()
        
        # Check if region exists
        info = region.getInfo()
        if not info:
            raise HTTPException(status_code=404, detail=f"District '{request.district}' or State '{request.state}' not found in GAUL dataset.")
        
        geom = region.geometry()

        # 2. Get recent Sentinel-2 image (last 30 days)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)
        
        # Sentinel-2 Surface Reflectance
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(geom) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .median() \
            .clip(geom)

        s2_info = s2.bandNames().getInfo()
        if not s2_info:
            raise HTTPException(status_code=404, detail="No Sentinel-2 imagery found for this region in the last 30 days.")

        # 3. Compute Indices
        # NDVI = (B8(NIR) - B4(Red)) / (B8 + B4)
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # NDMI = (B8(NIR) - B11(SWIR1)) / (B8 + B11)
        ndmi = s2.normalizedDifference(['B8', 'B11']).rename('NDMI')
        
        # NBR = (B8(NIR) - B12(SWIR2)) / (B8 + B12)
        nbr = s2.normalizedDifference(['B8', 'B12']).rename('NBR')
        
        # 4. Get Weather data (ERA5 Daily) for temperature and humidity proxy
        weather = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterBounds(geom) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .median() \
            .clip(geom)
        
        # Temperature at 2m (convert Kelvin to Celsius)
        temperature = weather.select('temperature_2m').subtract(273.15).rename('Temperature')
        
        # Estimate Relative Humidity using temperature and dewpoint 
        # Approx formula: RH ≈ 100 - 5 * (T - Td)
        dewpoint = weather.select('dewpoint_temperature_2m').subtract(273.15)
        humidity = ee.Image(100).subtract(temperature.subtract(dewpoint).multiply(5)).clamp(0, 100).rename('Humidity')
        
        # Normalize variables (0 to 1 scale)
        # Temperature Norm (assuming max 45C, min 10C)
        temp_norm = temperature.subtract(10).divide(35).clamp(0, 1).rename('TempNorm')
        
        # Dryness = 1 - NDMI
        # NDMI is typically -1 to 1. 1 - NDMI clamped to 0-1
        dryness = ee.Image(1).subtract(ndmi).clamp(0, 1).rename('Dryness')
        
        # Vegetation Fuel = NDVI clamped to 0-1
        veg_fuel = ndvi.clamp(0, 1).rename('VegFuel')
        
        # Low Humidity Risk = 1 - (humidity / 100)
        low_humidity = ee.Image(1).subtract(humidity.divide(100)).clamp(0, 1).rename('LowHumid')
        
        # 5. Fire Risk Formula based on the model:
        # FireRisk = 0.35 * Dryness + 0.25 * VegFuel + 0.25 * TempNorm + 0.15 * LowHumid
        fire_risk = dryness.multiply(0.35) \
            .add(veg_fuel.multiply(0.25)) \
            .add(temp_norm.multiply(0.25)) \
            .add(low_humidity.multiply(0.15)) \
            .rename('FireRisk')
            
        # 6. Extract top 5 high-risk coordinates
        # To get spatial coordinates associated with the Risk
        latlon = ee.Image.pixelLonLat()
        risk_with_coords = fire_risk.addBands(latlon)
        
        # Sample points at 1000m scale inside the district boundary
        samples = risk_with_coords.sample(
            region=geom,
            scale=1000,
            numPixels=10000,  # limit to max 10k pixels to prevent memory crash
            geometries=False
        )
        
        # Sort samples to get the top 5 highest fire risks
        top_5_feature_collection = samples.sort('FireRisk', False).limit(5)
        top_5 = top_5_feature_collection.getInfo()
        
        results = []
        if 'features' in top_5:
            for feature in top_5['features']:
                props = feature['properties']
                results.append({
                    "lat": round(props.get('latitude', 0), 4),
                    "lon": round(props.get('longitude', 0), 4),
                    "risk": round(props.get('FireRisk', 0), 4)
                })
                
        # 7. Compute Monthly NDVI History (Past 6 Months) in one EE call
        ee_months = []
        months_keys = []
        for i in range(6):
            m_end = end_date - datetime.timedelta(days=i*30)
            m_start = m_end - datetime.timedelta(days=30)
            
            s2_month = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterBounds(geom) \
                .filterDate(m_start.strftime('%Y-%m-%d'), m_end.strftime('%Y-%m-%d')) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                
            def calculate_ndvi(img):
                return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                
            month_ndvi_img = s2_month.map(calculate_ndvi).mean()
            m_ndvi_dict = month_ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=1000,
                maxPixels=1e9
            )
            
            # If the entire region was masked (e.g., cloudy), NDVI will not be present in dict.
            # Handle this gracefully.
            m_ndvi_safe = ee.Algorithms.If(m_ndvi_dict.contains('NDVI'), m_ndvi_dict.get('NDVI'), 0)
            ee_months.append(m_ndvi_safe)
            months_keys.append(f"month_{6-i}") # month_6 is current month
            
        monthly_stats = ee.Dictionary.fromLists(months_keys, ee_months).getInfo()
        
        monthly_ndvi_history = {}
        for k in months_keys:
            val = monthly_stats.get(k)
            monthly_ndvi_history[k] = round(val, 4) if val is not None else 0
            
        # Fix 0s by filling them with the closest valid month's value (back-fill then forward-fill)
        valid_vals = [v for v in monthly_ndvi_history.values() if v > 0]
        fallback_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0
        
        for k, v in monthly_ndvi_history.items():
            if v == 0:
                # Find the nearest valid neighbors by interpolating, or just use average
                monthly_ndvi_history[k] = round(fallback_val, 4)

        # Calculate the overall 6-month average from the monthly values
        avg_ndvi_6_months = fallback_val

        result = {
            "district": request.district,
            "state": request.state,
            "high_risk_zones": results,
            "avg_ndvi_6_months": round(avg_ndvi_6_months, 4),
            "monthly_ndvi_history": monthly_ndvi_history
        }
        set_cached_response(cache_key, result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vegetation-map")
async def get_vegetation_map(request: VegetationMapRequest):
    cache_key = f"veg_map_{request.district}_{request.state}".lower()
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    try:
        try:
            ee.Initialize()
        except:
             raise HTTPException(status_code=500, detail="Google Earth Engine is not authenticated. Please run `earthengine authenticate`.")

        gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
        region = gaul.filter(ee.Filter.And(
            ee.Filter.eq('ADM2_NAME', request.district),
            ee.Filter.eq('ADM1_NAME', request.state)
        )).first()
        
        info = region.getInfo()
        if not info:
            raise HTTPException(status_code=404, detail=f"District '{request.district}' or State '{request.state}' not found in GAUL dataset.")
        
        geom = region.geometry()
        
        # 1. District boundary feature
        boundary_info = region.getInfo()
        boundary_feature = {
            "type": "Feature",
            "properties": {"name": f"{request.district}, {request.state}"},
            "geometry": boundary_info['geometry']
        }
        
        # 2. Bounding Box and Center
        bounds_coords = geom.bounds().getInfo()['coordinates'][0]
        lon_min = min([p[0] for p in bounds_coords])
        lon_max = max([p[0] for p in bounds_coords])
        lat_min = min([p[1] for p in bounds_coords])
        lat_max = max([p[1] for p in bounds_coords])
        
        center_coords = geom.centroid().getInfo()['coordinates'] # [lon, lat]
        
        # 3. Sentinel-2 NDVI Classification (Health Focus)
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)
        s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(geom) \
            .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .median() \
            .clip(geom)
            
        # Calculate NDVI: (B8 - B4) / (B8 + B4)
        ndvi = s2_col.normalizedDifference(['B8', 'B4']).rename('NDVI')
            
        # Use standard where thresholds instead of expression string
        cluster_image = ee.Image(4) \
            .where(ndvi.lt(0.60), 3) \
            .where(ndvi.lt(0.40), 2) \
            .where(ndvi.lt(0.25), 1) \
            .where(ndvi.lt(0.15), 0) \
            .clip(geom).rename('cluster')
        
        # Palette: Red (Poor) to Green (Excellent)
        health_palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60'] # Red to Dark Green
        
        cluster_thumb_url = cluster_image.getThumbURL({
            'min': 0, 'max': 4,
            'palette': health_palette,
            'region': geom,
            'dimensions': 500, # Fixed 500x500
            'format': 'png'
        })
        
        # 4. Fetch and encode image
        def fetch_b64(url):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as response:
                return "data:image/png;base64," + base64.b64encode(response.read()).decode('utf-8')
                
        cluster_b64 = fetch_b64(cluster_thumb_url)
        
        # 4.5 & 4.7 Compute Stats server-side in one network call
        avg_ndvi_ee_dict = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=500,
            maxPixels=1e9
        )
        avg_ndvi_safe = ee.Algorithms.If(avg_ndvi_ee_dict.contains('NDVI'), avg_ndvi_ee_dict.get('NDVI'), 0)
        
        history_end_date = datetime.datetime.now()
        ee_months = []
        months_keys = []
        for i in range(6):
            m_end = history_end_date - datetime.timedelta(days=i*30)
            m_start = m_end - datetime.timedelta(days=30)
            s2_month = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterBounds(geom) \
                .filterDate(m_start.strftime('%Y-%m-%d'), m_end.strftime('%Y-%m-%d')) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            month_ndvi_img = s2_month.map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')).mean()
            m_ndvi_dict = month_ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=1000,
                maxPixels=1e9
            )
            
            # If the entire region was masked (e.g., cloudy), NDVI will not be present in dict.
            # Handle this gracefully.
            m_ndvi_safe = ee.Algorithms.If(m_ndvi_dict.contains('NDVI'), m_ndvi_dict.get('NDVI'), 0)
            ee_months.append(m_ndvi_safe)
            months_keys.append(f"month_{6-i}") # month_6 is current
            
        dict_ee = ee.Dictionary.fromLists(months_keys, ee_months).set('avg_ndvi', avg_ndvi_safe)
        all_stats = dict_ee.getInfo()
        
        avg_ndvi = all_stats.get('avg_ndvi')
        if avg_ndvi is None:
            avg_ndvi = 0
            
        monthly_ndvi_history = {}
        for k in months_keys:
            val = all_stats.get(k)
            monthly_ndvi_history[k] = round(val, 4) if val is not None else 0
            
        # Fix 0s by filling them with the overall average NDVI for the district
        for k, v in monthly_ndvi_history.items():
            if v == 0:
                monthly_ndvi_history[k] = round(avg_ndvi, 4)
        
        # 5. Construct Response
        result = {
            "avg_ndvi": round(avg_ndvi, 4) if avg_ndvi is not None else 0,
            "monthly_ndvi_history": monthly_ndvi_history,
            "maps": {
                "health_map": {
                    "image_data": cluster_b64,
                    "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
                    "opacity": 1.0,
                    "legend": [
                        { "color": "#e74c3c", "label": "Very Poor Health" },
                        { "color": "#e67e22", "label": "Poor Health" },
                        { "color": "#f1c40f", "label": "Average Health" },
                        { "color": "#2ecc71", "label": "Good Health" },
                        { "color": "#27ae60", "label": "Excellent Health" }
                    ]
                }
            },
            "district_boundary": {
                "type": "FeatureCollection",
                "features": [boundary_feature]
            },
            "center": [center_coords[1], center_coords[0]]
        }
        set_cached_response(cache_key, result)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        if hasattr(e, 'read'):
            print("HTTP Error Body:", e.read().decode('utf-8'))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tree-count")
async def get_tree_count(request: TreeCountRequest):
    cache_key = f"tree_count_{request.district}_{request.state}".lower()
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    try:
        try:
            ee.Initialize()
        except:
             raise HTTPException(status_code=500, detail="Google Earth Engine is not authenticated. Please run `earthengine authenticate`.")

        gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
        region = gaul.filter(ee.Filter.And(
            ee.Filter.eq('ADM2_NAME', request.district),
            ee.Filter.eq('ADM1_NAME', request.state)
        )).first()
        
        info = region.getInfo()
        if not info:
            raise HTTPException(status_code=404, detail=f"District '{request.district}' or State '{request.state}' not found in GAUL dataset.")
        
        geom = region.geometry()
        
        # Load ESA WorldCover v200
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(geom)
        
        # Class 10 is 'Trees'
        tree_mask = worldcover.eq(10).selfMask()
        
        # Calculate tree coverage area using pixelArea()
        # pixelArea() returns the area of each pixel in square meters
        tree_area_img = ee.Image.pixelArea().updateMask(tree_mask)
        
        # Sum the area (m2) and count the pixels
        stats = tree_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Count the pixels using sum() on the binary mask (1 for trees, 0 otherwise)
        tree_binary = worldcover.eq(10).rename('pixels')
        pixel_count_stats = tree_binary.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Total area of the district
        total_area_m2 = geom.area().getInfo()
        
        tree_area_m2 = stats.get('area', 0)
        # The key for the sum will be the band name 'pixels'
        tree_pixel_count = pixel_count_stats.get('pixels', 0)
        
        # Convert to sqkm
        tree_area_sqkm = round(tree_area_m2 / 1000000, 2)
        total_area_sqkm = round(total_area_m2 / 1000000, 2)
        
        # Calculate percentage
        coverage_percentage = round((tree_area_m2 / total_area_m2) * 100, 2) if total_area_m2 > 0 else 0
        
        # Estimate tree population (Assuming 50,000 trees per sq km)
        estimated_tree_population = int(tree_area_sqkm * 42500)
        
        result = {
            "district": request.district,
            "state": request.state,
            "tree_metrics": {
                "tree_area_sqkm": tree_area_sqkm,
                "total_district_area_sqkm": total_area_sqkm,
                "tree_coverage_percentage": coverage_percentage,
                "estimated_tree_population": estimated_tree_population
            }
        }
        set_cached_response(cache_key, result)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)