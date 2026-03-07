import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import datetime
import urllib.request
import base64
import ssl

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


@app.post("/api/fire-risk")
async def get_fire_risk(request: FireRiskRequest):
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
                
        return {
            "district": request.district,
            "state": request.state,
            "high_risk_zones": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vegetation-map")
async def get_vegetation_map(request: VegetationMapRequest):
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
        boundary_feature = {
            "type": "Feature",
            "properties": {"name": f"{request.district}, {request.state}"},
            "geometry": geom.getInfo()
        }
        
        # 2. Bounding Box and Center
        bounds_coords = geom.bounds().getInfo()['coordinates'][0]
        lon_min = min([p[0] for p in bounds_coords])
        lon_max = max([p[0] for p in bounds_coords])
        lat_min = min([p[1] for p in bounds_coords])
        lat_max = max([p[1] for p in bounds_coords])
        
        center_coords = geom.centroid().getInfo()['coordinates'] # [lon, lat]
        
        # 3. Sentinel-2 Clustering (Age/Health Proxy)
        # Pinning to 2023 to ensure sufficient cloud-free data for training
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(geom) \
            .filterDate('2023-01-01', '2023-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
            .median() \
            .clip(geom)
            
        training = s2.sample(
            region=geom,
            scale=500,
            numPixels=1000
        )
        clusterer = ee.Clusterer.wekaKMeans(5).train(training)
        cluster_image = s2.cluster(clusterer)
        
        cluster_thumb_url = cluster_image.getThumbURL({
            'min': 0, 'max': 4,
            'palette': ['fde725', '5dc863', '21908c', '3b528b', '440154'],
            'region': geom,
            'dimensions': 512,
            'format': 'png'
        })
        
        # 4. ESA WorldCover (Vegetation)
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(geom)
        veg_mask = worldcover.eq(10).Or(worldcover.eq(20)).Or(worldcover.eq(30))
        veg_layer = worldcover.updateMask(veg_mask)
        
        wc_thumb_url = veg_layer.getThumbURL({
            'min': 10, 'max': 30,
            'palette': ['006400', 'FFBB22', 'FFFF4C'], 
            'region': geom,
            'dimensions': 512,
            'format': 'png'
        })
        
        # 5. Fetch and encode images
        def fetch_b64(url):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx) as response:
                return "data:image/png;base64," + base64.b64encode(response.read()).decode('utf-8')
                
        cluster_b64 = fetch_b64(cluster_thumb_url)
        wc_b64 = fetch_b64(wc_thumb_url)
        
        # 6. Construct Response
        return {
            "maps": {
                "cluster_map": {
                    "image_data": cluster_b64,
                    "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
                    "opacity": 1.0,
                    "legend": [
                        { "color": "#fde725", "label": "C0" },
                        { "color": "#5dc863", "label": "C1" },
                        { "color": "#21908c", "label": "C2" },
                        { "color": "#3b528b", "label": "C3" },
                        { "color": "#440154", "label": "C4" }
                    ]
                },
                "worldcover_map": {
                    "image_data": wc_b64,
                    "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
                    "opacity": 0.8,
                    "legend": [
                        { "color": "rgb(0,100,0)", "label": "Tree Cover" },
                        { "color": "rgb(255,187,34)", "label": "Shrubland" },
                        { "color": "rgb(255,255,76)", "label": "Grassland" }
                    ]
                }
            },
            "district_boundary": {
                "type": "FeatureCollection",
                "features": [boundary_feature]
            },
            "center": [center_coords[1], center_coords[0]]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        if hasattr(e, 'read'):
            print("HTTP Error Body:", e.read().decode('utf-8'))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
