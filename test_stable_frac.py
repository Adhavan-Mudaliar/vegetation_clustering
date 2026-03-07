import ee
ee.Initialize(project='forest-monitoring-hackathon')
geom = ee.Geometry.Point([73.7, 20.8]).buffer(10000) # Near The Dangs

# Simulate a mostly cloudy month
s2_month = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate('2023-08-01', '2023-09-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

month_ndvi_img = s2_month.map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')).mean()

has_bands = month_ndvi_img.bandNames().size().gt(0)

tree_mask_proxy = ee.Image(ee.Algorithms.If(
    has_bands,
    month_ndvi_img.gte(0.5),
    ee.Image(0).rename('NDVI')
))

# Mean reducer on the mask itself gives the FRACTION of unmasked area that is 1!
m_tree_frac_dict = tree_mask_proxy.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=geom,
    scale=1000,
    maxPixels=1e9
)

print(m_tree_frac_dict.getInfo())
