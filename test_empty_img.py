import ee
ee.Initialize()
geom = ee.Geometry.Point([0, 0]).buffer(100)
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate('2099-01-01', '2099-01-02')

month_ndvi_img = s2.map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')).mean()

# Check if image has bands
has_bands = month_ndvi_img.bandNames().size().gt(0)

# Only run .gte() if it has bands
tree_mask_proxy = ee.Algorithms.If(
    has_bands,
    month_ndvi_img.gte(0.6),
    ee.Image(0).rename('NDVI')
)

tree_area_img_proxy = ee.Image.pixelArea().updateMask(ee.Image(tree_mask_proxy))

m_tree_area = tree_area_img_proxy.reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=geom,
    scale=1000,
    maxPixels=1e9
)

safe_val = ee.Algorithms.If(m_tree_area.contains('area'), m_tree_area.get('area'), 0)
print(safe_val.getInfo())
