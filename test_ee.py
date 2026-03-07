import ee
ee.Initialize()
dict_ee = ee.Dictionary({})
try:
    print(dict_ee.get('NDVI').getInfo())
except Exception as e:
    print("ERROR:", e)

# Test the safe get
safe_val = ee.Algorithms.If(dict_ee.contains('NDVI'), dict_ee.get('NDVI'), 0)
print("SAFE:", safe_val.getInfo())
