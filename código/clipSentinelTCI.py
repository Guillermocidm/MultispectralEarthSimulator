"""
Este fichero utiliza la API de SentinelSat, es necessario tener usuario y contraseña de https://scihub.copernicus.eu/dhus
Recibe un archivo de tipo shp con las coordenadas deseadas, descarga imagenes de Sentinel-2 de esa zona 
Genera la True-Color-Image y la clipea segun tu archivo deseado
El acceso a los ficheros es para descargas con procesado L2A, en el caso de L1C deberia utilizar software Sen2cor para 
convertirlo a L2A
"""
from osgeo import gdal, gdal_array
import rasterio as rio
from sentinelsat import SentinelAPI,read_geojson, geojson_to_wkt
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from rasterio.mask import mask
import geopandas as gpd
import folium
import glob
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str)
parser.add_argument("--password", type=str)
parser.add_argument("--pathShapeFile", type=str)
args = parser.parse_args()



api = SentinelAPI(args.user, args.password, 'https://scihub.copernicus.eu/dhus')
shapeFile = gpd.read_file(args.pathShapeFile)
footprint = None
for i in shapeFile['geometry']:
    footprint = i
"""
En el caso de no funcionar con shp file utilizar un gejson de la misma region
"""
#footprint = geojson_to_wkt(read_geojson(pathGeojson))
products = api.query(footprint,
                     date = ('20170702','20190920'),
                     platformname = 'Sentinel-2',
                     processinglevel = 'Level-2A',
                     cloudcoverpercentage = (0,10)
                    )
products_gdf = api.to_geodataframe(products)
products_gdf_sorted = products_gdf.sort_values(['cloudcoverpercentage'], ascending=[True])



shapeFile_wgt = shapeFile.to_crs(epsg=4326)

number_download = 0
max_downloads = 4
dict_percentage = {}

for i in range(len(products_gdf_sorted)):
    actual_polygon = products_gdf_sorted[i:i+1]
    points_actual = actual_polygon.total_bounds
    points_searched = shapeFile_wgt.total_bounds
    if (points_actual[0] < points_searched[0] and points_actual[1] < points_searched[1] and points_actual[2] > points_searched[2] and points_actual[3] > points_searched[3]):
        api.download(actual_polygon['uuid'].item())
        with zipfile.ZipFile('./'+actual_polygon['identifier'].item()+'.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
        path_to_name = './'+actual_polygon['filename'].item() + '/GRANULE/'
        name_files = os.listdir(path_to_name)
        name_files = name_files[0]
        path_to_files = path_to_name + name_files + '/IMG_DATA/R10m/'
        R = glob.glob(path_to_files+"*B04*")[0]
        G = glob.glob(path_to_files+"*B03*")[0]
        B = glob.glob(path_to_files+"*B02*")[0]
        gdal.BuildVRT(path_to_files+'TCI.vrt', [R,G,B],separate=True)
   

        #os.system("gdal_translate -ot Byte -co TILED=YES -scale 0 4096 0 255 "+path_to_files+"TCI.vrt "+path_to_files+"TCI.tif")
        gdal.Translate(path_to_files + 'TCI.tif', gdal.Open(path_to_files + "TCI.vrt"), scaleParams=[[0, 4096, 0, 255]],
                       outputType=gdal.GDT_Byte, creationOptions=['TILED=YES'])
        #gdal_translate -ot Byte -co TILED=YES -scale 0 4096 0 255 TCI.vrt RGB.tif


        ds = None
        
        with rio.open(path_to_files + 'TCI.tif') as src:
            out_image, out_transform = mask(src, shapeFile.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
            path_cliped = './image_clipped/'
        if not os.path.isdir(path_cliped):
            os.mkdir(path_cliped)
        with rio.open(path_cliped + "RGB_masked_"+str(number_download+1)+".tif", "w", **out_meta) as dest:
            dest.write(out_image)

        im_gdal = gdal.Open(path_cliped+"RGB_masked_"+str(number_download+1)+".tif")

        im = im_gdal.ReadAsArray()
        im = np.transpose(im,(1,2,0))
        black_white = np.any(im!=0,axis=2)
        unique,counts = np.unique(black_white,return_counts=True)
        dict_percentage["RGB_masked_"+str(number_download+1)]=counts[0]/(counts[0]+counts[1])
        number_download+=1
    if number_download==max_downloads:
        break







