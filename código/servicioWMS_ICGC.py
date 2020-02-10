import rasterio as rio
from sentinelsat import SentinelAPI,read_geojson, geojson_to_wkt
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
from rasterio.mask import mask
from shapely.geometry import LineString, shape
import geopandas as gpd
import folium
from owslib.wms import WebMapService
from osgeo import gdal
from PIL import Image 

wms_icgc = WebMapService('http://geoserveis.icgc.cat/icc_ortohistorica/wms/service', version='1.1.1')
wms_pnoa = WebMapService('http://www.ign.es/wms-inspire/pnoa-ma', version='1.1.1')

def peticionICGC_divisiones(pathImagen,numeroTiles,size, tipo,outputPath, tipoFoto='orto25c2016'):
	data = gdal.Open(pathImage)
	geoTransform = data.GetGeoTransform()
	minx = geoTransform[0]
	maxy = geoTransform[3]
	maxx = minx + geoTransform[1] * data.RasterXSize
	miny = maxy + geoTransform[5] * data.RasterYSize

	for x in range(numeroTiles):
		for y in range(numeroTiles):

			try:
				img = wms_icgc.getmap(layers=[tipoFoto],
							 srs='EPSG:32631',
							 bbox=(minx+(abs(maxx-minx)/numeroTiles)*x, maxy-(abs(maxy-miny)/numeroTiles)*(y+1), minx+(abs(maxx-minx)/numeroTiles)*(x+1), maxy-(abs(maxy-miny)/numeroTiles)*(y)),
							 size=(size, size),
							 format='image/'+tipo
							 )
				out = open( outputPath+pathImage[:-3]+'_'+str(x)+'_'+str(y)+tipo, 'wb')
				out.write(img.read())
				out.close()
			except:
				print("Error_"+i[1:-4]+'_'+str(x)+'_'+str(y))
				pass 
				
def peticionICGC(pathImagen,size, tipo,outputPath, tipoFoto):
	data = gdal.Open(pathImage)
	geoTransform = data.GetGeoTransform()
	minx = geoTransform[0]
	maxy = geoTransform[3]
	maxx = minx + geoTransform[1] * data.RasterXSize
	miny = maxy + geoTransform[5] * data.RasterYSize
	try:
		img = wms_icgc.getmap(layers=[tipoFoto],
					 srs='EPSG:32631',
					 bbox=(minx+(abs(maxx-minx)/numeroTiles)*x, maxy-(abs(maxy-miny)/numeroTiles)*(y+1), minx+(abs(maxx-minx)/numeroTiles)*(x+1), maxy-(abs(maxy-miny)/numeroTiles)*(y)),
					 size=(sizex, sizey),
					 format='image/'+tipo
					 )
		out = open( outputPath+pathImage[:-3]+'_'+str(x)+'_'+str(y)+tipo, 'wb')
		out.write(img.read())
		out.close()
	except:
		print("Error_"+i[1:-4]+'_'+str(x)+'_'+str(y))
		pass 


parser = argparse.ArgumentParser()
parser.add_argument("--imageInput", type=str)
parser.add_argument("--pathOutput", type=str)
parser.add_argument("--size", type=int,default=1024)
parser.add_argument("--tipo", type=str,default='tif')
parser.add_argument("--tipofoto", type=str,default='orto25c2016')
args = parser.parse_args()
peticionICGC(args.imageInput,args.size, args.tipo, args.pathOutput, args.tipofoto)







