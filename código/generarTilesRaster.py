import os, gdal, argparse

def generateRasterTiles(pathImageRaster,pathOutputFile,sizeTileX=32,sizeTileY=32):
	"""
	Recibe un raster de entrada 
	genera las tiles mateniendo sus coordenandas geograficas 
	"""
	ds = gdal.Open(pathImageRaster)
	band = ds.GetRasterBand(1)
	xsize = band.XSize
	ysize = band.YSize
	for i in range(0, xsize, sizeTileX):
		for j in range(0, ysize, sizeTileY):
			com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(
				sizeTileX) + ", " + str(sizeTileY) + " " + str(pathImageRaster) + " " + str(
				pathOutputFile) + "tile" + str(i) + "_" + str(j) + ".tif"
			os.system(com_string)
			
parser = argparse.ArgumentParser()
parser.add_argument("--imageInput", type=str)
parser.add_argument("--pathOutput", type=str)
args = parser.parse_args()
generateRasterTiles(args.imageInput,args.pathOutput)