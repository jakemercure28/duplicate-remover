import sys
import os
import argparse
import cv2 as cv
import time
import concurrent.futures
import shutil


FEATURES_DISTANCE = 0.3
MIN_MATCHES = 50 #matches found when comparing two images together
MAX_KEYPOINTS = 100 #total keypoints on image
scale_percent = 12 #reduces image resolution to speed up comparison

files = []
imgs = []
features = []
kp = []
des = []
duplicates = []
itr = []

def get_file_list(directory):

		count = 0
		for file in os.listdir(directory): #reads every file in the directory given, only takes the formats listed below
			if(file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))):
				try:
					path = os.path.join(directory, file)
					files.append(path)
								
				except:
					print("[FAILURE OPENING FILE]", path)
							
				count+=1
		print('[ADDING FILES]', count)        

def compute_image(file):
	
	try:
		img = cv.imread(file, cv.IMREAD_GRAYSCALE) #reads individual image in grayscale
		h1, w1 = img.shape[:2] # notes the height and width of individual image
				
		if(h1 >= 800 and w1 >= 800): #reduce resolution if too large
			width = int(img.shape[1] * scale_percent / 100)
			height = int(img.shape[0] * scale_percent / 100)
			dim = (width, height)
			img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

		sift = cv.SIFT_create(MAX_KEYPOINTS) #mark keypoint on image
		k,d = sift.detectAndCompute(img, None)
	except:
		print('[FAILED TO READ IMAGE]', file)
		return 0
	

	return d

def similarity_check(d, file,i):
	temp = []
	for data in range(0 + i, len(files)):

		FLANN_INDEX_KDTREE = 1
		index_params = dict(
			algorithm = FLANN_INDEX_KDTREE,
			trees = 5
		)

		search_params = dict(checks=50)
		flann = cv.FlannBasedMatcher(index_params, search_params) #comparison (O-1/2)
		matches = flann.knnMatch(d, des[data], k=2)
		matchesCount = 0
		for i,(m,n) in enumerate(matches):
			if m.distance < FEATURES_DISTANCE * n.distance:
				matchesCount += 1
					
				itr[data] = None
			if(matchesCount > MIN_MATCHES):
				temp.append(files[data])
							

			else:
				temp.append(0)                                
				# adds the lower resolution image to the deletion list
				# h1, w1 = img.shape[:2]
				# h2, w2 = imgs[i2].shape[:2]
								
					
	return temp
				

def delete(duplicates):
		for path in duplicates:
				try:
					shutil.move(f, 'duplicates')
				except:
					
					try:
						os.remove(path)
						print('[DELETED]', path)
					except:
						continue

def argparser():
	
		parser = argparse.ArgumentParser()
		parser.add_argument("directory", type=str,
				help="directory with the images")
		parser.add_argument("-d", "--delete", action='store_true',
				help="delete the duplicate images found with smaller res")
		parser.add_argument("-s", "--silent", action='store_true',
				help="quiet execution without logging")
		parser.add_argument("-min", '--min_matches', type=int,
				help="minimum number of matching features to accept the images as being similar")
		parser.add_argument("-f", '--features_distance', type=float,
				help="[0,1] - higher number results in more matching features but with less accuracy")
		parser.add_argument("-max", '--max_keypoints', type=int,
				help="max keypoints to mark on each photo when scanning for similarity")
		args = parser.parse_args()

		if(args.silent):
				sys.stdout = open(os.devnull, 'a')
		if(args.min_matches):
				MIN_MATCHES = args.min_matches
		if(args.features_distance):
				FEATURES_DISTANCE = args.features_distance
		if(args.max_keypoints):
				MAX_KEYPOINTS = args.max_keypoints

		return args


def main():
	
		start_time = time.time()
		
		args = argparser()

		get_file_list(args.directory)


		with concurrent.futures.ProcessPoolExecutor() as executor: #reads images using all cpu threads
			results = executor.map(compute_image, files)

		for result in results:
			des.append(result)


		count = 0
		for item in range(0, len(files)):
			count += 1
			itr.append(count)

		with concurrent.futures.ProcessPoolExecutor() as executor:
				results = executor.map(similarity_check, des, files, itr)

		for result in results:
			for item in result:
				if(item != 0):
					duplicates.append(item)
	  
		if args.delete:
				delete(duplicates)

				
		print("--- %.8s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":

	main()
