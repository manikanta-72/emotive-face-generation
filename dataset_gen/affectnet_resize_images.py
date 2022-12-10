import os
import cv2

print("Python Program to print list the files in a directory.")
rootdir = '/home/mani/git/emotive-face-generation/data/affectNet_data'
newrootdir = '/home/mani/git/emotive-face-generation/data/affectNet_data_processed'


dirs = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
for dir in dirs:
    print(f"Files in the directory: {rootdir+ '/' + dir}")
    if not os.path.exists(rootdir+ '/' + dir):
        os.makedirs(rootdir+ '/' + dir)
    files = os.listdir(rootdir+ '/' + dir)
    files = [f for f in files if os.path.isfile(rootdir+ '/' + dir+'/'+f)] #Filtering only the files.

    for fl in files:
        print(fl)
        img = cv2.imread(rootdir+ '/' + dir+'/'+fl, cv2.IMREAD_UNCHANGED)
    
        print('Original Dimensions : ',img.shape)
        
        dim = (64, 64)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        print('Resized Dimensions : ',resized.shape)
        
        #cv2.imshow("Resized image", resized)
        if not os.path.exists(newrootdir+ '/' + dir):
            os.makedirs(newrootdir+ '/' + dir)
        cv2.imwrite(newrootdir + '/' + dir+ '/' + fl, resized)
