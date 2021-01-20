
import argparse
import os
import datetime
from OMR_Solver import OMR


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")
args = parser.parse_args()


OMR(args.inputfolder, args.outputfolder)

#Our code:#
'''
imgs_path = os.listdir(f"{args.inputfolder}")
for path in imgs_path:
    img = io.imread(f"{args.inputfolder}/" + path)
    # Process
    #io.imsave(f"{args.outputfolder}/" + path, img)
    
    #save
    test_number = path.split('.')[0]
    output_file = open(f"{args.outputfolder}/" + test_number + ".txt", "w")
    output_file.write("The file number is: %s" %test_number)
    output_file.close()
    
    print('done!')
'''



'''
with open(f"{args.outputfolder}/Output.txt", "w") as text_file:
    text_file.write("Input Folder: %s" % args.inputfolder)
    text_file.write("Output Folder: %s" % args.outputfolder)
    text_file.write("Date: %s" % datetime.datetime.now())
'''


print('Finished !!') 
