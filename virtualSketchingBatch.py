# This is for separate operation to stroklize sketch in raster form
from virtual_sketching.test_vectorization import *
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='output/Pipeline_and_vectorize/sketch/', help='The path of sketches in raster form')
    parser.add_argument('--outpath', type=str, default='output/Pipeline_and_vectorize/', help='The path of output')
    parser.add_argument('--cuda', type=str, default='False')
    parser.add_argument('--sz', type=int, default=600, help='resize the sketch image to this size and input to vectorize net')
    args = parser.parse_args()

    if not os.path.exists(args.inpath):
        print('ERROR: Cannot find path of :', args.inpath)
        exit()

    cuda = True if args.cuda == 'True' else False

    # If haven't specify output, same as input
    if args.outpath == '':
        args.outpath = args.inpath

    vector_folder = args.outpath + '/' + 'vector/'
    gif_folder = args.outpath + '/' + 'gif/'

    os.makedirs(vector_folder, exist_ok=True)
    os.makedirs(gif_folder, exist_ok=True)

    name_list = os.listdir(args.inpath)
    for i in range(len(name_list)):
        name = name_list[i]
        # img = cv.imread(args.inpath + name)
        # img2 = gamma(img, 3)
        sketch2vector(args.inpath, args.outpath, name, 1, model_base_dir='checkpoints/snapshot/')
        print('Processing {}/{}'.format(i, len(name_list)))
