import cv2
import os
import zipfile
import numpy as np
import argparse
FILENAMES = ['seg_calibrate2-P01.068.TIF', 'seg_calibrate2-P01.072.TIF']
BASE_SHAPE = (512,640)
VALID_LABELS = np.array([0,1,2])
zipfilename = os.path.join('.', 'Outputs', 'TestOutput', 'Outputs.zip')

def check_submition(zipfilename, verbose=False):
    all_ok = True
    exist_file = os.path.exists(zipfilename)
    c = 1
    if verbose:
        print("Check {}. Zip File Exist: {}".format(c, 'Pass' if exist_file else 'Fail!'))
    c +=1
    try:
        with zipfile.ZipFile(zipfilename,'r') as ziph:
            zipped_files = ziph.namelist()
            for file in FILENAMES:
                ok = 'Pass' if file in zipped_files else 'Fail'
                all_ok = all_ok and ok == 'Pass'
                if verbose:
                    print('Check {}. Seg Image {} Exists in Zip File: {}!'.format(c,file, ok))
                c += 1
                seg_bytes = ziph.read(file)
                nparr = np.fromstring(seg_bytes, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                if img_np.shape == BASE_SHAPE:
                    valid_string = 'Pass!'
                else:
                    valid_string = 'Fail! Expected {} and got {}'.format(BASE_SHAPE, img_np.shape)
                    all_ok = False
                if verbose:
                    print('Check {}. Seg Image {} size: {}'.format(c, file, valid_string))
                c+=1
                if img_np.dtype == np.uint8:
                    valid_string = 'Pass!'
                else:
                    valid_string = 'Fail! Expected {} and got {}'.format(np.uint8.__name__, img_np.dtype)
                    all_ok = False

                if verbose:
                    print('Check {}. Seg Image {} Type: {}'.format(c, file, valid_string))
                c += 1
                image_vals = np.unique(img_np)
                valid_values = np.isin(image_vals, VALID_LABELS)

                if np.all(valid_values):
                    valid_string = 'Pass!'
                else:

                    valid_string = 'Fail! {} not in {}'. format(image_vals[np.logical_not(valid_values)], VALID_LABELS)
                    all_ok = False
                if verbose:
                    print('Check {}. Seg Image {} Values: {}'.format(c, file, valid_string))
                c += 1

    except:
        if verbose:
            print('Read Zip File Failed!')

    if verbose:
        if all_ok:

            print('Done! File ready for submmision')
        else:
            print('SUBMISSION CHECK FAILED! View error log')
    return all_ok

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check Submission File')
    parser.add_argument('filepath', metavar='FilePath',
                        help='The full path to the zip file containing the segmentation images. Zip should contain: {}'.format(FILENAMES))
    args = parser.parse_args()
    print('Checking file: {}'.format(args.filepath))
    check_submition(zipfilename=args.filepath, verbose=True)


