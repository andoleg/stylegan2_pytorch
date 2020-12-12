import argparse
import cv2
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-D", "--dir", help="image dir (input)")
parser.add_argument("-S", "--start", help="Image name: Where to start filtering")
parser.add_argument("-F", "--finish", help="Image name: Where to finish filtering")

args = parser.parse_args()

if not args.dir:
    print("Укажи путь до картинок через --dir")


keys = {
    49: {'key': 1, 'dir': 'black_and_white'},
    50: {'key': 2, 'dir': 'color'},
    48: {'key': 0, 'dir': 'temporary'}

}


def check_dir_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    for k, v in keys.items():
        check_dir_exists(v['dir'])
        print(v)
    print({'key': 6, 'dir': 'unknown (secret feature)'})
    print('space - пропустить\nesc - выйти')
    started = False
    for imagefile in os.listdir(args.dir):
        if imagefile == args.start:
            started = True
        elif imagefile == args.finish:
            break

        if not args.start or started:
            cv2.imshow("", cv2.imread(os.path.join(args.dir, imagefile)))
            k = cv2.waitKey(0)
            if k == 27:  # esc
                break
            elif k in keys:
                shutil.copyfile(os.path.join(args.dir, imagefile), os.path.join(keys[k]['dir'], imagefile))
            elif k == 32: # space
                continue
            elif k == 54:
                print('Danil pidor')


