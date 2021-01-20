import os
import time
from autotrack import AutoTrack
from arcf import ARCF
import glob
import cv2
import numpy as np

def get_init_gt(anno_path):
    gt_path = os.path.join(anno_path)
    with open(gt_path, 'r') as f:
        line = f.readline()
        if ',' in line:
            gt_pos = line.split(',')
        else:
            gt_pos=line.split()
        gt_pos_int=[int(float(element)) for element in gt_pos]
    return tuple(gt_pos_int)

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list

def get_ground_truthes(anno_path):
    gt_path = os.path.join(anno_path)
    gts=[]
    with open(gt_path, 'r') as f:
        while True:
            line = f.readline()
            if line=='':
                gts=np.array(gts,dtype=np.float32)
                return gts
            if ',' in line:
                gt_pos = line.split(',')
            else:
                gt_pos=line.split()
            gt_pos_int=[(float(element)) for element in gt_pos]
            gts.append(gt_pos_int)


def main(visulization=False,track=AutoTrack):
    data_dir='/home/v4r/Dataset/UAVDT/data_seq'
    anno_dir='/home/v4r/Dataset/UAVDT/anno'
    data_names=sorted(os.listdir(data_dir))
    print(data_names)
    tracker = track()

    # setup experiments
    video_paths = sorted(glob.glob(os.path.join(data_dir, '*')))
    video_num = len(video_paths)
    output_dir = os.path.join('results', tracker.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
#    overall_performance = []
    overall_FPS=[]

    # run tracking experiments and report performance
    for video_id, video_path in enumerate(video_paths, start=1):
        data_name=data_names[video_id-1]
        gt_path=os.path.join(anno_dir,data_name+'_gt.txt')
        frame_list = get_img_list(video_path)
        frame_list.sort()
        init_rect=get_init_gt(gt_path)
        gts=get_ground_truthes(gt_path)
        out_res = []
        ut=0
        for frame_id in range(len(frame_list)):
            frame=cv2.imread(frame_list[frame_id])
            if frame_id == 0:
                s=time.time()
                tracker.init(frame, init_rect)  # initialization
                delta=time.time()-s
                ut+=delta
                # tracker.initialize(frame, init_rect)
                out = init_rect
                out_res.append(init_rect)
            else:
                s=time.time()
                out = tracker.update(frame,frame_id)  # tracking
                delta=time.time()-s
                ut+=delta
                # out = tracker.track(frame)
                out_res.append(out)
            if visulization:
                _gt = gts[frame_id]
#                _exist = label_res['exist'][frame_id]
#                if _exist:
                cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                              (0, 255, 0))
                FPS=str(1/delta)
                cv2.putText(frame, FPS,
                            (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0), 2)

                cv2.rectangle(frame, (int(out[0]), int(out[1])), (int(out[0] + out[2]), int(out[1] + out[3])),
                              (0, 255, 255))
                cv2.imshow(data_name,frame)
                cv2.waitKey(1)
            frame_id += 1
        if visulization:
            cv2.destroyAllWindows()
        overall_FPS.append(len(frame_list)/ut)
        # save result
        output_file = os.path.join(output_dir, '%s.txt' % (data_name))
        with open(output_file, 'w') as f:
            np.savetxt(f,np.array(out_res).astype(np.int16))
#        mixed_measure = eval(out_res, gts)
#        overall_performance.append(mixed_measure)
        print('[%03d/%03d] %20s  FPS: %.03f' % (video_id, video_num, data_name, len(frame_list)/ut))

    print('[Overall] FPS: %.03f\n' % (np.mean(overall_FPS)))


if __name__ == '__main__':
    main(True,AutoTrack)