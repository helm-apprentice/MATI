import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import glob
import re


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        self.has_infrared = False

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': [],
                  'fps': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        processed_frames = 1
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            # init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            frame_start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) == 1:
                info['gt_bbox'] = seq.ground_truth_rect[0]
            elif len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track([image], seq.name, info)
            prev_output = OrderedDict(out)

            frame_time = time.time() - frame_start_time
            processed_frames += 1
            current_fps = processed_frames / (time.time() - start_time)
            _store_outputs(out, {'time': frame_time, 'fps': current_fps})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        # elif multiobj_mode == 'parallel':
        #     tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        "videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_image_sequence(self, folder_path, optional_box=None, debug=None, use_visdom=False, save_results=False):
        """
        Run the tracker on an image sequence in a folder.
        args:
            debug: Debug level.
        """
        params = self.get_parameters()
        
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        params.use_visdom = use_visdom

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        # ------------------------------增加红外判断-----------------------------------------------------------------
        infrared_path = os.path.join(folder_path, 'infrared')
        self.has_infrared = os.path.exists(infrared_path)
        print(f"Double mode: {self.has_infrared}")
        params.has_infrared = self.has_infrared
        #------------------------------手动设置是否中间插入cmc------------------------------------
        # params.has_infrared = True
        #------------------------------手动设置是否中间插入cmc------------------------------------
        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        else:
            raise ValueError(f"Unknown multi object mode {multiobj_mode}")

        
        if self.has_infrared:
            visible_path = os.path.join(folder_path, 'visible')
            assert os.path.isdir(folder_path), f"INvalid param {folder_path}, folder_path must be a valid directory"
            assert os.path.isdir(infrared_path), f"INvalid param {infrared_path}, folder_path must be a valid directory"
            image_files = sorted([os.path.join(visible_path, f) for f in os.listdir(visible_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            inf_image_files = sorted([os.path.join(infrared_path, f) for f in os.listdir(infrared_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            assert len(image_files) > 0, "No image files found in the directory"
            assert len(inf_image_files) > 0, "No image files found in the directory"
        else:
            assert os.path.isdir(folder_path), f"INvalid param {folder_path}, folder_path must be a valid directory"
            image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            assert len(image_files) > 0, "No image files found in the directory"

        output_boxes = []

        display_name = 'Display: ' + tracker.params.tracker_name
        inf_display_name = 'Display(Infrared): ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        if self.has_infrared:
            first_frame = cv.imread(image_files[0])
            first_inf_frame = cv.imread(inf_image_files[0])
        else:
            first_frame = cv.imread(image_files[0])
        cv.imshow(display_name, first_frame)

        def _build_init_info(box):
            return {'init_bbox': box, 
                    'inf_init_bbox': []}
        
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"
            tracker.initialize(first_frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                if self.has_infrared:
                    frame_disp = first_frame.copy()
                    inf_frame_disp = first_inf_frame.copy()
                    first_frame = [first_frame, first_inf_frame]
                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                            1.5, (0, 0, 0), 1)
                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    
                    cv.putText(inf_frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                            1.5, (0, 0, 0), 1)
                    # cv.imshow(inf_display_name, inf_frame_disp)
                    ix, iy, iw, ih = cv.selectROI(display_name, inf_frame_disp, fromCenter=False)
                    inf_init_state = [ix, iy, iw, ih]
                    info = {'init_bbox': init_state, 'inf_init_bbox': inf_init_state}
                    tracker.initialize(first_frame, info)
                    #output_boxes.append(init_state)
                    break
                else:
                    frame_disp = first_frame.copy()

                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                            1.5, (0, 0, 0), 1)
                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    info = {'init_bbox': init_state}
                    tracker.initialize(first_frame, info)
                    #output_boxes.append(init_state)
                    break

        pause = False
        start_time = time.time()
        processed_frames = 0
        
        for i in range(len(image_files)): # 从0循环
            frame_start_time = time.time()
            loop_time = frame_start_time - start_time
            # print(f"loop_time: {loop_time}")
            
            infrared_path = os.path.join(folder_path, 'infrared')
            self.has_infrared = os.path.exists(infrared_path)
            
            if not pause:
                if self.has_infrared:
                    frame = [cv.imread(image_files[i]), cv.imread(inf_image_files[i])]
                    
                else:
                    frame = [cv.imread(image_files[i])]
                    
                frame_disp = frame[0].copy()
                if frame is None:
                    break
                read_time = time.time() - frame_start_time
                # print(f"read_time: {read_time}") # read_time: 0.0010120868682861328
                # out = tracker.track([image], seq.name, info)
                out = tracker.track(frame, info=info)
                frame_time = time.time() - frame_start_time
                # print(f"frame_time: {frame_time}") # frame_time: 0.008500099182128906
                processed_frames += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                fps = processed_frames / elapsed_time
                time1 = time.time()
                state = [int(s) for s in out['target_bbox']]
                output_boxes.append(state)
                time2 = time.time()
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)
                font_color = (0, 0, 0)
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, f'FPS: {1/frame_time:.2f}', (300, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv.putText(frame_disp, f'Frame Time: {frame_time*1000:.2f} ms', (300, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                time3 = time.time()
            # print(f"time21: {time2-time1}, time32: {time3-time2}") # time21: 2.1457672119140625e-06, time32: 4.363059997558594e-05
            cv.imshow(display_name, frame_disp)  # show_time: 0.0002453327178955078
            
            time4 = time.time()
            key = cv.waitKey(10)
            """如果延迟时间设置得太低，可能会使得帧显示得太快，难以观察；设置得太高，则会降低实时性。根据具体应用场景和需求来确定最适合的延迟时间。"""
            if key == ord('q'):
                break
            elif key == ord('r'):
                if self.has_infrared:
                    frame_disp = frame[0].copy()
                    inf_frame_disp = frame[1].copy()
                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (0, 0, 0), 1)
                    cv.imshow(display_name, frame_disp)
                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    cv.putText(inf_frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                            1.5, (0, 0, 0), 1)
                    # cv.imshow(inf_display_name, inf_frame_disp)
                    ix, iy, iw, ih = cv.selectROI(display_name, inf_frame_disp, fromCenter=False)
                    inf_init_state = [ix, iy, iw, ih]
                    info = {'init_bbox': init_state, 'inf_init_bbox': inf_init_state}
                    tracker.initialize(frame, info)
                    #print('check')
                    output_boxes.append([-1, -1, -1, -1])
                else:
                    frame_disp = frame[0].copy()

                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (0, 0, 0), 1)
                    cv.imshow(display_name, frame_disp)
                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    info = {'init_bbox': init_state}
                    tracker.initialize(frame[0], info)
                    output_boxes.append(init_state)
            elif key == 32:
                pause = not pause
            time5 = time.time()
            rest_time = time5-time4
            # print(f"rest_time: {rest_time}") # rest_time: 0.018720149993896484
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            sequence_name = Path(folder_path).stem
            base_results_path = os.path.join(self.results_dir, f'sequence_{sequence_name}')
            print(f"base_results_path: {base_results_path}")
            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = f'{base_results_path}.txt'
            np.savetxt(bbox_file, tracked_bb, delimiter=',', fmt='%d')
        

    def run_all_image_sequences(self, data_root, ground_truth_pattern='*.txt', has_infrared=False):
        """
        使用文本文件中列出的所有图像序列运行跟踪器。
        结果将分别保存每个序列，无需进行可视化或交互式目标区域选择。

        参数：
            results_dir (str): 保存跟踪结果的目录。
            data_root (str): 包含图像序列及关联地面实况文件的根目录。
            ground_truth_pattern (str, 可选): 匹配地面实况文件的glob模式。默认为'*.txt'。
        """
        print(f"Double mode(function argument): {has_infrared}")
        # 在根目录下的'list.txt'文件中找到所有序列名称
        list_file_path = os.path.join(data_root, 'list.txt')
        with open(list_file_path, 'r') as f:
            sequence_names = f.read().splitlines()

        for sequence_name in sequence_names:
            sequence_folder = os.path.join(data_root, sequence_name)
            self.run_single_image_sequence(sequence_folder, ground_truth_pattern, has_infrared)
        print("All sequences done.")

    def run_single_image_sequence(self, sequence_folder, ground_truth_pattern='*.txt', has_infrared=False):
        """
        对单个图像序列运行跟踪器，使用地面实况数据初始化跟踪器。
        结果将分别保存，无需进行可视化或交互式目标区域选择。

        参数：
            sequence_folder (str): 包含图像序列及关联地面实况文件的目录。
            ground_truth_pattern (str, 可选): 匹配地面实况文件的glob模式。默认为'*.txt'。
        """

        params = self.get_parameters()
        params.debug = 0   # 因为没在函数参数里设置debug和use_visdom参数，所以在这里手动设置
        params.use_visdom = 1
        tracker = self.create_tracker(params)
        if has_infrared:
            # 查找可见光和红外两种模态的地面实况文件
            visible_gt_files = glob.glob(os.path.join(sequence_folder, 'visible', ground_truth_pattern))
            infrared_gt_files = glob.glob(os.path.join(sequence_folder, 'infrared', ground_truth_pattern))

            assert len(visible_gt_files) == 1, f"{sequence_folder} 中未找到或存在多个可见光地面实况文件"
            assert len(infrared_gt_files) == 1, f"{sequence_folder} 中未找到或存在多个红外地面实况文件"

            visible_gt_path = visible_gt_files[0]
            infrared_gt_path = infrared_gt_files[0]

            # 从地面实况文件加载初始状态
            init_state_visible = self.load_initial_state_from_ground_truth(visible_gt_path)
            init_state_infrared = self.load_initial_state_from_ground_truth(infrared_gt_path)

            # 使用初始状态初始化info字典
            info = {
                'init_bbox': init_state_visible,
                'inf_init_bbox': init_state_infrared
            }

            # ... 其余代码放在这里 ...
            # 将现有的初始化和主跟踪循环逻辑替换为
            # 调用您的跟踪器API，并使用提供的`info`字典，
            # 同时跳过可视化和交互式ROI选择部分。

            visible_path = os.path.join(sequence_folder, 'visible')
            infrared_path = os.path.join(sequence_folder, 'infrared')
            assert os.path.isdir(visible_path), f"INvalid param {visible_path}, visible_path should be a folder"
            assert os.path.isdir(infrared_path), f"INvalid param {infrared_path}, infrared_path should be a folder"
            image_files = sorted([os.path.join(visible_path, f) for f in os.listdir(visible_path) if f.endswith(('.jpg', 'png', '.bmp'))])
            inf_image_files = sorted([os.path.join(infrared_path, f) for f in os.listdir(infrared_path) if f.endswith(('.jpg', 'png', '.bmp'))]) 
            assert len(image_files) > 0, f"No image files found in {visible_path}"
            assert len(inf_image_files) > 0, f"No image files found in {infrared_path}"
            assert len(image_files) == len(inf_image_files), f"Number of images in {visible_path} and {infrared_path} do not match"
            first_frame = self._read_image(image_files[0])
            first_inf_frame = self._read_image(inf_image_files[0])
            first_frame = [first_frame, first_inf_frame]
            output_boxes = []
            fps = []
            start_time = time.time()
            tracker.initialize(first_frame, info)
            output_boxes.append(info['init_bbox'])
            processed_frames = 1
            sequence_name = Path(sequence_folder).stem
            for i in range(1, len(image_files)):
                frame = self._read_image(image_files[i])
                inf_frame = self._read_image(inf_image_files[i])
                frame = [frame, inf_frame]
                out = tracker.track(frame, sequence_name, info)
                processed_frames += 1
                current_fps = processed_frames / (time.time() - start_time)
                fps.append(current_fps)
                output_boxes.append(out['target_bbox'])
            avg_fps = sum(fps) / len(fps)
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            
            base_results_path = os.path.join(self.results_dir, sequence_name + '.txt')
            with open (base_results_path, 'w') as f:
                for box in output_boxes:
                    formatted_box = [f"{b:.2f}" for b in box]
                    f.write(','.join(formatted_box) + '\n')
                f.write(f'FPS: {avg_fps:.2f}')
        else:
            # 查找可见光模态的地面实况文件
            visible_gt_path1 = os.path.join(sequence_folder, 'visible', ground_truth_pattern)
            visible_gt_path2 = os.path.join(sequence_folder, ground_truth_pattern)
            visible_gt_path = visible_gt_path1 if os.path.exists(visible_gt_path1) else visible_gt_path2
            visible_gt_files = glob.glob(visible_gt_path)

            assert len(visible_gt_files) == 1, f"{sequence_folder} 中未找到或存在多个可见光地面实况文件"

            visible_gt_path = visible_gt_files[0]

            # 从地面实况文件加载初始状态
            init_state_visible = self.load_initial_state_from_ground_truth(visible_gt_path)

            # 使用初始状态初始化info字典
            info = {
                'init_bbox': init_state_visible
            }

            # ... 其余代码放在这里 ...
            # 将现有的初始化和主跟踪循环逻辑替换为
            # 调用您的跟踪器API，并使用提供的`info`字典，
            # 同时跳过可视化和交互式ROI选择部分。

            visible_path1 = os.path.join(sequence_folder, 'visible')
            visible_path2 = os.path.join(sequence_folder, 'img')
            visible_path3 = sequence_folder
            visible_path = visible_path1 if os.path.exists(visible_path1) else visible_path2 if os.path.exists(visible_path2) else visible_path3
            assert os.path.isdir(visible_path), f"INvalid param {visible_path}, visible_path should be a folder"
            image_files = sorted([os.path.join(visible_path, f) for f in os.listdir(visible_path) if f.endswith(('.jpg', 'png', '.bmp'))]) 
            assert len(image_files) > 0, f"No image files found in {visible_path}"
            first_frame = self._read_image(image_files[0])
            
            output_boxes = []
            fps = []
            start_time = time.time()
            tracker.initialize(first_frame, info)
            output_boxes.append(info['init_bbox'])
            processed_frames = 1
            sequence_name = Path(sequence_folder).stem
            for i in range(1, len(image_files)):
                frame = self._read_image(image_files[i])
                frame = [frame]
                out = tracker.track(frame, sequence_name, info)
                processed_frames += 1
                current_fps = processed_frames / (time.time() - start_time)
                fps.append(current_fps)
                output_boxes.append(out['target_bbox'])
            avg_fps = sum(fps) / len(fps)
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            
            base_results_path = os.path.join(self.results_dir, sequence_name + '.txt')
            with open (base_results_path, 'w') as f:
                for box in output_boxes:
                    formatted_box = [f"{b:.2f}" for b in box]
                    f.write(','.join(formatted_box) + '\n')
                f.write(f'FPS: {avg_fps:.2f}')
        print(f"{sequence_name} Done.")
        
    def load_initial_state_from_ground_truth(self, gt_path):
        """
        从地面实况文件加载初始状态（边界框）。
        假设第一行包含初始边界框，格式为[x, y, w, h]。

        参数：
            gt_path (str): 地面实况文件的路径。

        返回值：
            list: 初始边界框，以整数列表形式表示[x, y, w, h]。
        """

        with open(gt_path, 'r') as f:
            init_line = f.readline().strip()

        init_state = [int(float(coord)) for coord in re.split(',|\s+', init_line)]
        assert len(init_state) == 4, f"{gt_path} 中初始状态格式无效"

        return init_state


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



