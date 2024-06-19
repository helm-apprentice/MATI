import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')

if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)

def run_video(tracker_name, tracker_param, run_id, dataset_name, videofilepath, sequence=None, debug=0, threads=0,
              num_gpus=1):
    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    trackers[0].run_video(videofilepath)

def run_image_sequence(tracker_name, tracker_param, run_id, dataset_name, imageSequencePath, sequence=None, debug=0,
                       threads=0, use_visdom=0, save_results=0):
    # dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    # print(imageSequencepath)
    trackers[0].run_image_sequence(imageSequencePath, debug=debug, use_visdom=use_visdom, save_results=save_results)

def run_all_image_sequences(tracker_name, tracker_param, dataset_name, run_id, data_root):
    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    trackers[0].run_all_image_sequences(data_root)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='procontext', help='Name of tracking method.')
    # parser.add_argument('--tracker_param', type=str, default='vitb_384_mae_ce_32x4_got10k_ep300', help='Name of config file.')
    parser.add_argument('--tracker_param', type=str, default='cmc3', help='Name of config file.')
    parser.add_argument('--runid', type=int, default=972, help='The run id.')
    # parser.add_argument('--dataset_name', type=str, default='got10k_test', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--dataset_name', type=str, default='got10k_test', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--use_visdom', type=int, default=1, help='whether to use visdom.')
    parser.add_argument('--save_results', type=int, default=1, help='whether to save results.')

    # dark_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible002jpg(random_dark3+4)"
    # dark_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible002(jpg)"
    dark_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible001jpg(random_dark3+5)"
    #double_test_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/test/plane1/visible"
    double_test_path = "/home/helm/tracker/ProContEXT-main/data/test()/plane1"
    double_full_path = "/home/helm/tracker/ProContEXT-main/data/sequence_plane2"
    vis_path = "/home/helm/tracker/ProContEXT-main/data/val(single)/GOT-10kplus_Val_c"
    test_path = "/home/helm/tracker/mine/data/test/GOT-10k_Test_000035"
    gtot_test_path = "/home/helm/tracker/ProContEXT-main/data/test/bikeman"
    data_root = '/home/helm/tracker/ProContEXT-main/data/test'
    val_path = '/media/helm/C4E1CE1E0192B573/udata/GTOT/val'
    otb_path = "/media/helm/C4E1CE1E0192B573/udata/otb100/otb100"
    self_data = "/home/helm/tracker/mine/data/val"
    # double_full_path = "/home/helm/tracker/ProContEXT-main/data/train(double)/bikemove1"
    parser.add_argument('--videofilepath', type=str, default="")
    # parser.add_argument('--imageSequencePath', type=str, default="/home/helm/tracker/datas/可见plus红外/可见/可见02")
    parser.add_argument('--imageSequencePath', type=str, default=double_full_path)
    args = parser.parse_args()
    
    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    # run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
    #              args.threads, num_gpus=args.num_gpus)
    # run_video(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, args.videofilepath, seq_name, args.debug,
    #             args.threads, num_gpus=args.num_gpus)
    
    run_image_sequence(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, args.imageSequencePath, seq_name, args.debug,
               args.threads, use_visdom=args.use_visdom, save_results=args.save_results)
    # run_all_image_sequences(args.tracker_name, args.tracker_param, args.dataset_name, args.runid, args.imageSequencePath)


if __name__ == '__main__':
    main()
