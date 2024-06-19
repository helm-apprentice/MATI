import os
import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
#     parser.add_argument('--tracker_name', type=str, default='procontext', help='Name of tracking method.')
#     parser.add_argument('--tracker_param', type=str, default='', help='Name of config file.')
#     parser.add_argument('--runid', type=int, default=None, help='The run id.')
#     parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
#     parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
#     parser.add_argument('--debug', type=int, default=1, help='Debug level.')
#     parser.add_argument('--threads', type=int, default=8, help='Number of threads.')
#     parser.add_argument('--num_gpus', type=int, default=1)

#     parser.add_argument('--videofilepath', type=str, default="")
#     parser.add_argument('--imageSequencePath', type=str, default="")
#     args = parser.parse_args()

#     return args

def main():
    # args = parse_args()
    # test_cmd = "python tracking/run_test.py --tracker_name %s --tracker_param %s --runid %s --dataset_name %d " \
    #     "--imageSequencePath %s --debug %s --threads %s --num_gpus %d > test_warnings.log 2>&1"\
    #     % (args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
    #         args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
    test_path = "/home/helm/tracker/ProContEXT-main/data/test/GOT-10k_Test_000001"
    test_cmd = "python tracking/run_test.py --imageSequencePath %s > test_warnings.log 2>&1" \
                % (test_path)
    os.system(test_cmd)

if __name__ == "__main__":
    main()