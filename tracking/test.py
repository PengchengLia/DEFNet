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
    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus
    )


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='tbsi_track', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='vitb_256_tbsi_32x1_1e4_lasher_15ep_sot', help='Name of config file.')
    parser.add_argument('--runid', type=str, default=10, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='lasher', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default='boyaftertree',help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=1, help='Debug level.')   #开可视化就设置成1，不开设置成0
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--vis_gpus', type=str, default='1')  #可视化需要的卡号

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_gpus

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()