from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.check_dir = '/models'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data1/Datasets/Tracking/got10k_lmdb'
    settings.got10k_path = '/data1/Datasets/Tracking/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data1/Datasets/Tracking/itb'
    settings.lasot_extension_subset_path_path = '/data1/Datasets/Tracking/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data1/Datasets/Tracking/lasot_lmdb'
    settings.lasot_path = '/data1/Datasets/Tracking/lasot'
    settings.network_path = '/DATA/lipengcheng/code/PECNet/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data1/Datasets/Tracking/nfs'
    settings.otb_path = '/data1/Datasets/Tracking/otb'
    settings.prj_dir = '/DATA/lipengcheng/code/TBSI-main'
    settings.result_plot_path = '/DATA/lipengcheng/code/PECNet/output_newer/test/result_plots'
    settings.results_path = '/DATA/lipengcheng/code/PECNet/output_newer_/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/DATA/lipengcheng/code/PECNet/output_newer_'
    settings.segmentation_path = '/DATA/lipengcheng/code/PECNet/output/test/segmentation_results'
    settings.tc128_path = '/data1/Datasets/Tracking/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data1/Datasets/Tracking/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data1/Datasets/Tracking/trackingnet'
    settings.uav_path = '/data1/Datasets/Tracking/uav'
    settings.vot18_path = '/data1/Datasets/Tracking/vot2018'
    settings.vot22_path = '/data1/Datasets/Tracking/vot2022'
    settings.vot_path = '/data1/Datasets/Tracking/VOT2019'
    settings.youtubevos_dir = ''
    settings.gtot_path = "/DATA/lipengcheng/dataset/GTOT/"
    settings.rgbt210_path = "/DATA/lipengcheng/dataset/DATA/liqi/my_work/data/RGBT210/"
    settings.rgbt234_path = "/DATA/lipengcheng/dataset/RGBT234/"
    settings.lasher_path = "/DATA/lipengcheng/dataset/LasHeR/"
    settings.vtuav_path = "/DATA/liqi/my_work/data/VTUAV/"

    return settings

