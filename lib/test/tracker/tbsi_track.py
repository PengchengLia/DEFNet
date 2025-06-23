import math

from lib.models.tbsi_track import build_tbsi_track
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import torch.nn.functional as F


class TBSITrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TBSITrack, self).__init__(params)
        network = build_tbsi_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]], search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:]], ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        

        # y = out_dict['backbone_feat']
        # q = k = y.reshape(1, 320, 12, 64).permute(0, 2, 1, 3)
        # attn = (q @ (k.transpose(-2, -1))) * (64** -0.5)
        # attn = attn.softmax(dim=-1)

        # for debug
        if self.debug:
            # visualization
            center = 28

            attn_before_v = self.network.backbone.Fusion_Layers[2].DIIM1.attn_before_v
            attn_before_v = torch.mean(attn_before_v[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            attn_before_v = F.interpolate(attn_before_v, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            attn_before_i = self.network.backbone.Fusion_Layers[2].DIIM1.attn_before_i
            attn_before_i = torch.mean(attn_before_i[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            attn_before_i = F.interpolate(attn_before_i, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            attn_after_v = self.network.backbone.Fusion_Layers[2].DIIM1.attn_after_v
            attn_after_v = torch.mean(attn_after_v[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            attn_after_v = F.interpolate(attn_after_v, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            attn_after_i = self.network.backbone.Fusion_Layers[2].DIIM1.attn_after_i
            attn_after_i = torch.mean(attn_after_i[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            attn_after_i = F.interpolate(attn_after_i, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            attn_fuse = self.network.backbone.Fusion_Layers[2].DIIM1.attn_fuse
            attn_fuse = torch.mean(attn_fuse[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            attn_fuse = F.interpolate(attn_fuse, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)



            # x_v, x_i = out_dict['x_v'], out_dict['x_i']
            # q_v = k_v = x_v.reshape(1, 320, 12, 64).permute(0, 2, 1, 3)
            # q_i = k_i = x_i.reshape(1, 320, 12, 64).permute(0, 2, 1, 3)
            # attn_v = (q_v @ (k_v.transpose(-2, -1))) * (64** -0.5)
            # attn_i = (q_i @ (k_i.transpose(-2, -1))) * (64** -0.5)

            # attn_v = attn_v.softmax(dim=-1)
            # attn_i = attn_i.softmax(dim=-1)

            # # attn = torch.mean(attn[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            # # attn = F.interpolate(attn, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            # attn_v = torch.mean(attn_v[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            # attn_v = F.interpolate(attn_v, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            # attn_i = torch.mean(attn_i[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            # attn_i = F.interpolate(attn_i, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            # q_m_v = k_m_v = self.network.backbone.Final_Fusion.fun.D1.v1.reshape(1, 320, 12, 64).permute(0, 2, 1, 3)
            # q_m_i = k_m_i = self.network.backbone.Final_Fusion.fun.D2.v1.reshape(1, 320, 12, 64).permute(0, 2, 1, 3)
            # q_output_v = k_output_v = self.network.backbone.Final_Fusion.fun.v_1.reshape(1, 320, 12, 64).permute(0, 2, 1, 3)
            # q_output_i = k_output_i = self.network.backbone.Final_Fusion.fun.v_2.reshape(1, 320, 12, 64).permute(0, 2, 1, 3)
            # attn_m_v = (q_m_v @ (k_m_v.transpose(-2, -1))) * (64** -0.5)
            # attn_m_i = (q_m_i @ (k_m_i.transpose(-2, -1))) * (64** -0.5)
            # attn_m_v = attn_m_v.softmax(dim=-1)
            # attn_m_i = attn_m_i.softmax(dim=-1)

            # attn_out_v = (q_output_v @ (k_output_v.transpose(-2, -1))) * (64** -0.5)
            # attn_out_i = (q_output_i @ (k_output_i.transpose(-2, -1))) * (64** -0.5)
            # attn_out_v = attn_out_v.softmax(dim=-1)
            # attn_out_i = attn_out_i.softmax(dim=-1)

            # attn_m_v = torch.mean(attn_m_v[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            # attn_m_v = F.interpolate(attn_m_v, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            # attn_m_i = torch.mean(attn_m_i[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            # attn_m_i = F.interpolate(attn_m_i, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            # attn_out_v = torch.mean(attn_out_v[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            # attn_out_v = F.interpolate(attn_out_v, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)

            # attn_out_i = torch.mean(attn_out_i[..., 64:, center],dim=1, keepdim=True).reshape(1,1,16,16)
            # attn_out_i = F.interpolate(attn_out_i, size=(256, 256), mode='bilinear', align_corners=False).reshape(256,256)



            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image[:,:,0:3], cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                # self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register((image[:,:,0:3], info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,0:3]).permute(2, 0, 1), 'image', 1, 'search_region_v')
                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,3:]).permute(2, 0, 1), 'image', 1, 'search_region_i')
                self.visdom.register(torch.from_numpy(self.z_patch_arr[:,:,0:3]).permute(2, 0, 1), 'image', 1, 'template_v')
                self.visdom.register(torch.from_numpy(self.z_patch_arr[:,:,3:]).permute(2, 0, 1), 'image', 1, 'template_i')
                # self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                # self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                # add visualization
                self.visdom.register(attn_before_v, 'heatmap', 1, 'attn_before_v')
                self.visdom.register(attn_before_i, 'heatmap', 1, 'attn_before_i')
                self.visdom.register(attn_after_v, 'heatmap', 1, 'attn_after_v')
                self.visdom.register(attn_after_i, 'heatmap', 1, 'attn_after_i')
                self.visdom.register(attn_fuse, 'heatmap', 1, 'attn_fuse')
                # self.visdom.register(attn, 'heatmap', 1, 'attn')
                # self.visdom.register(attn_v, 'heatmap', 1, 'attn_v')
                # self.visdom.register(attn_i, 'heatmap', 1, 'attn_i')
                # self.visdom.register(attn_m_v, 'heatmap', 1, 'attn_m_v')
                # self.visdom.register(attn_m_i, 'heatmap', 1, 'attn_m_i')
                # self.visdom.register(attn_out_v, 'heatmap', 1, 'attn_out_v')
                # self.visdom.register(attn_out_i, 'heatmap', 1, 'attn_out_i')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TBSITrack
