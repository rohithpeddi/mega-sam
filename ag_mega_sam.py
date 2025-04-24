import argparse
import glob
import os
from timeit import default_timer as timer
import cv2
from Depth_Anything.depth_anything.dpt import DPT_DINOv2
from Depth_Anything.depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from PIL import Image

from UniDepth.unidepth.models import UniDepthV2


class AgMegaSam:

    def __init__(self, datapath):
        self.datapath = datapath
        self.frames_path = os.path.join(self.datapath, "frames")
        self.annotations_path = os.path.join(self.datapath, "annotations")
        self.video_list = sorted(os.listdir(self.frames_path))
        self.gt_annotations = sorted(os.listdir(self.annotations_path))
        print("Total number of ground truth annotations: ", len(self.gt_annotations))

        # ------ Depth Anything parameters ------
        self.margin_width = 50
        self.caption_height = 60

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2

        self._depth_anything_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "depth_anything")

        # ------- UniDepth parameters -------
        self._unidepth_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "unidepth")

        self.LONG_DIM = 640


    # -------------------------- DEPTH ANYTHING --------------------------

    def _load_depth_anything_model(self, args):
        if args.encoder == 'vits':
            self.depth_anything = DPT_DINOv2(
                encoder='vits',
                features=64,
                out_channels=[48, 96, 192, 384],
                localhub=args.localhub,
            ).cuda()
        elif args.encoder == 'vitb':
            self.depth_anything = DPT_DINOv2(
                encoder='vitb',
                features=128,
                out_channels=[96, 192, 384, 768],
                localhub=args.localhub,
            ).cuda()
        else:
            self.depth_anything = DPT_DINOv2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024],
                localhub=args.localhub,
            ).cuda()

        total_params = sum(param.numel() for param in self.depth_anything.parameters())
        print('Total parameters: {:.2f}M'.format(total_params / 1e6))
        self.depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'), strict=True)
        self.depth_anything.eval()

        # ------ Initialize transformations ------
        self._depth_anything_transforms = Compose([
            Resize(
                width=768,
                height=768,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='upper_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])


    def video_depth_anything_estimation(self, video_id, img_paths):
        output_dir = os.path.join(self._depth_anything_root, video_id)

        if os.path.exists(output_dir):
            if len(os.listdir(output_dir)) == len(img_paths):
                print(f"Depth estimation already completed for {video_id}. Skipping...")
                return
            else:
                # Remove the existing directory if it is not empty and not complete
                os.rmdir(output_dir)
                print(f"Removing incomplete directory for {video_id}...")

        os.makedirs(output_dir, exist_ok=True)

        for filename in tqdm(img_paths, desc=f"Processing {video_id}"):
            raw_image = cv2.imread(filename)[..., :3]
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]

            image = self._depth_anything_transforms({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).cuda()

            # start = timer()
            with torch.no_grad():
                depth = self.depth_anything(image)
            # end = timer()

            depth = F.interpolate(
                depth[None], (h, w), mode='bilinear', align_corners=False
            )[0, 0]
            depth_npy = np.float32(depth.cpu().numpy())

            np.save(
                os.path.join(output_dir, filename.split('/')[-1][:-4] + '.npy'),
                depth_npy,
            )

    def run_ag_depth_anything_estimation(self):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = sorted(glob.glob(os.path.join(video_frames_path, '*.png')))
            img_paths += sorted(glob.glob(os.path.join(video_frames_path, '*.jpg')))

            self.video_depth_anything_estimation(video_id, img_paths)

        print("Depth estimation completed for all videos.")

    # -------------------------- UNIDEPTH --------------------------

    def _load_unidepth_model(self, args):
        self._unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14",
                                           revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._unidepth_model = self._unidepth_model.to(device)
        self._unidepth_model.eval()

    def video_unidepth_estimation(self, video_id, img_paths):
        output_dir = os.path.join(self._unidepth_root, video_id)

        if os.path.exists(output_dir):
            if len(os.listdir(output_dir)) == len(img_paths):
                print(f"Depth estimation already completed for {video_id}. Skipping...")
                return
            else:
                # Remove the existing directory if it is not empty and not complete
                os.rmdir(output_dir)
                print(f"Removing incomplete directory for {video_id}...")

        os.makedirs(output_dir, exist_ok=True)

        fovs = []
        for img_path in tqdm(img_paths):
            rgb = np.array(Image.open(img_path))[..., :3]
            if rgb.shape[1] > rgb.shape[0]:
                final_w, final_h = self.LONG_DIM, int(
                    round(self.LONG_DIM * rgb.shape[0] / rgb.shape[1])
                )
            else:
                final_w, final_h = (
                    int(round(self.LONG_DIM * rgb.shape[1] / rgb.shape[0])),
                    self.LONG_DIM,
                )
            rgb = cv2.resize(
                rgb, (final_w, final_h), cv2.INTER_AREA
            )  # .transpose(2, 0, 1)

            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
            # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
            # predict
            predictions = self._unidepth_model.infer(rgb_torch)
            fov_ = np.rad2deg(
                2
                * np.arctan(
                    predictions["depth"].shape[-1]
                    / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
                )
            )
            depth = predictions["depth"][0, 0].cpu().numpy()
            # print(fov_)
            fovs.append(fov_)
            # breakpoint()
            np.savez(
                os.path.join(output_dir, img_path.split("/")[-1][:-4] + ".npz"),
                depth=np.float32(depth),
                fov=fov_,
            )

    def run_ag_unidepth_estimation(self):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = sorted(glob.glob(os.path.join(video_frames_path, '*.png')))
            img_paths += sorted(glob.glob(os.path.join(video_frames_path, '*.jpg')))

            self.video_unidepth_estimation(video_id, img_paths)

        print("Depth estimation completed for all videos.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/data/rohith/ag/")
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--load_from', type=str, default='/home/rxp190007/CODE/mega-sam/Depth_Anything/checkpoints/depth_anything_vitl14.pth')
    parser.add_argument('--localhub', dest='localhub', action='store_true', default=False)

    parser.add_argument("--mode", type=str, choices=["depth_anything", "uni_depth", "camera_tracking", "preprocess_flow", "cvd_opt"], required=True)

    args = parser.parse_args()
    ag_mega_sam = AgMegaSam(datapath=args.datapath)

    # ----- Run depth anything depth estimation -----
    if args.mode == "depth_anything":
        print("Running depth estimation using Depth Anything...")
        ag_mega_sam._load_depth_anything_model(args)
        ag_mega_sam.run_ag_depth_anything_estimation()
    # ----- Run unidepth depth estimation -----
    elif args.mode == "uni_depth":
        print("Running depth estimation using UniDepth...")
        ag_mega_sam._load_unidepth_model(args)
        ag_mega_sam.run_ag_unidepth_estimation()
    else:
        raise ValueError("Invalid mode selected. Choose from ['depth_anything', 'uni_depth']")


if __name__ == '__main__':
    main()