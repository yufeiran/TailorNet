import os
import numpy as np
import torch
import sys


from psbody.mesh import Mesh
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment
from TailorNet.utils.rotation import normalize_y_rotation
from visualization.blender_renderer import visualize_garment_body

from TailorNet.dataset.canonical_pose_dataset import get_style, get_shape
from visualization.vis_utils import get_specific_pose, get_specific_style_old_tshirt
from visualization.vis_utils import get_specific_shape, get_amass_sequence_thetas
from TailorNet.utils.interpenetration import remove_interpenetration_fast

# Set output path where inference results will be stored
# OUT_PATH = "/home/yufeiran/project/TailorNet/output"


class TailorNet():
    def __init__(self):
        self.gender = 'male'
        self.garment_class = 't-shirt'
        thetas, betas, gammas = get_single_frame_inputs(self.garment_class, self.gender)
        # # uncomment the line below to run inference on sequence data
        # thetas, betas, gammas = get_sequence_inputs(garment_class, gender)

        # load model
        self.tn_runner = get_tn_runner(gender=self.gender, garment_class=self.garment_class)
        # from TailorNet.trainer.base_trainer import get_best_runner
        # tn_runner = get_best_runner("/BS/cpatel/work/data/learn_anim/tn_baseline/{}_{}/".format(garment_class, gender))
        self.smpl = SMPL4Garment(gender=self.gender)
        self.out_path =  "/home/yufeiran/project/TailorNet/output"

        # make out directory if doesn't exist
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)


    def run_demo(self):
        thetas, betas, gammas = get_single_frame_inputs(self.garment_class, self.gender)
        # run inference
        for i, (theta, beta, gamma) in enumerate(zip(thetas, betas, gammas)):
            print(i, len(thetas))
            # normalize y-rotation to make it front facing
            theta_normalized = normalize_y_rotation(theta)
            with torch.no_grad():
                pred_verts_d = self.tn_runner.forward(
                    thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                    betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                    gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
                )[0].cpu().numpy()

            # get garment from predicted displacements
            body, pred_gar = self.smpl.run(beta=beta, theta=theta, garment_class=self.garment_class, garment_d=pred_verts_d)
            pred_gar = remove_interpenetration_fast(pred_gar, body)

            # save body and predicted garment
            body.write_ply(os.path.join(self.out_path, "body_{:04d}.ply".format(i)))
            pred_gar.write_ply(os.path.join(self.out_path, "pred_gar_{:04d}.ply".format(i)))

    def run_tailornet(self,theta,beta,name):
        # run inference

        thetas, betas, gammas = get_single_frame_inputs(self.garment_class, self.gender)

        gamma = gammas[0]

        # normalize y-rotation to make it front facing
        # theta_normalized = normalize_y_rotation(theta)
        theta_normalized = theta
        with torch.no_grad():
            pred_verts_d = self.tn_runner.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

        # get garment from predicted displacements
        body, pred_gar = self.smpl.run(beta=beta, theta=theta, garment_class=self.garment_class, garment_d=pred_verts_d)
        pred_gar = remove_interpenetration_fast(pred_gar, body)

        # save body and predicted garment
        body.write_ply(os.path.join(self.out_path, "body_{}.ply".format(name)))
        pred_gar.write_ply(os.path.join(self.out_path, "pred_gar_{}.ply".format(name)))



def get_single_frame_inputs(garment_class, gender):
    """Prepare some individual frame inputs."""
    betas = [
        get_specific_shape('tallthin'),
        get_specific_shape('shortfat'),
        get_specific_shape('mean'),
        get_specific_shape('somethin'),
        get_specific_shape('somefat'),
    ]
    # old t-shirt style parameters are centered around [1.5, 0.5, 1.5, 0.0]
    # whereas all other garments styles are centered around [0, 0, 0, 0]
    if garment_class == 'old-t-shirt':
        gammas = [
            get_specific_style_old_tshirt('mean'),
            get_specific_style_old_tshirt('big'),
            get_specific_style_old_tshirt('small'),
            get_specific_style_old_tshirt('shortsleeve'),
            get_specific_style_old_tshirt('big_shortsleeve'),
        ]
    else:
        gammas = [
            get_style('000', garment_class=garment_class, gender=gender),
            get_style('001', garment_class=garment_class, gender=gender),
            get_style('002', garment_class=garment_class, gender=gender),
            get_style('003', garment_class=garment_class, gender=gender),
            get_style('004', garment_class=garment_class, gender=gender),
        ]
    thetas = [
        get_specific_pose(0),
        get_specific_pose(1),
        get_specific_pose(2),
        get_specific_pose(3),
        get_specific_pose(4),
    ]
    return thetas, betas, gammas


def get_sequence_inputs(garment_class, gender):
    """Prepare sequence inputs."""
    beta = get_specific_shape('somethin')
    if garment_class == 'old-t-shirt':
        gamma = get_specific_style_old_tshirt('big_longsleeve')
    else:
        gamma = get_style('000', gender=gender, garment_class=garment_class)

    # downsample sequence frames by 2
    thetas = get_amass_sequence_thetas('05_02')[::2]

    betas = np.tile(beta[None, :], [thetas.shape[0], 1])
    gammas = np.tile(gamma[None, :], [thetas.shape[0], 1])
    return thetas, betas, gammas

