import minisam
from minisam import DiagonalLoss, PriorFactor, SE3, SO3, BetweenFactor, LevenbergMarquardtOptimizer, \
    LevenbergMarquardtOptimizerParams, \
    NonlinearOptimizerVerbosityLevel, NonlinearOptimizationStatus, MarginalCovarianceSolver, \
    MarginalCovarianceSolverStatus
import numpy as np
import math3d as m3d


class MinisamPoseGraphOptim:
    def __init__(self, prior_sigma=1e-4):
        self.graph = minisam.FactorGraph()
        self.prior_loss = DiagonalLoss.Sigmas(np.ones(6) * prior_sigma)
        self.vertices = {}
        self.initials = minisam.Variables()
        self.optim_results = None

    def add_vertex(self, idx, pose=None, fixed=False):  # pose in m3d.Transform
        new_vertex = minisam.key('v', idx)  # idx>0
        self.vertices[idx] = new_vertex
        pose = np.eye(4) if pose is None else pose.array
        self.initials.add(new_vertex, SE3(pose))
        if fixed:
            # prior_loss = self.prior_loss if prior_loss is None else DiagonalLoss.Sigmas(np.ones(6) * prior_loss)
            self.graph.add(PriorFactor(new_vertex, SE3(pose), self.prior_loss))

    def add_edge(self, vertices, pose, sigma=1e-2):  # voTv1
        measure_loss = DiagonalLoss.Sigmas(np.ones(6) * sigma)
        v0, v1 = vertices
        factor = BetweenFactor(self.vertices[v0], self.vertices[v1], SE3(pose.array), measure_loss)
        self.graph.add(factor)

    def optimize(self, mute=False):
        opt_param = LevenbergMarquardtOptimizerParams()

        opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.WARNING
        opt = LevenbergMarquardtOptimizer(opt_param)
        results = minisam.Variables()
        status = opt.optimize(self.graph, self.initials, results)

        if status != NonlinearOptimizationStatus.SUCCESS:
            if not mute:
                print("optimization error: ", status)
        # print(results, '\n')
        # mcov_solver = MarginalCovarianceSolver()
        # status = mcov_solver.initialize(self.graph, results)
        # if status != MarginalCovarianceSolverStatus.SUCCESS:
        #     print("maginal covariance error")
        #     print(status)
        # result_covs = {}
        # for k, v in self.vertices.items():
        #     result_covs[k] = mcov_solver.marginalCovariance(v)
        self.optim_results = results

    def get_pose(self, id):
        return m3d.Transform(self.optim_results.at(self.vertices[id]).matrix())


def opt_poses(poses):  # Optimized poses from list of measurements of the same relative poses
    g = MinisamPoseGraphOptim()
    g.add_vertex(0, None, True)  # base
    g.add_vertex(1, poses[0])  # object
    for p in poses:
        g.add_edge([0, 1], p)
    g.optimize()
    return g.get_pose(0).inverse * g.get_pose(1)


# def n_cams_optim_minisam(cnTos_lst):  # , tag_num_os):
#     """
#     :param cnTos_lst: [cam0Tos,cam1Tos,cam2Tos...]
#         each item is a list of poses cam0Tos=[cam0To0,cam0To1,cam0To2...cam0Tom],
#          total m observations o0~om
#     :return:
#     """
#     g = MinisamPoseGraphOptim()
#     n_cam = len(cnTos_lst)
#     n_obs = len(cnTos_lst[0])
#     print('Optimization: {} cameras with {} observations'.format(n_cam, n_obs))
#
#     for cam_idx in range(n_cam):  # add camera vertices
#         if cam_idx == 0:
#             g.add_vertex(cam_idx, None, fixed=True)
#         else:
#             g.add_vertex(cam_idx)
#     obs_id = n_cam  # start from n_cam
#
#     for cnTos in zip(*cnTos_lst):
#         g.add_vertex(obs_id)
#         for cam_idx, ckTos in enumerate(cnTos):
#             g.add_edge([cam_idx, obs_id], ckTos, sigma=1)
#         obs_id += 1
#
#     # Note: Initial Estimate for cameras, use first frame, crucial !!
#     cam0To0 = cnTos_lst[0][0]
#     for cam_idx, cnTos in enumerate(cnTos_lst):
#         if cam_idx != 0:
#             g.add_edge([0, cam_idx], cam0To0 * cnTos[0].inverse, sigma=1e-2)
#
#     g.optimize()
#     poses = []  # cam0Tcam1,cam0Tcam2....
#     cam_0_pose = g.get_pose(0)
#     for cam_idx in range(1, n_cam):
#         pose = g.get_pose(cam_idx)
#         pose = cam_0_pose.inverse * pose
#         poses.append(pose)
#     return poses
#
#
# def three_cam_estimation(c1Tc2: m3d.Transform, c1Tc3: m3d.Transform, c1To: m3d.Transform, c2To: m3d.Transform,
#                          c3To: m3d.Transform):  # return optimized c2To
#     """
#     all input/output poses in m3d.Transform type
#     """
#     g = MinisamPoseGraphOptim()
#     g.add_vertex(1, m3d.Transform(), fixed=True)  # camera 1,origin
#     g.add_vertex(2, c1Tc2, fixed=True)  # camera 2
#     g.add_vertex(3, c1Tc3, fixed=True)  # camera 3
#
#     # object, initial guess relative to camera 1
#     if c1To is not None:
#         g.add_vertex(0, c1To)
#     elif c2To is not None:
#         g.add_vertex(0, c1Tc2 * c2To)
#     elif c3To is not None:
#         g.add_vertex(0, c1Tc3 * c3To)
#     else:
#         raise NotImplementedError('At least one camera should observe the object')
#
#     if c1To is not None:
#         g.add_edge([1, 0], c1To)
#     if c2To is not None:
#         g.add_edge([2, 0], c2To)
#     if c3To is not None:
#         g.add_edge([3, 0], c3To)
#     g.optimize(mute=True)
#     gTobj = g.get_pose(0)
#     gTcam1 = g.get_pose(1)  # should be small
#     return gTcam1.inverse * gTobj  # cam1Tobj
#
#
# def three_cam_estimation_n_measure(c1Tc2: m3d.Transform, c1Tc3: m3d.Transform, c1To: [], c2To: [],
#                                    c3To: []):  # return optimized c2To
#     """
#     all input/output poses in m3d.Transform type
#     """
#     g = MinisamPoseGraphOptim()
#     g.add_vertex(1, m3d.Transform(), fixed=True)  # camera 1,origin
#     g.add_vertex(2, c1Tc2, fixed=True)  # camera 2
#     g.add_vertex(3, c1Tc3, fixed=True)  # camera 3
#
#     # object, initial guess relative to camera 1
#     if len(c1To) != 0:
#         g.add_vertex(0, c1To[0])
#     elif len(c2To) != 0:
#         g.add_vertex(0, c1Tc2 * c2To[0])
#     elif len(c3To) != 0:
#         g.add_vertex(0, c1Tc3 * c3To[0])
#     else:
#         raise NotImplementedError('At least one camera should observe the object')
#
#     for tr in c1To:
#         g.add_edge([1, 0], tr)
#     for tr in c2To:
#         g.add_edge([2, 0], tr)
#     for tr in c3To:
#         g.add_edge([3, 0], tr)
#     g.optimize(mute=True)
#     gTobj = g.get_pose(0)
#     gTcam1 = g.get_pose(1)  # should be small
#     return gTcam1.inverse * gTobj  # cam1Tobj
#
#
# # Works fine
# def n_cams_optim_minisam_direct(cnTos_lst):
#     """
#     :param cnTos_lst: [cam0Tos,cam1Tos,cam2Tos...]
#         each item is a list of poses cam0Tos=[cam0To0,cam0To1,cam0To2...cam0Tom]
#     :return:
#     """
#     g = MinisamPoseGraphOptim()
#     n_cam = len(cnTos_lst)
#     n_obs = len(cnTos_lst[0])
#     print('Optimization: {} cameras with {} observations'.format(n_cam, n_obs))
#
#     for cam_idx in range(n_cam):  # add camera vertices
#         if cam_idx == 0:
#             g.add_vertex(cam_idx, None, fixed=True)
#         else:
#             g.add_vertex(cam_idx)
#
#     for cnTos in zip(*cnTos_lst):
#         cam0Ton = None
#         for cam_idx, ckTos in enumerate(cnTos):
#             if cam_idx == 0:
#                 cam0Ton = ckTos
#                 continue
#             cam0Tcamk = cam0Ton * ckTos.inverse
#             g.add_edge([0, cam_idx], cam0Tcamk, sigma=1e-2)
#
#     # # TODO: Initial Estimate for cameras, use first frame
#     # cam0To0 = cnTos_lst[0][0]
#     # for cam_idx, cnTos in enumerate(cnTos_lst):
#     #     if cam_idx != 0:
#     #         g.add_edge([0, cam_idx], cam0To0 * cnTos[0].inverse, sigma=1e-1)
#
#     g.optimize()
#     poses = []  # cam0Tcam1,cam0Tcam2....
#     cam_0_pose = g.get_pose(0)
#     for cam_idx in range(1, n_cam):
#         pose = g.get_pose(cam_idx)
#         pose = cam_0_pose.inverse * pose
#         poses.append(pose)
#     return poses
