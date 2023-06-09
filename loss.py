# -*- coding = utf-8 -*-
# @File Name : loss
# @Date : 2023/5/21 13:09
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import torch.nn.functional as F


# --------- Utility Functions ---------
def preproc_output(output):
    """
    Pre-process the output of the network
    :param output: Output dictionary of the network
    :return: vessel: 2D image [B, H, W, 2], 3D image [B, H, W, D, 3]. Optimal direction
             radius: 2D image [B, H, W, 1], 3D image [B, H, W, D, 1].
    """
    vessel = output['vessel']  # this should have shape of [B, 3, H, W, D] / [B, 2, H, W]
    radius = output['radius']  # this should have shape of [B, 1, H, W, D] / [B, 1, H, W]
    vessel = F.normalize(vessel, dim=1)         # normalize the optimal dir
    # if 3D image such as CT and MRA
    if vessel.dim() == 5:
        vessel = vessel.permute(0, 2, 3, 4, 1)  # change to [B, H, W, D, 3]
        radius = radius.permute(0, 2, 3, 4, 1)  # change to [B, H, W, D, 1]
    else:
        # if 2D image such as OCT
        vessel = vessel.permute(0, 2, 3, 1)     # change to [B, H, W, 2]
        radius = radius.permute(0, 2, 3, 1)     # change to [B, H, W, 1]
    return vessel, radius


def get_orthogonal_basis(optimal_dir):
    """
    Get orthogonal vectors of other two directions
    :param optimal_dir: 3D image [B, H, W, D, 3] / 2D image [B, H, W, 2]
    :return: basis: 3D image [B, H, W, D, n(3), 3] / 2D image [B, H, W, n(2), 2]
    """
    # if 3D image such as CT and MRA
    if optimal_dir.dim() == 5:
        c = torch.randn_like(optimal_dir, device=optimal_dir.device)
        ortho_dir_1 = torch.cross(c, optimal_dir, dim=4)
        ortho_dir_1 = ortho_dir_1 / ortho_dir_1.norm(dim=4, keepdim=True) + 1e-10
        ortho_dir_2 = torch.cross(optimal_dir, ortho_dir_1, dim=4)
        ortho_dir_2 = ortho_dir_2 / ortho_dir_2.norm(dim=4, keepdim=True) + 1e-10
        basis = torch.stack((optimal_dir, ortho_dir_1, ortho_dir_2), dim=4)
    else:
        # if 2D image such as OCT
        index = torch.LongTensor([1, 0]).to(optimal_dir.device)
        ortho_dir_1 = torch.index_select(optimal_dir, -1, index)
        ortho_dir_1[:, :, :, 1] = -ortho_dir_1[:, :, :, 1]
        ortho_dir_1 = ortho_dir_1 / ortho_dir_1.norm(dim=3, keepdim=True) + 1e-10
        basis = torch.stack((optimal_dir, ortho_dir_1), dim=3)
    return basis


def get_sampling_vec(num_pts, estimated_r):
    """
    Get the sampling vectors, sphere or circle
    :param num_pts: sampling num points
    :param estimated_r: estimated radius, used to parse the device
    :return: sampling vectors
    """
    # if 3D image such as CT and MRA
    if estimated_r.dim() == 5:
        indices = torch.arange(0, num_pts, dtype=torch.float32)
        phi = torch.arccos(1 - 2 * indices / num_pts)
        theta = torch.pi * (1 + 5 ** 0.5) * indices
        x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
        # flip coordinates according to the sample grid
        vectors = torch.vstack((z, y, x)).T.to(estimated_r.device)      # This is a sphere sampling
    else:
        # if 2D image such as OCT
        angle = 2.0 * torch.pi * torch.arange(0, num_pts) / num_pts
        x, y = 1.0 * torch.cos(angle), 1.0 * torch.sin(angle)
        vectors = torch.vstack((x, y)).T.to(estimated_r.device)         # This is a circle sampling
    return vectors


def get_gradients(image, dims=(2, 3, 4), channel_dim=1):
    """
    Get gradients of batch image
    :param image: 3D image [B, 1, H, W, D] / 2D image [B, 1, H, W]
    :param dims: 3D image (2, 3, 4) / 2D image (2, 3)
    :param channel_dim: channel dimension, default=1
    :return: gradients: 3D image [B, 3, H, W, D] / 2D image [B, 2, H, W]
    """
    gradients = torch.gradient(image, dim=dims)
    gradients = torch.cat(gradients, dim=channel_dim)
    gradients += torch.randn(gradients.size(), device=gradients.device) * 1e-10
    return gradients


def get_grid_base(gradients):
    """
    Get the image grid
    meshgrid method will switch x and z axis
    :param gradients: [B, 2, H, W] / [B, 3, H, W, D]
    :return: grid : [B, H, W, 2] / [B, H, W, D, 3]
    """
    b, c, h, w = gradients.shape[:4]
    d = gradients.shape[4] if gradients.dim() == 5 else None
    dh = torch.linspace(-1.0, 1.0, h)
    dw = torch.linspace(-1.0, 1.0, w)
    if d:
        dd = torch.linspace(-1.0, 1.0, d)
        meshx, meshy, meshz = torch.meshgrid((dh, dw, dd), indexing='ij')
        # need to swap the order of xyz
        grid = torch.stack((meshz, meshy, meshx), dim=3).repeat((b, 1, 1, 1, 1))    # [B, H, W, D, 3]
    else:
        meshx, meshy = torch.meshgrid((dh, dw), indexing='ij')
        grid = torch.stack((meshy, meshx), dim=2).repeat((b, 1, 1, 1))           # [B, H, W, 2]
    return grid.to(gradients.device)


def sample_space_to_img_space(grid, h, w, d=None):
    """
    Convert the image space to sample space
    [[-1, 1], [-1, 1], [-1, 1]] -> [[0, H], [0, W], [0, D]]
    grid is of size [B, H, W, D, 3] or [B, H, W, 2]
    convert [-1, 1] scale to [0, H] scale
    :param grid: [B, H, W, D, 3] or [B, H, W, 2]
    :param h: image height, int
    :param w: image width, int
    :param d: image depth, int, only used if the grid is 3D
    :return: [B, H, W, D, 3] or [B, H, W, 2]
    """
    #
    grid = grid + 0
    grid = grid * 0.5 + 0.5
    grid[..., 0] = grid[..., 0] * h
    grid[..., 1] = grid[..., 1] * w
    if d:
        grid[..., 2] = grid[..., 2] * d
    return grid


def img_space_to_sample_space(grid, h, w, d=None):
    """
    Convert the image space to sample space
    [[0, H], [0, W], [0, D]] -> [[-1, 1], [-1, 1], [-1, 1]]
    grid is of size [B, H, W, D, 3] or [B, H, W, 2]
    convert [0, H] scale to [-1, 1] scale
    :param grid: [B, H, W, D, 3] or [B, H, W, 2]
    :param h: image height, int
    :param w: image width, int
    :param d: image depth, int, only used if the grid is 3D
    :return: [B, H, W, D, 3] or [B, H, W, 2]
    """
    grid = grid + 0
    grid[..., 0] = 2.0 * grid[..., 0] / h - 1
    grid[..., 1] = 2.0 * grid[..., 1] / w - 1
    if d:
        grid[..., 2] = 2.0 * grid[..., 2] / d - 1
    return grid


def grid_sample(image, sample_grid, permute):
    """
    Functional grid sample overload
    :param image: [B, C, H, W] / [B, C, H, W, D]
    :param sample_grid: [B, H, W, 2] / [B, H, W, D, 3]
    :param permute: bool, indicate to change the shape or not
    :return:  [B, H, W, 1, 2] / [B, H, W, D, 1, 3]
    """
    sampled = F.grid_sample(image, sample_grid, align_corners=True, padding_mode='border')
    if image.dim() == 5:
        if permute:
            sampled = sampled.permute(0, 2, 3, 4, 1).unsqueeze(4)
    else:
        if permute:
            sampled = sampled.permute(0, 2, 3, 1).unsqueeze(3)
    return sampled


def project(vectors, basis, proj=False):
    """
    Project gradients on to the basis
    :param vectors: 2D image [B, H, W, k(1/2), 2] / 3D image [B, H, W, D, k(1/3), 3]
    :param basis: 2D image [B, H, W, 2, 2] / 3D image [B, H, W, D, 3, 3]
    :param proj: bool, whether to project ob the base
    :return:
        proj = True
            2D image [B, H, W, 2, 2] / 3D image [B, H, W, D, 3, 3]
        proj = False
            2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    """
    proj_vectors = torch.sum(torch.mul(vectors, basis), dim=-1)         # [B, H, W, 2] / [B, H, W, D, 3]
    if proj:
        proj_vectors = torch.mul(proj_vectors.unsqueeze(-1), basis)     # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    return proj_vectors


def swap_order(direction, order, dim):
    """
    Swap the axis of the tensor
    :param direction: 2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    :param order: list, the order of X, Y and Z axis
    :param dim: -1 or 3 or 4
    :return: 2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    """
    index = torch.LongTensor(order).to(direction.device)
    direction = torch.index_select(direction, dim, index)
    return direction


def proc_sample_dir(sample_dir_scaled, estimated_r):
    """
    Process the sampled direction
    :param sample_dir_scaled: 2D image [B, H, W, 2] / 3D image [B, H, W, D, 3]
    :param estimated_r: 2D image [B, H, W, 1] / 3D image [B, H, W, D, 1]
    :return: 2D image [B, H, W, 2, 2] / 3D image [B, H, W, D, 3, 3]
    """
    if sample_dir_scaled.dim() == 5:
        sample_dir_scaled = swap_order(sample_dir_scaled, [2, 1, 0], dim=-1)
    else:
        sample_dir_scaled = swap_order(sample_dir_scaled, [1, 0], dim=-1)
    sample_dir_scaled = torch.div(sample_dir_scaled, estimated_r)   # [B, H, W, 2] / [B, H, W, D, 3]
    sample_dir_scaled = sample_dir_scaled.unsqueeze(-2)             # [B, H, W, 1, 2] / [B, H, W, D, 1, 3]
    repeat_size = torch.ones(sample_dir_scaled.dim(), dtype=torch.int).tolist()
    repeat_size[-2] = sample_dir_scaled.size(-1)                    # [1, 1, 1, 2, 1] / [1, 1, 1, 1, 3, 1]
    sample_dir_scaled = sample_dir_scaled.repeat(repeat_size)       # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    return sample_dir_scaled


def get_sample_grid(sample_dir, curr_radius, grid_base, b, h, w, d):
    """
    Compute the sampling grid based on the radius and sample direction
    :param sample_dir: sampling direction   2D or 3D vector
    :param curr_radius: radius corresponding to this direction
    :param grid_base: grid base for sampling of the original image [B, H, W, 2] or [B, H, W, D, 3]
    :param b: batch size, int
    :param h: image, height, int
    :param w: image width, int
    :param d: image depth, int or None
    :return: the sampling grid, same size as grid base
    """
    order = [2, 1, 0] if d else [1, 0]
    sample_dir = sample_dir.repeat((b, h, w, d, 1)) if d else sample_dir.repeat((b, h, w, 1))
    sample_dir_scaled = swap_order(torch.mul(sample_dir, curr_radius), order, dim=-1)
    sample_grid = img_space_to_sample_space(grid_base + sample_dir_scaled, h, w, d)     # convert to [-1, 1]
    return sample_grid, sample_dir_scaled


def calc_dir_response(sample_dir, curr_radius, gradients, basis, grid_base, b, h, w, d):
    """
    Compute the projected response for given direction and radius
    :param sample_dir: sampling direction 2D or 3D vector
    :param curr_radius: radius corresponding to this direction
    :param gradients: image gradients [B, 2, H, W] or [B, 3, H, W, D]
    :param basis: basis for vessel flow of each pixel or voxel [B, H, W, n(2), 2] or [B, H, W, D, n(3), 3]
    :param grid_base: grid base for sampling [B, H, W, 2] or [B, H, W, D, 3]
    :param b: batch size, int
    :param h: image, height, int
    :param w: image width, int
    :param d: image depth, int or None
    :return: projected responses [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    """
    # get sample grid [B, H, W, 2] or [B, H, W, D, 3]
    sample_grid, sample_dir_scaled = get_sample_grid(sample_dir, curr_radius, grid_base, b, h, w, d)
    # projected gradients has shape of [B, H, W, 1, 2] / [B, H, W, D, 1, 3]
    proj_gradients = project(grid_sample(gradients, sample_grid, True), basis, proj=True)
    # compute projected flux
    sample_dir_scaled = proc_sample_dir(sample_dir_scaled, curr_radius)     # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    proj_response = project(proj_gradients, sample_dir_scaled)              # [B, H, W, 2, 2] / [B, H, W, D, 3, 3]
    return proj_response


# --------- Loss Functions ---------
def recon_loss(image, output, sup=False):
    """
    Compute the reconstruction loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: reconstructed image [B, 1, H, W] / [B, 1, H, W, D]
    :param sup: whether supervision is used
    :return: reconstruction loss
    """
    recon = output['recon']
    if sup:
        recon = torch.sigmoid(recon)
    rec_loss = F.mse_loss(image, recon)
    return rec_loss


def flux_loss_symmetry(image, output, sample_num, grad_dims):
    """
    Compute the Symmetry Flux Loss of the output of the network
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param sample_num: num of sampling directions of a sphere / circle
    :param grad_dims: 2D image (2, 3) / 3D image (2, 3, 4)
    :return: flux response, mean flux loss
    """
    b, c, h, w = image.shape[:4]                               # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    d = image.shape[4] if image.dim() == 5 else None
    # 2D image [B, H, W, 2], [B, H, W, 1] / 3D image [B, H, W, D, 3], [B, H, W, D, 1]
    optimal_dir, estimated_r = preproc_output(output)
    basis = get_orthogonal_basis(optimal_dir)                               # get the basis of the optimal directions
    sampling_vec = get_sampling_vec(sample_num, estimated_r)                # get sampling sphere / circle
    gradients = get_gradients(image, dims=grad_dims)                        # [B, 2, H, W] / [B, 3, H, W, D]
    grid_base = sample_space_to_img_space(get_grid_base(gradients), h, w, d)     # convert to [0, H]

    response = torch.zeros(optimal_dir.size(), device=optimal_dir.device)   # get responses of 2 / 3 directions
    for i in range(sample_num):
        sample_dir = sampling_vec[i]                                        # this is a 2d / 3d vector
        curr_radius = estimated_r[..., i:i+1] if estimated_r.size(-1) > 1 else estimated_r
        proj_response = calc_dir_response(sample_dir, curr_radius, gradients, basis, grid_base, b, h, w, d)
        response += proj_response / sample_num                              # [B, H, W, 2] / [B, H, W, D, 3]
    response = - torch.sum(response[..., 1:], dim=-1)                       # [B, H, W] / [B, H, W, D]
    response = torch.clip(response, min=0.0)
    if 'attention' in output.keys():
        response = torch.mul(response, 1.0 + output['attention'])
    max_flux_loss = - response.mean()
    return response, max_flux_loss


def flux_loss_asymmetry(image, output, sample_num, grad_dims):
    """
    Compute the None-Symmetry Flux Loss of the output of the network
    find the minimum magnitude of the direction and the opposite direction
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param sample_num: num of sampling directions of a sphere / circle
    :param grad_dims: 2D image (2, 3) / 3D image (2, 3, 4)
    :return: flux response, mean flux loss
    """
    b, c, h, w = image.shape[:4]                            # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    d = image.shape[4] if image.dim() == 5 else None
    # 2D image [B, H, W, 2], [B, H, W, 1] / 3D image [B, H, W, D, 3], [B, H, W, D, 1]
    optimal_dir, estimated_r = preproc_output(output)
    basis = get_orthogonal_basis(optimal_dir)                               # get the basis of the optimal directions
    sampling_vec = get_sampling_vec(sample_num, estimated_r)                # get sampling sphere / circle
    gradients = get_gradients(image, dims=grad_dims)                        # [B, 2, H, W] / [B, 3, H, W, D]
    grid_base = sample_space_to_img_space(get_grid_base(gradients), h, w, d)     # convert to [0, H]

    shift = int(sample_num / 2)
    response = torch.zeros(optimal_dir.size(), device=optimal_dir.device)   # get responses of 2 / 3 directions
    for i in range(shift):
        # get the sampling direction and the opposite direction
        sample_dir1, sample_dir2 = sampling_vec[i], sampling_vec[i+shift]   # this is a 2d / 3d vector
        curr_rad1 = estimated_r[..., i:i+1] if estimated_r.size(-1) > 1 else estimated_r
        curr_rad2 = estimated_r[..., i+shift:i+shift+1] if estimated_r.size(-1) > 1 else estimated_r
        # compute the projected responses of the two directions [B, H, W, 2] / [B, H, W, D, 3]
        proj_response1 = calc_dir_response(sample_dir1, curr_rad1, gradients, basis, grid_base, b, h, w, d)
        proj_response2 = calc_dir_response(sample_dir2, curr_rad2, gradients, basis, grid_base, b, h, w, d)
        # find the minimum responses of the two directions
        response += torch.maximum(proj_response1, proj_response2) * 2 / sample_num
    response = - torch.sum(response[..., 1:], dim=-1)  # - response[..., 0]    # [B, H, W] / [B, H, W, D]
    response = torch.clip(response, min=0.0)
    if 'attentions' in output.keys():
        attention = output['attentions'][-1]
        response = torch.mul(response, 1.0 + attention)
    mean_flux_loss = - response.mean()
    return response, mean_flux_loss


def continuity_loss(image, output, flux_response, sample_num):
    """
    Compute the continuity loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param flux_response: flux response [B, H, W] / [B, H, W, D]
    :param sample_num: num of sampling directions of a sphere / circle
    :return: mean direction_loss and mean intensity loss
    """
    b, c, h, w = image.shape[:4]                                    # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    d = image.shape[4] if image.dim() == 5 else None
    # 2D image [B, H, W, 2], [B, H, W, 1] / 3D image [B, H, W, D, 3], [B, H, W, D, 1]
    optimal_dir, estimated_r = preproc_output(output)
    mean_rad = torch.mean(estimated_r, dim=-1).unsqueeze(-1)
    optimal_dir_scaled = torch.mul(optimal_dir, mean_rad)
    # get the sample grid on the optimal direction
    order = [2, 1, 0] if d else [1, 0]
    grid_base = sample_space_to_img_space(get_grid_base(image), h, w, d)              # convert to [0, H]
    sample_grid = grid_base + swap_order(optimal_dir_scaled, order, dim=-1)

    # compute the direction loss
    optimal_dir = output['vessel']  # original vessel direction, [B, 2, H, W] / [B, 3, H, W, D]
    sampled_optimal_dir = grid_sample(optimal_dir, sample_grid, permute=False)
    similarity = F.cosine_similarity(optimal_dir, sampled_optimal_dir)
    similarity_low = similarity * 0
    direction_loss = - torch.min(similarity, similarity_low).mean()

    # intensity continuity loss
    intensity_loss = 0.0
    flux_response = flux_response.unsqueeze(1)
    for scale in torch.linspace(-1.0, 1.0, sample_num):
        curr_grid = grid_base + optimal_dir_scaled * scale
        sampled_optimal_response = grid_sample(flux_response, curr_grid, permute=False)
        intensity_loss += F.mse_loss(flux_response, sampled_optimal_response) / sample_num
    return direction_loss, intensity_loss


def attention_loss(output, mean_val):
    att_loss = 0.0
    attentions = output['attentions']
    for i in range(len(attentions)):
        att_loss += torch.pow((attentions[i].mean() - mean_val), 2)
    return att_loss


def supervised_loss(ground_truth, output):
    recon = output['recon']
    recon = torch.sigmoid(recon)
    return F.mse_loss(recon, ground_truth) + F.l1_loss(recon, ground_truth)


def vessel_loss(image, output, loss_config):
    """
    Aggregate all the vessel loss
    :param image: original image [B, 1, H, W] / [B, 1, H, W, D]
    :param output: output dictionary 'vessel', 'radius', 'recon', 'attention'
    :param loss_config: dict, store the loss configurations
    :return: losses: dict, store the losses
    """
    grad_dims = loss_config['grad_dims']
    flux_sample_num = loss_config['flux_sample_num']
    intensity_sample_num = loss_config['intensity_sample_num']
    l_flux = loss_config['lambda_flux']
    l_direction = loss_config['lambda_direction']
    l_intensity = loss_config['lambda_intensity']
    l_recon = loss_config['lambda_recon']

    # get flux loss function configuration
    flux_loss_type = loss_config['flux_loss_type']
    if flux_loss_type == 'asymmetry':
        flux_loss_func = flux_loss_asymmetry
    else:
        flux_loss_func = flux_loss_symmetry

    # calculate losses
    flux_response, optimal_flux_loss = flux_loss_func(image, output, flux_sample_num, grad_dims)
    dir_loss, intense_loss = continuity_loss(image, output, flux_response, intensity_sample_num)
    rec_loss = recon_loss(image, output)

    # assign weights of losses
    dir_loss, intense_loss = dir_loss * l_direction, intense_loss * l_intensity
    optimal_flux_loss, rec_loss = optimal_flux_loss * l_flux, rec_loss * l_recon
    total_loss = optimal_flux_loss + rec_loss + dir_loss + intense_loss

    losses = {
        'flux_loss': optimal_flux_loss,
        'dirs_loss': dir_loss,
        'ints_loss': intense_loss,
        'rcon_loss': rec_loss,
        'total_loss': total_loss
    }
    if 'attentions' in output.keys():
        mean_exposure = loss_config['mean_exp']
        if mean_exposure != 0:
            l_att = loss_config['lambda_attention']
            att_loss = attention_loss(output, mean_exposure) * l_att
            total_loss += att_loss
            losses['attn_loss'] = att_loss
            losses['total_loss'] = total_loss
    return losses


def calc_local_contrast(image, estimated_r, sample_num, scale_steps):
    b, c, h, w = image.shape[:4]                                    # 2D image [B, C, H, W] / 3D image [B, C, H, W, D]
    d = image.shape[4] if image.dim() == 5 else None
    sampling_vec = get_sampling_vec(sample_num, estimated_r)        # get sampling sphere / circle
    inside_scales = torch.linspace(0.1, 1.0, steps=scale_steps)     # multiple scales of inside
    outside_scales = torch.linspace(1.1, 2.0, steps=scale_steps)    # multiple scales of outside
    grid_base = sample_space_to_img_space(get_grid_base(image), h, w, d)     # Convert to [0, H]
    # distinguish the 2D and 3D image
    if d:
        estimated_r = estimated_r.permute(0, 2, 3, 4, 1)                    # [B, H, W, D, 1]
        img_contrast_i = torch.zeros(b, c, h, w, d).to(estimated_r.device)
        img_contrast_o = torch.zeros(b, c, h, w, d).to(estimated_r.device)
    else:
        estimated_r = estimated_r.permute(0, 2, 3, 1)                       # [B, H, W, 1]
        img_contrast_i = torch.zeros(b, c, h, w).to(estimated_r.device)
        img_contrast_o = torch.zeros(b, c, h, w).to(estimated_r.device)

    shift = int(sample_num / 2)
    for i in range(scale_steps):                                    # loop over the sampling scales
        scale_i, scale_o = inside_scales[i], outside_scales[i]
        for j in range(shift):                                      # loop over the sampling directions
            sample_dir1, sample_dir2 = sampling_vec[j], sampling_vec[j+shift]   # this is a 2d / 3d vector
            curr_rad1 = estimated_r[..., j:j+1] if estimated_r.size(-1) > 1 else estimated_r
            curr_rad2 = estimated_r[..., j+shift:j+shift+1] if estimated_r.size(-1) > 1 else estimated_r
            # get sample grids of the inside and outside for both directions
            sample_grid_pos_i, _ = get_sample_grid(sample_dir1, curr_rad1 * scale_i, grid_base, b, h, w, d)
            sample_grid_neg_i, _ = get_sample_grid(sample_dir2, curr_rad2 * scale_i, grid_base, b, h, w, d)
            sample_grid_pos_o, _ = get_sample_grid(sample_dir1, curr_rad1 * scale_o, grid_base, b, h, w, d)
            sample_grid_neg_o, _ = get_sample_grid(sample_dir2, curr_rad2 * scale_o, grid_base, b, h, w, d)
            # sampling intensities
            sampled_img_pos_i = torch.clip(image - grid_sample(image, sample_grid_pos_i, permute=False), min=0.0)
            sampled_img_neg_i = torch.clip(image - grid_sample(image, sample_grid_neg_i, permute=False), min=0.0)
            sampled_img_pos_o = torch.clip(image - grid_sample(image, sample_grid_pos_o, permute=False), min=0.0)
            sampled_img_neg_o = torch.clip(image - grid_sample(image, sample_grid_neg_o, permute=False), min=0.0)
            # adding the image local contrasts
            img_contrast_i += torch.mul(sampled_img_pos_i, sampled_img_neg_i) / sample_num / scale_steps
            img_contrast_o += torch.mul(sampled_img_pos_o, sampled_img_neg_o) / sample_num / scale_steps
    # compute the inside / outside ratio
    # img_contrast_i = torch.mean(img_contrast_i, dim=0)
    # img_contrast_o = torch.mean(img_contrast_o, dim=0)
    epsilon = 3e-2 if d else 1e-4
    img_local_contrast = torch.div(img_contrast_o, img_contrast_i + epsilon) - 1.0
    # regularization or not
    img_local_contrast = torch.sigmoid(img_local_contrast)
    return img_local_contrast
