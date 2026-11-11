import torch

def compute_patch_overlap(size, has_left, has_right, base=8):

    if has_left and has_right:

        left = base
        right = base

        total = size + left + right
        mod = total % 64

        if mod != 0:
            extra = 64 - mod
            left += extra // 2
            right += extra - extra // 2

    elif has_left:

        left = base
        total = size + left
        mod = total % 64

        if mod != 0:
            left += 64 - mod

        right = 0

    elif has_right:

        right = base
        total = size + right
        mod = total % 64

        if mod != 0:
            right += 64 - mod

        left = 0

    else:
        left = right = 0

    return left, right

def build_patch_grid(H, W, n_rows, n_cols):


    h_list = [H // n_rows] * n_rows
    h_list[-1] += H - sum(h_list)

    w_list = [W // n_cols] * n_cols
    w_list[-1] += W - sum(w_list)

    hs = [0]
    for h in h_list:
        hs.append(hs[-1] + h)

    ws = [0]
    for w in w_list:
        ws.append(ws[-1] + w)

    patches = []
    patch_meta = []

    for i in range(n_rows):
        for j in range(n_cols):

            h0, h1 = hs[i], hs[i+1]
            w0, w1 = ws[j], ws[j+1]

            ph = h1 - h0
            pw = w1 - w0


            top, bottom = compute_patch_overlap(
                ph,
                has_left=(i > 0),
                has_right=(i < n_rows - 1)
            )

            left, right = compute_patch_overlap(
                pw,
                has_left=(j > 0),
                has_right=(j < n_cols - 1)
            )

            hs_slice = slice(h0 - top, h1 + bottom)
            ws_slice = slice(w0 - left, w1 + right)

            patches.append((hs_slice, ws_slice))
            patch_meta.append((top, bottom, left, right))

    return patches, patch_meta

def merge_patch_grid(image_shape, patches, enhanced_patches, patch_meta, device='cpu'):
    output = torch.zeros(image_shape, dtype=torch.float32, device=device)

    weight = torch.zeros((1, 1, image_shape[2], image_shape[3]), dtype=torch.float32, device=device)

    for (hs, ws), enhanced_patch, (top, bottom, left, right) in zip(patches, enhanced_patches, patch_meta):
        enhanced_patch = enhanced_patch.to(device)

        ph, pw = enhanced_patch.shape[2:]
        mask = torch.ones((1, 1, ph, pw), dtype=torch.float32, device=device)

        if left > 0:
            ramp = torch.linspace(0, 1, left, device=device).view(1, 1, 1, -1)
            mask[:,:,:, :left] *= ramp

        if right > 0:
            ramp = torch.linspace(1, 0, right, device=device).view(1, 1, 1, -1)
            mask[:,:,:, -right:] *= ramp

        if top > 0:
            ramp = torch.linspace(0, 1, top, device=device).view(1, 1, -1, 1)
            mask[:,:, :top, :] *= ramp

        if bottom > 0:
            ramp = torch.linspace(1, 0, bottom, device=device).view(1, 1, -1, 1)
            mask[:,:, -bottom:, :] *= ramp

        output[:, :, hs, ws] += enhanced_patch * mask
        weight[:, :, hs, ws] += mask

    output.div_(weight)
    return output
