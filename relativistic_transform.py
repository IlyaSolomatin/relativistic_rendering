#!/usr/bin/env python3
"""
Relativistic Image Transformer

Simulates what a printed picture would look like when approached at near-light speed.
Implements: relativistic aberration, spectral Doppler shift, searchlight effect,
and UV/IR spectral redistribution via a Tier 2 spectral approach.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAMBDA_MIN = 380.0   # nm, visible range start
LAMBDA_MAX = 780.0   # nm, visible range end
LAMBDA_STEP = 1.0    # nm, integration step

# Asymmetric Gaussian basis for RGB spectral reconstruction
# (center_nm, sigma_left_nm, sigma_right_nm)
# Right-side tails are wide to model realistic near-IR reflectance of paper/ink.
RGB_BASIS = [
    (630.0,  50.0, 150.0),   # Red
    (532.0,  45.0, 100.0),   # Green
    (465.0,  35.0,  60.0),   # Blue
]

# Additional broadband IR emission model.
# Printed pictures under D65 illumination reflect significant near-IR.
# We add an IR component proportional to each channel's luminance weight.
IR_CENTER = 1000.0          # nm
IR_SIGMA = 200.0            # nm
IR_AMPLITUDE = 0.5          # relative to RGB Gaussian peaks
IR_LUM_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])   # sRGB luminance

# IEC 61966-2-1 sRGB <-> CIE XYZ (D65)
M_XYZ_TO_SRGB = np.array([
    [ 3.2406255, -1.5372080, -0.4986286],
    [-0.9689307,  1.8757561,  0.0415175],
    [ 0.0557101, -0.2040211,  1.0569959],
])

# ---------------------------------------------------------------------------
# CIE 1931 Color Matching Functions  (Wyman, Sloan & Shirley 2013 fit)
# ---------------------------------------------------------------------------

def _asymmetric_gaussian(x, mu, sigma_left, sigma_right):
    """Asymmetric Gaussian: different sigma on each side of the peak."""
    sigma = np.where(x < mu, sigma_left, sigma_right)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def cie_cmf(wavelengths):
    """CIE 1931 2-degree observer via Wyman et al. (2013) multi-lobe Gaussian fit.

    Args:
        wavelengths: 1-D array in nm.

    Returns:
        (3, N) array of [x_bar, y_bar, z_bar].
    """
    t = wavelengths
    x_bar = (  0.362 * _asymmetric_gaussian(t, 442.0,  0.0624 * 442.0, 0.0374 * 442.0)
             + 1.056 * _asymmetric_gaussian(t, 599.8,  0.0264 * 599.8, 0.0323 * 599.8)
             - 0.065 * _asymmetric_gaussian(t, 474.0,  0.0490 * 474.0, 0.0382 * 474.0))
    y_bar = (  0.821 * _asymmetric_gaussian(t, 568.8,  0.0213 * 568.8, 0.0247 * 568.8)
             + 0.286 * _asymmetric_gaussian(t, 530.9,  0.0613 * 530.9, 0.0322 * 530.9))
    z_bar = (  1.217 * _asymmetric_gaussian(t, 437.0,  0.0845 * 437.0, 0.0278 * 437.0)
             + 0.681 * _asymmetric_gaussian(t, 459.0,  0.0385 * 459.0, 0.0725 * 459.0))
    return np.stack([x_bar, y_bar, z_bar], axis=0)

# ---------------------------------------------------------------------------
# Spectral color-matrix precomputation
# ---------------------------------------------------------------------------

def _ir_basis(wavelengths):
    """Broad Gaussian centered in the near-IR."""
    return np.exp(-0.5 * ((wavelengths - IR_CENTER) / IR_SIGMA) ** 2)


def _compute_M_raw(D, lam_obs, cmf_obs):
    """Spectral transformation matrix for one Doppler factor.

    M_raw[i,j] = D^5 * integral[ B_j(D*lam) * cmf_i(lam) ] dlam

    where B_j is the j-th RGB basis (asymmetric Gaussian + IR tail)
    and the integral is over the observer visible range.
    """
    lam_source = D * lam_obs          # source wavelengths that map to observer lam_obs
    dlam = lam_obs[1] - lam_obs[0]    # integration step
    ir_at_source = _ir_basis(lam_source)  # IR component at source wavelengths

    M = np.zeros((3, 3))
    for j, (center, sig_l, sig_r) in enumerate(RGB_BASIS):
        basis_at_source = _asymmetric_gaussian(lam_source, center, sig_l, sig_r)
        # Add luminance-weighted IR component
        full_basis = basis_at_source + IR_AMPLITUDE * IR_LUM_WEIGHTS[j] * ir_at_source
        for i in range(3):
            M[i, j] = np.sum(full_basis * cmf_obs[i]) * dlam

    return M * (D ** 5)


def precompute_color_matrices(D_min=0.05, D_max=30.0, n_samples=2000):
    """Precompute normalised sRGB colour matrices on a log-spaced D grid.

    Returns (D_grid, C_matrices) where C_matrices has shape (n_samples, 3, 3).
    """
    lam_obs = np.arange(LAMBDA_MIN, LAMBDA_MAX + LAMBDA_STEP / 2, LAMBDA_STEP)
    cmf_obs = cie_cmf(lam_obs)

    # Build a grid that always includes D=1.0 exactly
    D_lo = np.geomspace(D_min, 1.0, n_samples // 2, endpoint=False)
    D_hi = np.geomspace(1.0, D_max, n_samples - n_samples // 2)
    D_grid = np.concatenate([D_lo, D_hi])

    raw = np.empty((len(D_grid), 3, 3))
    for k, D in enumerate(D_grid):
        raw[k] = _compute_M_raw(D, lam_obs, cmf_obs)

    # Convert XYZ -> linear sRGB
    C = np.einsum('ij,kjl->kil', M_XYZ_TO_SRGB, raw)

    # Normalisation: C_norm(D) = C(D) @ inv(C(1))
    idx_one = np.argmin(np.abs(D_grid - 1.0))
    C_one_inv = np.linalg.inv(C[idx_one])
    C_norm = np.einsum('kij,jl->kil', C, C_one_inv)

    return D_grid, C_norm


def interpolate_color_matrix(D_values, D_grid, C_matrices):
    """Interpolate the precomputed 3x3 matrices for arbitrary per-pixel D.

    Args:
        D_values:   (H, W)  Doppler factors.
        D_grid:     (S,)    precomputed grid.
        C_matrices: (S, 3, 3) precomputed matrices.

    Returns:
        (H, W, 3, 3) per-pixel colour matrices.
    """
    shape = D_values.shape
    D_flat = D_values.ravel()
    D_clipped = np.clip(D_flat, D_grid[0], D_grid[-1])

    out = np.empty((*shape, 3, 3))
    for i in range(3):
        for j in range(3):
            interp_fn = interp1d(D_grid, C_matrices[:, i, j],
                                 kind='linear', assume_sorted=True)
            out[..., i, j] = interp_fn(D_clipped).reshape(shape)
    return out

# ---------------------------------------------------------------------------
# sRGB gamma utilities
# ---------------------------------------------------------------------------

def srgb_to_linear(img):
    """Remove sRGB gamma.  Input & output in [0, 1] float64."""
    img = np.asarray(img, dtype=np.float64)
    return np.where(img <= 0.04045,
                    img / 12.92,
                    ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img):
    """Apply sRGB gamma.  Input & output in [0, 1] float64 (clamped)."""
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308,
                    img * 12.92,
                    1.055 * np.power(img, 1.0 / 2.4) - 0.055)

# ---------------------------------------------------------------------------
# Tone mapping
# ---------------------------------------------------------------------------

def tone_map(img, exposure=1.0, method='reinhard'):
    """Tone-map an HDR linear-RGB image to [0, 1].

    Methods
    -------
    reinhard : Reinhard global operator  L / (1 + L)
    log      : logarithmic compression   log(1 + exposure*L) / log(1 + max)
    clamp    : simple [0, 1] clamp
    """
    img = img * exposure
    img = np.maximum(img, 0.0)            # clip negatives (out-of-gamut)

    if method == 'reinhard':
        return img / (1.0 + img)
    elif method == 'log':
        max_val = img.max() or 1.0
        return np.log1p(img) / np.log1p(max_val)
    elif method == 'clamp':
        return np.clip(img, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown tone-map method: {method}")

# ---------------------------------------------------------------------------
# Camera model & aberration
# ---------------------------------------------------------------------------

def compute_pixel_angles(width, height, fov_h_deg=60.0):
    """Pinhole camera: per-pixel (theta_o, phi) and focal length f.

    Returns (theta_o, phi, f) where theta_o, phi are (H, W) arrays.
    """
    f = (width / 2.0) / np.tan(np.radians(fov_h_deg / 2.0))
    cx, cy = width / 2.0, height / 2.0

    jj, ii = np.meshgrid(np.arange(height, dtype=np.float64),
                          np.arange(width, dtype=np.float64),
                          indexing='ij')   # jj = row, ii = col

    dx = ii - cx + 0.5     # +0.5 to sample pixel centres
    dy = jj - cy + 0.5
    r = np.sqrt(dx ** 2 + dy ** 2)

    theta_o = np.arctan2(r, f)
    phi = np.arctan2(dy, dx)

    return theta_o, phi, f


def inverse_aberration(theta_o, beta):
    """Observer angle -> source angle (vectorised).

    Inverse of forward_aberration — maps output pixel direction to source:
    cos(theta_s) = (cos(theta_o) - beta) / (1 - beta * cos(theta_o))

    Source angles are larger than observer angles (picture appears compressed).
    """
    cos_o = np.cos(theta_o)
    cos_s = (cos_o - beta) / (1.0 - beta * cos_o)
    return np.arccos(np.clip(cos_s, -1.0, 1.0))


def compute_doppler_factor(theta_s, beta):
    """Doppler factor D = 1 / (gamma * (1 - beta * cos(theta_s)))."""
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    return 1.0 / (gamma * (1.0 - beta * np.cos(theta_s)))

# ---------------------------------------------------------------------------
# Image remapping
# ---------------------------------------------------------------------------

def compute_source_coordinates(theta_s, phi, f, width, height):
    """Map source angles back to pixel coordinates.

    Returns (src_row, src_col, valid_mask).
    """
    cx, cy = width / 2.0, height / 2.0
    r_s = f * np.tan(theta_s)
    src_col = cx + r_s * np.cos(phi) - 0.5   # -0.5 to undo pixel-centre offset
    src_row = cy + r_s * np.sin(phi) - 0.5

    valid = ((src_col >= 0) & (src_col <= width - 1) &
             (src_row >= 0) & (src_row <= height - 1))
    return src_row, src_col, valid


def remap_image(source_img, src_row, src_col, valid_mask):
    """Bilinear-interpolate source_img at (src_row, src_col).

    source_img: (H, W, 3) float64 linear RGB.
    Returns (H, W, 3) with invalid pixels = 0.
    """
    out = np.zeros_like(source_img)
    coords = np.stack([src_row, src_col], axis=0)   # (2, H, W)
    for c in range(source_img.shape[2]):
        out[..., c] = map_coordinates(source_img[..., c], coords,
                                      order=1, mode='constant', cval=0.0)
    out[~valid_mask] = 0.0
    return out

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def relativistic_transform(input_path, output_path, beta,
                           fov_deg=60.0, exposure=None,
                           tone_map_method='reinhard', show_debug=False):
    """Apply all relativistic effects and save the result."""

    # --- Load ----------------------------------------------------------------
    src_pil = Image.open(input_path).convert('RGB')
    src = np.asarray(src_pil, dtype=np.float64) / 255.0
    src_linear = srgb_to_linear(src)
    H, W = src_linear.shape[:2]
    print(f"Loaded {input_path}  ({W}x{H}),  beta = {beta}")

    # --- Geometry (aberration) -----------------------------------------------
    theta_o, phi, f = compute_pixel_angles(W, H, fov_deg)
    theta_s = inverse_aberration(theta_o, beta)
    src_row, src_col, valid = compute_source_coordinates(theta_s, phi, f, W, H)
    remapped = remap_image(src_linear, src_row, src_col, valid)

    # --- Doppler factor per pixel --------------------------------------------
    D = compute_doppler_factor(theta_s, beta)

    # --- Spectral colour transform -------------------------------------------
    print("Precomputing spectral colour matrices …")
    D_grid, C_matrices = precompute_color_matrices()
    print("Interpolating per-pixel colour matrices …")
    C_pixel = interpolate_color_matrix(D, D_grid, C_matrices)

    # Apply: out_rgb = C(D) @ in_rgb   per pixel
    result = np.einsum('hwij,hwj->hwi', C_pixel, remapped)

    # --- Tone mapping --------------------------------------------------------
    D_center = compute_doppler_factor(np.array(0.0), beta)
    if exposure is None:
        # Content-aware auto-exposure: target the 99th percentile to ~2.0
        peak = np.percentile(result[result > 0], 99) if np.any(result > 0) else 1.0
        exposure = 2.0 / max(peak, 1e-10)
    print(f"D_center = {float(D_center):.4f},  exposure = {exposure:.6g}")

    result = tone_map(result, exposure=exposure, method=tone_map_method)

    # --- Encode & save -------------------------------------------------------
    result_srgb = linear_to_srgb(result)
    out_uint8 = np.clip(result_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    Image.fromarray(out_uint8).save(output_path)
    print(f"Saved {output_path}")

    # --- Optional debug visualisation ----------------------------------------
    if show_debug:
        _show_debug(src, remapped, D, result_srgb, beta)


def _show_debug(src, remapped, D, final, beta):
    """Matplotlib 2x2 debug grid."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Relativistic Transform  (β = {beta})', fontsize=14)

    axes[0, 0].imshow(src)
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')

    # Aberration only (just the geometry warp, original colours)
    aberr_display = np.clip(remapped, 0, 1)
    aberr_display = linear_to_srgb(aberr_display)
    axes[0, 1].imshow(aberr_display)
    axes[0, 1].set_title('Aberration only')
    axes[0, 1].axis('off')

    im = axes[1, 0].imshow(D, cmap='plasma')
    axes[1, 0].set_title('Doppler factor D')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(np.clip(final, 0, 1))
    axes[1, 1].set_title('Final output')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Relativistic image transformer — simulate approaching '
                    'a picture at near-light speed.')
    p.add_argument('input', help='Path to input image')
    p.add_argument('-o', '--output', default=None,
                   help='Output path (default: <input>_relativistic.png)')
    p.add_argument('-b', '--beta', type=float, default=0.5,
                   help='Velocity as fraction of c  (default: 0.5)')
    p.add_argument('--fov', type=float, default=60.0,
                   help='Horizontal FOV in degrees  (default: 60)')
    p.add_argument('--exposure', type=float, default=None,
                   help='Manual exposure (default: auto)')
    p.add_argument('--tone-map', choices=['reinhard', 'log', 'clamp'],
                   default='reinhard', dest='tone_map',
                   help='Tone-mapping method  (default: reinhard)')
    p.add_argument('--debug', action='store_true',
                   help='Show matplotlib debug visualisation')
    args = p.parse_args(argv)

    if not 0.0 <= args.beta < 1.0:
        p.error('beta must be in [0, 1)')

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f'{stem}_relativistic.png'

    return args


if __name__ == '__main__':
    args = parse_args()
    relativistic_transform(
        input_path=args.input,
        output_path=args.output,
        beta=args.beta,
        fov_deg=args.fov,
        exposure=args.exposure,
        tone_map_method=args.tone_map,
        show_debug=args.debug,
    )
