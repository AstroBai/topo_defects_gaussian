import jax.numpy as np
import jax
from jax import jacfwd
import matplotlib.pyplot as plt
from tqdm import tqdm

def lens_potential(pos, epsilon=0.3):
    """Lens potential function taking position vector."""
    x, y = pos
    return -np.exp(-x**2/(1+epsilon) - y**2/(1-epsilon))


x_1d = np.linspace(-2, 2, 51)
y_1d = np.linspace(-2, 2, 51)
x, y = np.meshgrid(x_1d, y_1d)


positions = np.stack([x.flatten(), y.flatten()], axis=1)


hessian_fn = jacfwd(jacfwd(lens_potential))
hessians = jax.vmap(hessian_fn)(positions)


d2psi_dx2 = hessians[:, 0, 0].reshape(x.shape)
d2psi_dy2 = hessians[:, 1, 1].reshape(x.shape)
d2psi_dxdy = hessians[:, 0, 1].reshape(x.shape)


kappa = 0.5 * (d2psi_dx2 + d2psi_dy2)
gamm1 = 0.5 * (d2psi_dx2 - d2psi_dy2)
gamm2 = d2psi_dxdy


phi = 0.5 * np.arctan2(gamm2, gamm1)
ux = np.cos(phi)
uy = np.sin(phi)




def delta_phi(phi1, phi2):
    """Compute the difference between two angles, handling periodicity."""
    if np.abs(phi2-phi1) <= np.pi/2:
        beta = 0
    elif phi2-phi1 > np.pi/2:
        beta = -np.pi
    else:  # phi2-phi1 < -pi/2
        beta = np.pi
    dphi = phi2 - phi1 + beta
    return dphi


def winding_number(pix, phi, ngrid):
    # set winding number to zero if the pixel is at the boundary
    if pix[0] == 0 or pix[0] == ngrid-1 or pix[1] == 0 or pix[1] == ngrid-1:
        return 0
    # get the 8 neighboring pixels in counter-clockwise order
    neighbors = [
        (pix[0]-1, pix[1]),     # up
        (pix[0]-1, pix[1]+1),   # up-right
        (pix[0], pix[1]+1),     # right
        (pix[0]+1, pix[1]+1),   # down-right
        (pix[0]+1, pix[1]),     # down
        (pix[0]+1, pix[1]-1),   # down-left
        (pix[0], pix[1]-1),     # left
        (pix[0]-1, pix[1]-1)    # up-left
    ]
    # compute the total change in angle around the pixel
    total_dphi = 0.0
    for i in range(len(neighbors)): 
        phi1 = phi[neighbors[i]]
        phi2 = phi[neighbors[(i+1) % len(neighbors)]]
        dphi = delta_phi(phi1, phi2)
        total_dphi += dphi
    # determine winding number
    winding_num = total_dphi / (2 * np.pi)
    return winding_num

def find_defects(phi, ngrid, shear_amp):
    defects = {}
    for i in tqdm(range(ngrid), desc="Finding defects"):
        for j in range(ngrid):
            pix = (i, j)
            wnum = winding_number(pix, phi, ngrid)
            if abs(wnum) >= 0.5:
                defects[pix] = wnum
    #if multiple defects with same charge are found in neighboring pixels, keep only one, moving a mask of 3*3 pixels
    # Sort defects by shear amplitude (ascending) to keep the one with minimum shear
    sorted_defects = sorted(defects.keys(), key=lambda p: shear_amp[p])

    clustered_defects = {}
    visited = set()
    for pix in sorted_defects:
        if pix in visited:
            continue
        clustered_defects[pix] = defects[pix]
        # mark neighbors as visited
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                neighbor = (pix[0]+di, pix[1]+dj)
                visited.add(neighbor)

    return clustered_defects

shear_amp = np.sqrt(gamm1**2 + gamm2**2)
defect_dict = find_defects(phi, x.shape[0], shear_amp)
print(f"Found {len(defect_dict)} defects:")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32/3, 4))

im1 = ax1.imshow(kappa, extent=[-2, 2, -2, 2], cmap='RdYlBu_r', origin='lower')
plt.colorbar(im1, ax=ax1, label=r'Convergence $\kappa$')
step = 2
ax1.quiver(x[::step, ::step], y[::step, ::step], 
          ux[::step, ::step], uy[::step, ::step], 
          angles='xy', scale_units='xy',
          color='white', width=0.003, alpha=0.9,
          headwidth=0, headlength=0, headaxislength=0, pivot='mid')
# scatter defect locations
for pix, wnum in defect_dict.items():
    if wnum > 0:
        ax1.scatter(x[pix], y[pix], c='red', marker='o')
    elif wnum < 0:
        ax1.scatter(x[pix], y[pix], c='blue', marker='o')
ax1.set_title('Convergence with Shear Orientation')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xticks([-2, -1, 0, 1, 2])
ax1.set_yticks([-2, -1, 0, 1, 2])
ax1.set_aspect('equal')

shear_amp = np.sqrt(gamm1**2 + gamm2**2)
im2 = ax2.imshow(shear_amp, extent=[-2, 2, -2, 2], cmap='plasma', origin='lower')
plt.colorbar(im2, ax=ax2, label=r'Shear Amplitude $|\gamma|$')
ax2.quiver(x[::step, ::step], y[::step, ::step], 
          ux[::step, ::step], uy[::step, ::step], 
          angles='xy', scale_units='xy',
          color='white', width=0.003, alpha=0.9,
          headwidth=0, headlength=0, headaxislength=0, pivot='mid')
# scatter defect locations
for pix, wnum in defect_dict.items():
    if wnum > 0:
        ax2.scatter(x[pix], y[pix], c='red', marker='o')
    elif wnum < 0:
        ax2.scatter(x[pix], y[pix], c='blue', marker='o')
ax2.set_title('Shear Amplitude with Orientation')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xticks([-2, -1, 0, 1, 2])
ax2.set_yticks([-2, -1, 0, 1, 2])
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('outfiles/test_shear_field.png', dpi=300, bbox_inches='tight')