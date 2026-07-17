import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# grid
# -----------------------------------
N = 500
x = np.linspace(-3, 3, N)
y = np.linspace(-2.5, 2.5, N)
X, Y = np.meshgrid(x, y)

# -----------------------------------
# Construct two kinds of shear field
#
# Left:
#   F-only (banana / closed loop)
#
# Right:
#   G-only (Y-shape / 3-branch)
#
# by combining analytic lens potentials
# -----------------------------------

# =====================================================
# CASE 1 : F-only
#
# lens potential:
#
# ψ_F = r^3 cos(theta)
#
# gives strong first-flexion-like elongation
# and tangential shear structure
# =====================================================

R = np.sqrt(X**2 + Y**2) + 1e-8
Theta = np.arctan2(Y, X)

psi_F = R**3 * np.cos(Theta)

# second derivatives
dx = x[1] - x[0]
dy = y[1] - y[0]

psi_x = np.gradient(psi_F, dx, axis=1)
psi_y = np.gradient(psi_F, dy, axis=0)

psi_xx = np.gradient(psi_x, dx, axis=1)
psi_yy = np.gradient(psi_y, dy, axis=0)
psi_xy = np.gradient(psi_x, dy, axis=0)

gamma1_F = 0.5 * (psi_xx - psi_yy)
gamma2_F = psi_xy


# =====================================================
# CASE 2 : G-only
#
# lens potential:
#
# ψ_G = r^3 cos(3θ)
#
# produces characteristic trefoil / Y-shape
# =====================================================

psi_G = R**3 * np.cos(3 * Theta) / 3

psi_x = np.gradient(psi_G, dx, axis=1)
psi_y = np.gradient(psi_G, dy, axis=0)

psi_xx = np.gradient(psi_x, dx, axis=1)
psi_yy = np.gradient(psi_y, dy, axis=0)
psi_xy = np.gradient(psi_x, dy, axis=0)

gamma1_G = 0.5 * (psi_xx - psi_yy)
gamma2_G = psi_xy


# -----------------------------------
# plot function
# -----------------------------------

def plot_shear(ax, g1, g2, title, defect_finder=None):
    """
    shear stick plot with optional defect markers

    orientation:
        alpha = 1/2 arctan2(g2, g1)
    """

    alpha = 0.5 * np.arctan2(g2, g1)

    U = np.cos(alpha)
    V = np.sin(alpha)

    step = 48

    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        pivot='mid',
        headwidth=0,
        headlength=0,
        headaxislength=0,
        scale=20,
        linewidth=1.5,
        color='black'
    )

    # Add defect markers if DefectFinder is provided
    if defect_finder is not None:
        winding_numbers = defect_finder.winding_number()
        rows, cols = np.where(np.abs(winding_numbers) >= 0.5)
        cmap10 = plt.get_cmap('tab10')
        
        for r, c in zip(rows, cols):
            wn = winding_numbers[r, c]
            # Convert grid indices to physical coordinates
            x_coord = x[c]
            y_coord = y[r]
            
            # Positive charge (winding number > 0): tab10 color 0
            # Negative charge (winding number < 0): tab10 color 3
            if wn > 0:
                marker_color = cmap10(0)
                marker_label = '+'
            else:
                marker_color = cmap10(3)
                marker_label = '-'

            ax.scatter(
                x_coord,
                y_coord,
                s=200,
                marker='o',
                facecolors=marker_color,
                edgecolors='white',
                linewidths=1.5,
                alpha=0.9,
                zorder=3,
            )
            ax.text(
                x_coord,
                y_coord,
                marker_label,
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                fontweight='bold',
                zorder=4,
            )

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12)



# -----------------------------------
# Create DefectFinder objects for both cases
# -----------------------------------

def create_defect_finder(gamma1, gamma2):
    """Create a DefectFinder object for a given shear field."""
    # Calculate shear angles
    shear_angles = 0.5 * np.arctan2(gamma2, gamma1)
    shear_amp = np.sqrt(gamma1**2 + gamma2**2)
    
    # Create mock data structure compatible with DefectFinder
    class MockDefectFinder:
        def __init__(self, phi, gamm1, gamm2):
            self.phi = phi
            self.gamm1 = gamm1
            self.gamm2 = gamm2
            self.shape = phi.shape
            self.is_flat = True
            
        def delta_phi(self, phi1s, phi2s):
            """Compute the difference between two angles, handling periodicity."""
            diffs = phi2s - phi1s
            betas = np.zeros_like(diffs)
            betas[diffs > np.pi/2] = -np.pi
            betas[diffs < -np.pi/2] = np.pi
            dphis = diffs + betas
            return dphis

        def winding_number(self, pixs=None):
            """Compute winding number for 2D flat sky map."""
            rows, cols = self.shape
            
            # Define 8 neighbors in counter-clockwise order
            neighbor_offsets = [
                (0, 1), (1, 1), (1, 0), (1, -1),
                (0, -1), (-1, -1), (-1, 0), (-1, 1)
            ]
            
            # Pad with NaNs to handle boundaries
            padded_phi = np.pad(self.phi, 1, mode='constant', constant_values=np.nan)
            
            neighbors = []
            for dr, dc in neighbor_offsets:
                n_view = padded_phi[1+dr : rows+1+dr, 1+dc : cols+1+dc]
                neighbors.append(n_view)
                
            dphi_total = np.zeros(self.shape)
            
            for i in range(8):
                n1 = neighbors[i]
                n2 = neighbors[(i+1)%8]
                dphis = self.delta_phi(n1, n2)
                dphi_total += dphis
                
            wns = dphi_total / (2 * np.pi)
            wns = np.round(wns * 2) / 2
            
            # Set boundary/invalid pixels to 0
            wns[np.isnan(wns)] = 0
            
            return wns
    
    return MockDefectFinder(shear_angles, gamma1, gamma2)

# =====================================================
# CASE 3: Combined ψ1 + 0.5*ψ2
# =====================================================

psi_combined_1 = gamma1_F + 0.5 * gamma1_G
psi_combined_2 = gamma2_F + 0.5 * gamma2_G


# =====================================================
# CASE 4: Combined 0.5*ψ1 + ψ2
# =====================================================

psi_combined_3 = 0.5 * gamma1_F + gamma1_G
psi_combined_4 = 0.5 * gamma2_F + gamma2_G


# Create DefectFinder objects for all cases
defect_finder_F = create_defect_finder(gamma1_F, gamma2_F)
defect_finder_G = create_defect_finder(gamma1_G, gamma2_G)
defect_finder_combined_1 = create_defect_finder(psi_combined_1, psi_combined_2)
defect_finder_combined_2 = create_defect_finder(psi_combined_3, psi_combined_4)

# -----------------------------------
# figure
# -----------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(6, 6),
    facecolor='white'
)

plot_shear(axes[0, 0], gamma1_F, gamma2_F, r"$\psi_1 = r^3 \cos(\theta) $ ($\mathcal{F}=4,$ $\mathcal{G}=0$)", defect_finder_F)
plot_shear(axes[0, 1], gamma1_G, gamma2_G, r"$\psi_2 = \frac{1}{3} r^3 \cos(3\theta)$ ($\mathcal{F}=0,$ $\mathcal{G}=4$)", defect_finder_G)
plot_shear(axes[1, 0], psi_combined_1, psi_combined_2, r"$\psi_3 = \psi_1 + 0.5\psi_2$ ($\mathcal{F}=4,$ $\mathcal{G}=2$)", defect_finder_combined_1)
plot_shear(axes[1, 1], psi_combined_3, psi_combined_4, r"$\psi_4 = 0.5\psi_1 + \psi_2$ ($\mathcal{F}=2,$ $\mathcal{G}=4$)", defect_finder_combined_2)

plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.02, right=0.98, top=0.98, bottom=0.02)
plt.savefig("flexion_demo.png", dpi=300)
print("Saved to flexion_demo.png")