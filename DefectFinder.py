import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
import treecorr
import glass
from scipy.interpolate import interp1d

class DefectFinder:
    """Class to find topological defects in shear orientation field."""

    def __init__(self, data_path):
        """Initialize the DefectFinder with shear orientation data."""
        data = np.load(data_path)
        self.kappa_bar_unsmoothed = data['kappa_bar_unsmoothed']
        self.kappa_bar = data['kappa_bar']
        self.gamm1_bar = data['gamm1_bar']
        self.gamm2_bar = data['gamm2_bar']
        self.phi_bar = data['phi_bar']
        self.shear_amp = np.sqrt(self.gamm1_bar**2 + self.gamm2_bar**2)
        self.nside = hp.npix2nside(len(self.phi_bar))
        resolution = hp.nside2resol(self.nside, arcmin=True)
        #print(f"Loaded data with nside={self.nside}, resolution={resolution:.2f} arcmin")

    def load_defects(self, defect_datapath):
        """Load defects from a saved npz file."""
        data = np.load(defect_datapath, allow_pickle=True)
        self.defects = data['defects'].item()
        #print(f"Loaded {len(self.defects)} defects from {defect_datapath}")

    def delta_phi(self, phi1s, phi2s):
        """Compute the difference between two angles, handling periodicity."""
        diffs = phi2s - phi1s
        betas = np.zeros_like(diffs)
        betas[diffs > np.pi/2] = -np.pi
        betas[diffs < -np.pi/2] = np.pi
        dphis = diffs + betas
        return dphis

    def winding_number(self, pixs):
        """Compute the total delta phi around the 8 neighbors of a pixel, add up counter-clockwise."""
        # set winding number to zero if the pixel is at the boundary
        nside = hp.npix2nside(len(self.phi_bar))
        # if pix < nside or pix >= 12 * nside**2 - nside:
        #     return 0
        neighbors = hp.get_all_neighbours(nside, pixs) # 8xn_pixs array
        # remove invalid neighbors (at the boundary)
        valid_cols = np.all(neighbors >= 0, axis=0)
        valid_pixs = pixs[valid_cols]
        valid_neighbors = neighbors[:, valid_cols]
        dphi_totals = np.zeros(len(pixs))
        for i in range(8):
            pix1 = valid_neighbors[i,:]
            pix2 = valid_neighbors[(i+1) % 8,:]
            dphis = self.delta_phi(self.phi_bar[pix1], self.phi_bar[pix2])
            dphi_totals[valid_cols] += dphis
        wns = dphi_totals / (2 * np.pi)  # in units of 2 pi
        # avoid numerical fluctuations around half-integers
        wns = np.round(wns * 2) / 2
        return wns

    def find_defects(self, savepath=None, cluster=True):
        """Find defects in the shear orientation field phi_bar."""
        # store defect positions and winding numbers in a dictionary
        self.defects = {}
        npix = len(self.phi_bar)
        wnums = self.winding_number(np.arange(npix))
        defect_indices = np.where(np.abs(wnums) >= 0.5)[0]
        self.defects = dict(zip(defect_indices, wnums[defect_indices]))
        if cluster:
            #if multiple defects with same charge are found in neighboring pixels, keep only one, moving a mask of 3*3 pixels
            defect_pixels = list(self.defects.keys())
            # Sort defects by shear amplitude (ascending)
            defect_pixels.sort(key=lambda p: self.shear_amp[p])

            clustered_defects = {}
            visited = set()
            for pix in defect_pixels:
                if pix in visited:
                    continue
                wnum = self.defects[pix]
                clustered_defects[pix] = wnum
                neighbors = hp.get_all_neighbours(self.nside, pix)
                for neighbor in neighbors:
                    visited.add(neighbor)
            self.defects = clustered_defects
        if savepath:
            np.savez(savepath, defects=self.defects)
        #print(f"Found {len(self.defects)} defects out of {npix} pixels, saving to {savepath}")
        #print(f"Defect Probability: {len(self.defects)/npix:.2e}")
        return self.defects
        

    def save_defects(self, savepath):
        """Save the defects dictionary to a fits file."""
        # save different files for different charges, coord: RA, DEC
        charges = set(self.defects.values())
        for charge in charges:
            ra_list = []
            dec_list = []
            for pix in self.defects.keys():
                wnum = self.defects[pix]
                if wnum == charge:
                    theta, phi = hp.pix2ang(self.nside, pix)
                    ra = phi * 180/np.pi
                    dec = (np.pi/2 - theta) * 180/np.pi  # convert to declination
                    ra_list.append(ra)
                    dec_list.append(dec)
            ra_array = np.array(ra_list)
            dec_array = np.array(dec_list)
            col1 = fits.Column(name='RA', array=ra_array, format='E')
            col2 = fits.Column(name='DEC', array=dec_array, format='E')
            cols = fits.ColDefs([col1, col2])
            hdu = fits.BinTableHDU.from_columns(cols)
            hdu.writeto(f"{savepath}_charge_{charge:+.1f}.fits", overwrite=True)
        

    def plot_defects(self, savepath=None, defect_datapath=None):
        """Plot the defects on the convergence map."""
        if defect_datapath:
            data = np.load(defect_datapath, allow_pickle=True)
            self.defects = data['defects'].item()
        hp.mollview(self.kappa_bar, title='Convergence Map with Defects', cmap='RdYlBu_r', hold=True)
        positive_array = []
        negative_array = []
        for pix in tqdm(self.defects.keys(), desc="Plotting defects"):
            wnum = self.defects[pix]
            theta, phi = hp.pix2ang(self.nside, pix)
            if wnum > 0:
                positive_array.append((theta, phi))
            else:
                negative_array.append((theta, phi))
        hp.projscatter(*zip(*positive_array), s=0.1, c='r', label='+ Defect')
        hp.projscatter(*zip(*negative_array), s=0.1, c='k', label='- Defect')
        plt.legend()
        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches='tight')

    def plot_patch(self, loc, size_arcmin, savepath=None, resolution=128):
        """Plot a square patch of the shear and convergence field around a given location."""
        # Visualize a square patch of the shear and convergence field
        field_size_arcmin = size_arcmin
        field_size_rad = np.deg2rad(field_size_arcmin / 60.0)  # transform to radians

        # Choose the center of the patch (can be adjusted)
        center_theta = loc[0]
        center_phi = loc[1]
        
        # Create a denser version
        fig, ax = plt.subplots(figsize=(12, 12))

        # Higher resolution grid
        n_grid_hires = resolution
        x_range_hires = np.linspace(-field_size_rad/2, field_size_rad/2, n_grid_hires)
        y_range_hires = np.linspace(-field_size_rad/2, field_size_rad/2, n_grid_hires)
        X_hires, Y_hires = np.meshgrid(x_range_hires, y_range_hires)

        theta_grid_hires = center_theta + Y_hires
        phi_grid_hires = center_phi + X_hires / np.sin(center_theta)
        theta_grid_hires = np.clip(theta_grid_hires, 0, np.pi)
        phi_grid_hires = phi_grid_hires % (2*np.pi)

        pixels_hires = hp.ang2pix(self.nside, theta_grid_hires, phi_grid_hires)
        kappa_patch_hires = self.kappa_bar[pixels_hires]
        gamm1_patch_hires = self.gamm1_bar[pixels_hires]
        gamm2_patch_hires = self.gamm2_bar[pixels_hires]
        phi_patch_hires = 0.5 * np.arctan2(gamm2_patch_hires, gamm1_patch_hires)
        shear_amplitude_hires = np.sqrt(gamm1_patch_hires**2 + gamm2_patch_hires**2)

        # Background
        im = ax.imshow(kappa_patch_hires, extent=[-field_size_arcmin/2, field_size_arcmin/2, 
                                                 -field_size_arcmin/2, field_size_arcmin/2],
                       cmap='RdYlBu_r', origin='lower')

        # Denser vectors
        vector_step_dense = 2
        x_sample_dense = X_hires[::vector_step_dense, ::vector_step_dense] * 180/np.pi * 60
        y_sample_dense = Y_hires[::vector_step_dense, ::vector_step_dense] * 180/np.pi * 60
        phi_sample_dense = phi_patch_hires[::vector_step_dense, ::vector_step_dense]
        shear_amp_dense = 1 
        u_sample_dense = np.cos(phi_sample_dense)
        v_sample_dense = np.sin(phi_sample_dense)
        # Vector length scaling
        scale_factor_dense = shear_amp_dense / np.max(shear_amp_dense) * 6
        u_sample_dense *= scale_factor_dense
        v_sample_dense *= scale_factor_dense    
        # Plot vectors
        ax.quiver(x_sample_dense, y_sample_dense, u_sample_dense, v_sample_dense,
                  angles='xy', scale_units='xy',
                  color='white', width=0.001,
                  headwidth=0, headlength=0, headaxislength=0)
        # scatter defects in this patch
        positive_int_array = []
        negative_int_array = []
        positive_half_array = []
        negative_half_array = []
        for pix in tqdm(self.defects.keys(), desc="Plotting defects in patch"):
            wnum = self.defects[pix]
            theta_defect, phi_defect = hp.pix2ang(self.nside, pix)
            x_defect = (phi_defect - center_phi) * np.sin(center_theta) * 180/np.pi * 60
            # for those x_defect > pi * sin(center_theta) * 180/pi * 60, shift to negative side
            if x_defect > field_size_arcmin/2:
                x_defect -= 360 * np.sin(center_theta) * 60
            y_defect = (theta_defect - center_theta) * 180/np.pi * 60
            if wnum == 1:
                positive_int_array.append((x_defect, y_defect))
            elif wnum == -1:
                negative_int_array.append((x_defect, y_defect))
            elif wnum == 0.5:
                positive_half_array.append((x_defect, y_defect))
            elif wnum == -0.5:
                negative_half_array.append((x_defect, y_defect))
            else:
                print(f"Warning: unexpected winding number {wnum} at pixel {pix}")
        if len(positive_int_array) > 0:
            ax.scatter(*zip(*positive_int_array), s=30, c='r', marker='*', label='+1 Defect')
        if len(negative_int_array) > 0:
            ax.scatter(*zip(*negative_int_array), s=30, c='k', marker='*', label='-1 Defect')
        if len(positive_half_array) > 0:
            ax.scatter(*zip(*positive_half_array), s=3, c='r', marker='o', label='+0.5 Defect')
        if len(negative_half_array) > 0:
            ax.scatter(*zip(*negative_half_array), s=3, c='k', marker='o', label='-0.5 Defect')

        ax.set_xlabel('RA offset [arcmin]', fontsize=14)
        ax.set_ylabel('Dec offset [arcmin]', fontsize=14)
        ax.set_title(f'High-Resolution {field_size_arcmin}x{field_size_arcmin} arcmin² Shear Field', fontsize=16)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Convergence κ', fontsize=14)
        ax.set_aspect('equal')
        plt.xlim(-field_size_arcmin/2, field_size_arcmin/2)
        plt.ylim(-field_size_arcmin/2, field_size_arcmin/2)
        plt.legend()
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
        else:
            plt.savefig("outfiles/shear_field_patch_hires.png", dpi=300, bbox_inches='tight')

    def correlation_function(self, charge1_path, charge2_path, min_sep=1., max_sep=100., n_bins=10):
        """Compute the two-point correlation function of defects with given charges"""
        cat1 = treecorr.Catalog(charge1_path,ra_col='RA',dec_col='DEC',ra_units='degrees',dec_units='degrees')
        cat2 = treecorr.Catalog(charge2_path,ra_col='RA',dec_col='DEC',ra_units='degrees',dec_units='degrees')
        if charge1_path == charge2_path:
            # create random catalog, numbers should be 100x larger than data catalog
            #print("Computing auto-correlation")
            n_random = cat1.nobj * 100
            ra_random = np.random.uniform(0, 360, n_random)
            dec_random = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_random)))  # uniform in sin(dec)
            cat_rand = treecorr.Catalog(ra=ra_random, dec=dec_random, ra_units='degrees', dec_units='degrees')
            dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
            rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
            dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
            dd.process(cat1, cat1)
            rr.process(cat_rand, cat_rand)
            dr.process(cat1, cat_rand)
            thetas = dd.meanr
            xi, varxi = dd.calculateXi(rr=rr, dr=dr)
            covmat = dd.cov
            return thetas, xi, varxi, covmat
        
        else:
            # create random catalogs for both
            #print("Computing cross-correlation")
            n_random1 = cat1.nobj * 100
            ra_random1 = np.random.uniform(0, 360, n_random1)
            dec_random1 = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_random1)))  # uniform in sin(dec)
            cat_rand1 = treecorr.Catalog(ra=ra_random1, dec=dec_random1, ra_units='degrees', dec_units='degrees')
            n_random2 = cat2.nobj * 100
            ra_random2 = np.random.uniform(0, 360, n_random2)
            dec_random2 = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_random2)))  # uniform in sin(dec)
            cat_rand2 = treecorr.Catalog(ra=ra_random2, dec=dec_random2, ra_units='degrees', dec_units='degrees')
            dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
            rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
            d1r2 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
            r1d2 = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
            dd.process(cat1, cat2)
            rr.process(cat_rand1, cat_rand2)
            d1r2.process(cat1, cat_rand2)
            r1d2.process(cat2, cat_rand1)
            thetas = dd.meanr
            xi, varxi = dd.calculateXi(rr=rr, dr=d1r2, rd=r1d2)
            covmat = dd.cov
            return thetas, xi, varxi, covmat

    def nearest_distance(self, charge, k=1):
        """Compute the nearest neighbor distance distribution for defects with given charge. Unit: arcmin."""
        ra = []
        dec = []
        for pix in self.defects.keys():
            wnum = self.defects[pix]
            if wnum == charge:
                theta, phi = hp.pix2ang(self.nside, pix)
                ra.append(phi * 180/np.pi)
                dec.append((np.pi/2 - theta) * 180/np.pi)  # convert to declination
        ra = np.array(ra)
        dec = np.array(dec)
        from sklearn.neighbors import NearestNeighbors
        coords = np.vstack((ra, dec)).T
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        distances = distances * 60  # convert to arcmin
        nearest_distances = distances[:, k]  # skip the first column (distance to itself)
        
        return nearest_distances

    def nearest_opposite_distance(self, charge, k=1):
        """Compute the nearest neighbor distance distribution between +1/2 and -1/2 defects."""
        ra_pos = []
        dec_pos = []
        ra_neg = []
        dec_neg = []
        for pix in self.defects.keys():
            wnum = self.defects[pix]
            theta, phi = hp.pix2ang(self.nside, pix)
            if wnum == 0.5:
                ra_pos.append(phi * 180/np.pi)
                dec_pos.append((np.pi/2 - theta) * 180/np.pi)  # convert to declination
            elif wnum == -0.5:
                ra_neg.append(phi * 180/np.pi)
                dec_neg.append((np.pi/2 - theta) * 180/np.pi)  # convert to declination
        ra_pos = np.array(ra_pos)
        dec_pos = np.array(dec_pos)
        ra_neg = np.array(ra_neg)
        dec_neg = np.array(dec_neg)
        
        from sklearn.neighbors import NearestNeighbors
        coords_pos = np.vstack((ra_pos, dec_pos)).T
        coords_neg = np.vstack((ra_neg, dec_neg)).T
        if charge == 0.5:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords_neg)
            distances, indices = nbrs.kneighbors(coords_pos)
        else:
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords_pos)
            distances, indices = nbrs.kneighbors(coords_neg)
        distances = distances * 60  # convert to arcmin
        nearest_distances = distances[:, k-1]  # only one neighbor
    
        
        return nearest_distances

    def convergence_dist(self, charge):
        """Compute the convergence value distribution at defect locations with given charge."""
        kappa_values = []
        for pix in self.defects.keys():
            wnum = self.defects[pix]        
            if wnum == charge:
                kappa = self.kappa_bar[pix]
                kappa_values.append(kappa)
        kappa_values = np.array(kappa_values)
        return kappa_values

    def defect_convergence_correlation(self, charge, min_sep=1., max_sep=100., n_bins=10):
        """Compute the correlation between defect locations and convergence field."""
        ra = []
        dec = []
        for pix in self.defects.keys():
            wnum = self.defects[pix]
            if wnum == charge:
                theta, phi = hp.pix2ang(self.nside, pix)
                ra.append(phi * 180/np.pi)
                dec.append((np.pi/2 - theta) * 180/np.pi)  # convert to declination
        ra = np.array(ra)
        dec = np.array(dec)
        cat_defects = treecorr.Catalog(ra=ra, dec=dec, ra_units='degrees', dec_units='degrees')
        # create convergence catalog
        npix = len(self.kappa_bar)
        theta_all, phi_all = hp.pix2ang(self.nside, np.arange(npix))
        ra_all = phi_all * 180/np.pi
        dec_all = (np.pi/2 - theta_all) * 180/np.pi
        kappa_all = self.kappa_bar
        cat_kappa = treecorr.Catalog(ra=ra_all, dec=dec_all, k=kappa_all, ra_units='degrees', dec_units='degrees')
        dk = treecorr.NKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
        dk.process(cat_defects, cat_kappa)
        thetas = dk.meanr
        xi = dk.xi
        varxi = dk.varxi
        covmat = dk.cov
        return thetas, xi, varxi, covmat
    
    def convergence_correlation(self, min_sep=1., max_sep=100., n_bins=10, smoothed=True):
        """Compute the convergence auto-correlation function."""
        if smoothed:
            npix = len(self.kappa_bar)
            kappa_all = self.kappa_bar
        else:
            npix = len(self.kappa_bar_unsmoothed)
            kappa_all = self.kappa_bar_unsmoothed
        theta_all, phi_all = hp.pix2ang(self.nside, np.arange(npix))
        ra_all = phi_all * 180/np.pi
        dec_all = (np.pi/2 - theta_all) * 180/np.pi
        cat_kappa = treecorr.Catalog(ra=ra_all, dec=dec_all, k=kappa_all, ra_units='degrees', dec_units='degrees')
        kk = treecorr.KKCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=n_bins, sep_units='arcmin', bin_type='Log')
        kk.process(cat_kappa)
        thetas = kk.meanr
        xi = kk.xi
        varxi = kk.varxi
        covmat = kk.cov
        return thetas, xi, varxi, covmat
        
    def get_lens_potential(self):
        """Return the lensing potential field."""
        self.psi = glass.from_convergence(self.kappa_bar, potential=True, lmax=int(self.nside/4))
        return self.psi

    
    def convergence_power_spectrum(self):
        """Compute the convergence power spectrum Cl."""
        self.cl_kappa = hp.anafast(self.kappa_bar)
        self.ell = np.arange(len(self.cl_kappa))
        return self.ell, self.cl_kappa

    
    def lens_potential_power_spectrum(self):
        """Compute the lensing potential power spectrum Cl."""
        if not hasattr(self, 'psi'):
            self.get_lens_potential()
        self.cl_psi = hp.anafast(self.psi, use_pixel_weights=True, datapath='/home/jbai/.astropy/healpy-data', lmax=int(self.nside/4))#* hp.sphtfunc.gauss_beam(np.radians(10.0), lmax=3*self.nside-1)**2  # 10 arcmin beam smoothing
        self.ell = np.arange(len(self.cl_psi))
        return self.ell, self.cl_psi

    
    def potential_power_moments(self, n):
        """Compute the moments of the lensing potential power spectrum."""
        if not hasattr(self, 'cl_psi'):
            self.lens_potential_power_spectrum()
        def legendre_derivative(ell, n):
            import math
            m = int(n/2)
            if ell <= m:
                return 0
            else:
                # math.factorial(ell+m)/math.factorial(ell-m)
                if m == 2:
                    return (-1)**m * math.factorial(2*m)/math.factorial(m)**2 * (ell-1)*ell*(ell+1)*(ell+2)/4**m
                elif m == 3:
                    return (-1)**m * math.factorial(2*m)/math.factorial(m)**2 * (ell-2)*(ell-1)*ell*(ell+1)*(ell+2)*(ell+3)/4**m
        lgd_factorials = np.array([legendre_derivative(l, n) for l in self.ell])
        moment = np.sum((2*self.ell+1)/(4*np.pi) * self.cl_psi * lgd_factorials)
        return moment
    
    def number_estimation(self):
        m6 = self.potential_power_moments(6)
        m4 = self.potential_power_moments(4)
        self.d = np.abs(m6 / m4) * 3 / (10 * np.pi)
        return self.d
    
    def number(self):
        # for integer charge, count as 2 defects
        total_defects = 0
        n_pos = 0
        n_neg = 0
        for wnum in self.defects.values():
            if wnum == 0.5:
                total_defects += 1
                n_pos += 1
            elif wnum == -0.5:
                total_defects += 1
                n_neg += 1
            elif wnum == 1:
                total_defects += 2
                n_pos += 2
            elif wnum == -1:
                total_defects += 2
                n_neg += 2
        self.defect_count = total_defects  
        n = total_defects / (4 * np.pi)  # number per steradian
        return n, n_pos, n_neg




if __name__ == "__main__":
    df = DefectFinder("outfiles/ShearConvergenceMapData.npz")
    df.find_defects(savepath="outfiles/defects_data.npz", cluster=True)
    df.save_defects(savepath="outfiles/defects")
    df.plot_defects(savepath="outfiles/defects_on_convergence_map.png", defect_datapath="outfiles/defects_data.npz")
    df.plot_patch(loc=(np.pi/4, 0), size_arcmin=1000, savepath="outfiles/defects_field_patch_hires.png", resolution=256)

    print(df.number(),df.number_estimation())