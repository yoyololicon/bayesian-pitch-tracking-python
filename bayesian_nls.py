import numpy as np
from audiolazy.lazy_lpc import levinson_durbin
#from statsmodels.tsa.stattools import levinson_durbin
from spectrum import LEVINSON
from scipy.signal import lfilter, get_window
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
from tqdm import tqdm


class Noise_Estimator(object):
    def __init__(self,
                 prior_ratio=1,
                 static_snr=15,
                 p_threshold=0.99,
                 p_alpha=0.9,
                 noise_alpha=0.8):
        self.noise_var = None
        self.smoothed_p = None
        self.prior_snr = static_snr
        self.prior_ratio = prior_ratio
        self.p_threh = p_threshold
        self.a1 = p_alpha
        self.a2 = noise_alpha

        xih1 = 10 ** (static_snr / 10)
        self.xih1r = 1 / (1 + xih1) - 1
        self.pfac = prior_ratio * (1 + xih1)

    def __call__(self, spec: np.array):
        if self.noise_var is None:
            self.noise_var = np.copy(spec)
            self.smoothed_p = np.zeros_like(spec)
        else:
            r = spec / self.noise_var
            ph1y = 1 / (1 + self.pfac * np.exp(self.xih1r * r))

            self.smoothed_p *= self.a1
            self.smoothed_p += (1 - self.a1) * ph1y
            ph1y = np.where(self.smoothed_p < self.p_threh, ph1y, np.minimum(self.p_threh, ph1y))
            noise_update_spec = spec * (1 - ph1y) + self.noise_var * ph1y
            self.noise_var *= self.a2
            self.noise_var += (1 - self.a2) * noise_update_spec

        return np.copy(self.noise_var)


class Bayesian_Fast_F0(object):
    def __init__(self,
                 sr,
                 window_size,
                 hop_size,
                 max_order=19,
                 nfft=2 ** 14,
                 min_freq=70,
                 max_freq=400,
                 voicing_p=0.7,
                 speech_var=5e-6,
                 cost_function='nls',
                 prew=False,
                 **kwargs):
        self.sr = sr
        self.hop_size = hop_size
        self.N = window_size
        self.L = max_order
        self.F = nfft
        self.small_nfft = 1
        while self.small_nfft < window_size:
            self.small_nfft <<= 1
        self.speech_var = speech_var

        if prew:
            self.denoiser = Noise_Estimator(**kwargs)
        self.voicing_prior = voicing_p

        self.min_idx = int(nfft * min_freq / sr)
        self.max_idx = math.ceil(nfft * max_freq / sr)

        valid_idxs = np.arange(self.min_idx, self.max_idx + 1)
        self.pitch_grid = valid_idxs / nfft
        self.pitch_num = len(valid_idxs)
        self.harm_idxs = valid_idxs[:, None] * np.arange(1, self.L + 1)
        np.clip(self.harm_idxs, 0, self.F // 2 + 1, out=self.harm_idxs)
        #self.harm_idxs[np.where(self.harm_idxs > self.F // 2)] = self.F // 2 + 1
        self.cost_matrix = np.zeros((self.pitch_num, self.L))
        self.cost_mask = self.harm_idxs < self.F / 2
        self.pitch_ll = np.zeros_like(self.cost_matrix)

        self.scaled_alpha_buffer = None
        self.scaled_alpha_buffer2 = np.ones((self.pitch_num, self.L))
        self.scaled_alpha_buffer2 /= self.scaled_alpha_buffer2.sum()

        self.logpi = np.log(voicing_p / self.pitch_num / max_order)
        self.p02p1 = 0.4
        self.p12p0 = 0.3

        std_pitch = max(0.5, 0.2 * hop_size / sr * 1000)
        self.A = np.eye(self.pitch_num)
        for i in range(self.pitch_num):
            self.A[i] = norm.pdf(self.pitch_grid, self.pitch_grid[i], std_pitch / self.sr)
        self.A /= self.A.sum(1, keepdims=True)

        self.B = np.eye(self.L)
        for i in range(self.L):
            self.B[i] = norm.pdf(i, np.arange(self.L), 1)
        self.B /= self.B.sum(0)

        self.cross_corr_vector = np.row_stack(
            (np.full(self.pitch_num, self.N / 2),
             0.5 * np.sin(math.pi * np.arange(1, 2 * self.L + 1)[:, None] * self.pitch_grid * self.N)
             / np.sin(math.pi * np.arange(1, 2 * self.L + 1)[:, None] * self.pitch_grid)))
        self.fft_shifter = np.exp(2j * math.pi * np.arange(self.F // 2 + 1) / self.F * 0.5 * (self.N - 1))

        self.gamma1, self.gamma2 = self._compute_gamma()

        self.cost_func = self._hs_cost_function if cost_function is 'hs' else self._nls_cost_function

    def __call__(self, x: np.array):
        if hasattr(self, 'denoiser'):
            x_norm = np.linalg.norm(x)
            pseudo_spec = np.abs(np.fft.rfft(x, self.small_nfft)) ** 2
            est_noise = self.denoiser(pseudo_spec)
            noise_corr = np.fft.irfft(est_noise)
            filter_coef, *_ = LEVINSON(noise_corr, 30)
            xp = lfilter([1] + filter_coef.tolist(), 1, x)
            xp *= x_norm / np.linalg.norm(xp)
        else:
            xp = x

        self.cost_func(xp)

        #plt.imshow(self.cost_matrix, aspect='auto')
        #plt.show()

        delta = 3  # g prior
        gHat, tauVar = self._laplace_params(self.cost_matrix, 1,
                                            (self.N - 2 * np.arange(1, self.L + 1) - delta) / 2,
                                            self.N / 2)

        self.pitch_ll[:] = np.log(gHat * (delta - 2) * 0.5) + (
                self.N - 2 * np.arange(1, self.L + 1) - delta) * 0.5 * np.log1p(gHat) - self.N * 0.5 * np.log1p(
            gHat * (1 - self.cost_matrix)) + 0.5 * np.log(2 * math.pi * tauVar)

        null_modellike = 1
        inx = np.isnan(self.pitch_ll)
        self.pitch_ll[inx] = -np.inf
        if self.scaled_alpha_buffer is None:
            bar_alpha = self.logpi + self.pitch_ll
            unvoicing_bar_alpha = np.log((1 - self.voicing_prior) * null_modellike)
        else:
            state_prior = (1 - self.p12p0) * self.A @ self.scaled_alpha_buffer @ self.B
            state_prior += self.scaled_alpha_buffer2 * self.p02p1 * self.unvoicing_scaled_alpha_buffer
            bar_alpha = np.log(state_prior) + self.pitch_ll

            unvoicing_bar_alpha = np.log(self.p12p0 * (1 - self.unvoicing_scaled_alpha_buffer) + (
                    1 - self.p02p1) * self.unvoicing_scaled_alpha_buffer) + np.log(null_modellike)

        log_scale = self._log_sumsum_exp_ls(np.concatenate((bar_alpha.ravel(), [unvoicing_bar_alpha])))
        scaled_alpha = bar_alpha - log_scale
        unvoicing_scaled_alpha = unvoicing_bar_alpha - log_scale

        inx = scaled_alpha.argmax()
        pitch_inx, order_inx = np.unravel_index(inx, shape=scaled_alpha.shape)
        estimatedPitch = self.pitch_grid[pitch_inx] * self.sr

        if np.exp(unvoicing_scaled_alpha) < 0.5:
            np.exp(scaled_alpha - np.log1p(-np.exp(unvoicing_scaled_alpha)), out=self.scaled_alpha_buffer2)

        self.scaled_alpha_buffer = np.exp(scaled_alpha)
        self.unvoicing_scaled_alpha_buffer = np.exp(unvoicing_scaled_alpha)

        return estimatedPitch, 1 - np.exp(unvoicing_scaled_alpha), order_inx + 1

    def _TH_matrix(self, nr, nc, corr, is_added):
        offset = 3
        t = corr[nr - 1::-1]
        h = corr[np.arange(nc) + offset + nr - 2]

        if is_added:
            return t + h
        else:
            return t - h

    def _gamma_single_sin(self, corr, a, is_added):
        R = self._TH_matrix(1, 1, corr, is_added)
        psi = 1 / R[0]
        gamma = psi
        phi = a * gamma
        return psi, phi, gamma

    def _gamma_2sin(self, corr, psi, gamma, is_added):
        R = self._TH_matrix(2, 2, corr, is_added)
        alpha = R[0] * gamma
        gamma = np.row_stack((-R[0] * psi, np.ones(self.pitch_num))) / (R[1] - R[0] ** 2 * psi)
        return R, alpha, gamma

    def _gamma_multi_sin(self, R_old, order, corr, a, phi, psi, gamma_old, gamma_new, alpha_old, is_added):
        R_new = self._TH_matrix(order, order, corr, is_added)
        lmd = a - np.sum(R_old * phi, 0)
        mu = -np.sum(R_old * psi, 0)
        phi = np.row_stack((phi, np.zeros(self.pitch_num))) + lmd * gamma_new
        psi = np.row_stack((psi, np.zeros(self.pitch_num))) + mu * gamma_new
        alpha_new = np.sum(R_new[:-1] * gamma_new, 0)
        b = (alpha_old - alpha_new) * gamma_new + \
            np.row_stack((np.zeros(self.pitch_num), gamma_new[:order - 2])) + \
            np.row_stack((gamma_new[1:] - gamma_old[:order - 2], np.zeros(self.pitch_num))) + \
            psi[-1] * phi - phi[-1] * psi
        gamma_old = gamma_new
        nu = np.sum(R_new[:-1] * b, 0) / gamma_new[-1]
        new_dim = 1 / (nu + R_new[order - 1])
        gamma_new = np.row_stack((new_dim / gamma_old[-1] * b, new_dim))
        return R_new, phi, psi, alpha_new, gamma_old, gamma_new

    def _compute_gamma(self):
        a1 = np.copy(self.cross_corr_vector[1:self.L])
        a1[1:] += self.cross_corr_vector[2:self.L]
        a2 = a1

        gamma1_list, gamma2_list = [], []

        assert self.L > 2

        # l = 1
        psi1, phi1, gamma_old1 = self._gamma_single_sin(self.cross_corr_vector[:3], a1[0], is_added=True)
        gamma1_list.append(gamma_old1)

        psi2, phi2, gamma_old2 = self._gamma_single_sin(self.cross_corr_vector[:3], a2[0], is_added=False)
        gamma2_list.append(gamma_old2)

        gamma_old1 = gamma_old1[None, :]
        gamma_old2 = gamma_old2[None, :]

        # l = 2
        R1, alpha1, gamma_new1 = self._gamma_2sin(self.cross_corr_vector[:5], psi1, gamma_old1, is_added=True)
        gamma1_list.append(gamma_new1)

        R2, alpha2, gamma_new2 = self._gamma_2sin(self.cross_corr_vector[:5], psi2, gamma_old2, is_added=False)
        gamma2_list.append(gamma_new2)

        for l in range(3, self.L + 1):
            R1, phi1, psi1, alpha1, gamma_old1, gamma_new1 = self._gamma_multi_sin(
                R1[:-1], l, self.cross_corr_vector[:2 * l + 1], a1[l - 2], phi1, psi1, gamma_old1, gamma_new1, alpha1,
                is_added=True)
            R2, phi2, psi2, alpha2, gamma_old2, gamma_new2 = self._gamma_multi_sin(
                R2[:-1], l, self.cross_corr_vector[:2 * l + 1], a2[l - 2], phi2, psi2, gamma_old2, gamma_new2, alpha2,
                is_added=False)

            gamma1_list.append(gamma_new1)
            gamma2_list.append(gamma_new2)

        return gamma1_list, gamma2_list

    def _log_sumsum_exp_ls(self, x: np.array):
        max_temp = x.max()
        return np.log(np.exp(x - max_temp).sum()) + max_temp

    def _hs_cost_function(self, x: np.array):
        spec = np.abs(np.fft.rfft(x, self.F)) ** 2
        spec = np.pad(spec, (0, 1), 'constant', constant_values=0)
        np.take(spec, self.harm_idxs, out=self.cost_matrix)
        np.cumsum(self.cost_matrix, 1, out=self.cost_matrix)
        self.cost_matrix *= 2 / self.N / (x @ x + self.speech_var * self.N)
        return

    def _nls_cost_function(self, x: np.array):
        spec = np.fft.rfft(x, self.F) * self.fft_shifter
        spec = np.pad(spec, (0, 1), 'constant', constant_values=0)
        Zc = np.take(spec, self.harm_idxs.T)

        for l, (g1, g2) in enumerate(zip(self.gamma1, self.gamma2)):
            if l == 0:
                ls_sol1 = Zc[0].real * g1[None, :]
                ls_sol2 = Zc[0].imag * g2[None, :]
            else:
                R1 = self._TH_matrix(l + 1, l + 1, self.cross_corr_vector[:2 * self.L + 2], is_added=True)
                lambda1 = Zc[l].real - np.sum(R1[:-1] * ls_sol1, 0)
                ls_sol1 = np.row_stack((ls_sol1, np.zeros(self.pitch_num))) + lambda1 * g1

                R2 = self._TH_matrix(l + 1, l + 1, self.cross_corr_vector[:2 * self.L + 2], is_added=False)
                lambda2 = Zc[l].imag - np.sum(R2[:-1] * ls_sol2, 0)
                ls_sol2 = np.row_stack((ls_sol2, np.zeros(self.pitch_num))) + lambda2 * g2

            self.cost_matrix[:, l] = np.sum(Zc[:l + 1].real * ls_sol1, 0) + np.sum(Zc[:l + 1].imag * ls_sol2, 0)
        self.cost_matrix /= x @ x + self.speech_var * self.N
        self.cost_matrix[~self.cost_mask] = np.nan
        return

    def _laplace_params(self, cod, v, w, u):
        a = (1 - cod) * (v + w - u)
        b = (u - v) * cod + 2 * v + w - u
        gHat = - 0.5 * (b + np.sqrt(b ** 2 - 4 * a * v)) / a
        tauVar = 1 / (gHat * (1 - cod) * u / (1 + gHat * (1 - cod)) ** 2 - gHat * w / (1 + gHat) ** 2)
        return gHat, tauVar


def bayesian_nls(y, sr,
                 window_size,
                 hop_size,
                 verbose=1,
                 **kwargs):
    pitch_tracker = Bayesian_Fast_F0(sr, window_size, hop_size, **kwargs)

    y = np.pad(y - y.mean(), (window_size // 2, 0), 'constant', constant_values=0)
    y *= np.sqrt(3.1623e-5 / np.mean(y ** 2))
    output = []
    for i in tqdm(range(0, y.shape[0] - window_size, hop_size), disable=not verbose):
        p, u, l = pitch_tracker(y[i:i + window_size])
        if u < 0.5:
            p = 0
        output.append([p, u, l])
    return np.array(output)


if __name__ == '__main__':

    from resampy import resample
    from scipy import signal

    parser = argparse.ArgumentParser(description='Bayesian Pitch Tracking Using Harmonic model and Fast NLS')
    parser.add_argument('infile', type=str)
    parser.add_argument('--segment_length', default=0.025, type=float)
    parser.add_argument('--segment_shift', default=0.01, type=float)
    parser.add_argument('-L', '--max_order', default=30, type=int)
    parser.add_argument('--cost_function', default='nls', type=str)

    args = parser.parse_args()
    y, sr = sf.read(args.infile, always_2d=True)
    y = y.sum(1)

    hipass_symmetric_filter = signal.firwin(sr // 5 + 1, 15 / sr * 2, pass_zero=False)
    y = signal.fftconvolve(y, hipass_symmetric_filter, 'same')

    if sr != 16000:
        y = resample(y, sr, 16000)
        sr = 16000
    y += np.random.randn(*y.shape) * 0.1

    plt.plot(y)
    plt.show()

    N = int(sr * args.segment_length)
    hop = int(sr * args.segment_shift)

    import librosa

    S = np.log(np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)))
    p = bayesian_nls(y, sr, N, hop, max_order=10, min_freq=100, max_freq=1000, prew=False)
    ax = plt.subplot(4, 1, 1)
    plt.imshow(S, aspect='auto', origin='lower')
    plt.subplot(4, 1, 2, sharex=ax)
    plt.plot(p[:, 0])
    plt.subplot(4, 1, 3, sharex=ax)
    plt.plot(p[:, 1])
    plt.subplot(4, 1, 4, sharex=ax)
    plt.plot(p[:, 2])
    plt.show()
