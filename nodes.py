import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


COLORMAPS = ["inferno", "magma", "viridis", "plasma", "hot", "cool", "gray", "jet"]

# 청크당 메모리 목표: ~200 MB (complex128 = 16 bytes)
_TARGET_CHUNK_BYTES = 200 * 1024 * 1024


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """waveform: [1, C, T] → [T] mono float32"""
    audio = waveform.squeeze(0)
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    return audio.float()


def _make_frames(audio_np: np.ndarray, win_length: int, hop_length: int, nfft: int) -> np.ndarray:
    """
    오디오 → 윈도우 프레임 → full FFT
    반환: [n_frames, nfft] complex128
    """
    window = np.hanning(win_length)
    N = len(audio_np)
    frames = []
    for start in range(0, max(N - win_length + 1, 1), hop_length):
        frame = audio_np[start: start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)))
        frames.append(np.fft.fft(frame * window, n=nfft))
    if not frames:
        frames.append(np.zeros(nfft, dtype=complex))
    return np.array(frames, dtype=np.complex128)  # [n_frames, nfft]


def _compute_bispectrum(
    audio_np: np.ndarray,
    nfft: int,
    hop_length: int,
    win_length: int,
    normalize: bool,
    max_frames: int,
    fmax_hz: float,
    sample_rate: int,
) -> np.ndarray:
    """
    Full 2D Bispectrum 계산
      B(f1, f2) = E[ X(f1) · X(f2) · X*((f1+f2) mod N) ]

    양/음 주파수 모두 포함한 전체 2D 평면을 계산.
    반환: [2*bin_max+1, 2*bin_max+1] float64 (magnitude 또는 bicoherence²)
    축 순서: [f1_index, f2_index], f1/f2 ∈ [-fmax, +fmax]
    """
    X = _make_frames(audio_np, win_length, hop_length, nfft)  # [n_frames, nfft]

    # 프레임 수 제한 (균등 서브샘플링)
    n_frames = len(X)
    if n_frames > max_frames:
        idx = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        X = X[idx]
    n_frames = len(X)

    # fmax에 해당하는 bin 수
    freq_res = sample_rate / nfft          # Hz / bin
    bin_max = max(1, int(fmax_hz / freq_res))
    bin_max = min(bin_max, nfft // 2)

    # f1, f2 ∈ [-bin_max, ..., bin_max] (정수 bin 인덱스)
    bins = np.arange(-bin_max, bin_max + 1)  # length = 2*bin_max+1
    nb = len(bins)

    # 모든 (f1, f2) 조합 → nfft 기준 양수 인덱스로 변환
    k1_g, k2_g = np.meshgrid(bins, bins, indexing="ij")   # [nb, nb]
    k12_g = (k1_g + k2_g) % nfft
    k1_mod = k1_g % nfft
    k2_mod = k2_g % nfft

    k1_v = k1_mod.ravel()    # [nb²]
    k2_v = k2_mod.ravel()
    k12_v = k12_g.ravel()
    n_pairs = len(k1_v)

    # 청크 크기 자동 결정
    chunk_size = max(256, _TARGET_CHUNK_BYTES // (n_frames * 16 * 3))

    bispec_v = np.zeros(n_pairs, dtype=np.complex128)
    denom_v = np.zeros(n_pairs, dtype=np.float64) if normalize else None

    for i in range(0, n_pairs, chunk_size):
        sl = slice(i, i + chunk_size)
        Xf1 = X[:, k1_v[sl]]      # [n_frames, chunk]
        Xf2 = X[:, k2_v[sl]]
        Xf12 = X[:, k12_v[sl]]

        triple = Xf1 * Xf2 * np.conj(Xf12)
        bispec_v[sl] = triple.mean(axis=0)

        if normalize:
            denom_v[sl] = (
                np.mean(np.abs(Xf1 * Xf2) ** 2, axis=0) *
                np.mean(np.abs(Xf12) ** 2, axis=0)
            )

    if normalize:
        bic2 = np.abs(bispec_v) ** 2 / np.maximum(denom_v, 1e-30)
        return np.clip(bic2, 0.0, 1.0).reshape(nb, nb)

    return np.abs(bispec_v).reshape(nb, nb)


def _render_bispectrum(
    bispec: np.ndarray,
    sample_rate: int,
    nfft: int,
    normalize: bool,
    log_scale: bool,
    colormap: str,
    fmax_hz: float,
    width: int,
    height: int,
    show_axes: bool,
) -> torch.Tensor:
    """Bispectrum 2D 배열 → matplotlib 렌더링 → [1, H, W, 3] float32 tensor"""
    freq_res = sample_rate / nfft
    bin_max = (bispec.shape[0] - 1) // 2
    actual_fmax = bin_max * freq_res      # 실제 표시 최대 Hz

    data = bispec.copy()

    if log_scale and not normalize:
        data = np.log1p(data)

    # 퍼센타일 기반 컬러맵 범위
    flat = data.ravel()
    pos = flat[flat > 0]
    vmin = 0.0
    vmax = float(np.percentile(pos, 99.5)) if len(pos) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    extent = [-actual_fmax, actual_fmax, -actual_fmax, actual_fmax]

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=colormap,
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )

    if show_axes:
        ax.set_xlabel("f₁ (Hz)")
        ax.set_ylabel("f₂ (Hz)")
        ax.axhline(0, color="white", linewidth=0.4, linestyle="--", alpha=0.4)
        ax.axvline(0, color="white", linewidth=0.4, linestyle="--", alpha=0.4)
        if normalize:
            cb_label = "Bicoherence²"
            title = "Bispectrum — Bicoherence²"
        elif log_scale:
            cb_label = "log(1 + |B(f₁,f₂)|)"
            title = "Bispectrum Magnitude (log scale)"
        else:
            cb_label = "|B(f₁,f₂)|"
            title = "Bispectrum Magnitude"
        plt.colorbar(im, ax=ax, label=cb_label, pad=0.02)
        ax.set_title(title)
        plt.tight_layout()
    else:
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize((width, height), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, 3]


class AudioToBispectrum:
    """
    ComfyUI 커스텀 노드: 오디오 → Bispectrum(삼중상관 스펙트럼) 이미지

    B(f1, f2) = E[ X(f1) · X(f2) · X*((f1+f2) mod N) ]

    양/음 주파수를 모두 포함한 전체 2D 평면을 계산하여
    대칭 패턴과 위상 결합(phase coupling)을 시각화합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "nfft": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 1024,
                        "step": 64,
                        "tooltip": "FFT 크기. 클수록 주파수 해상도↑, 속도↓",
                    },
                ),
                "hop_length": (
                    "INT",
                    {"default": 256, "min": 32, "max": 1024, "step": 32},
                ),
                "win_length": (
                    "INT",
                    {"default": 512, "min": 64, "max": 1024, "step": 64},
                ),
                "fmax": (
                    "FLOAT",
                    {
                        "default": 4000.0,
                        "min": 100.0,
                        "max": 22050.0,
                        "step": 500.0,
                        "tooltip": "표시/계산할 최대 주파수 (Hz). 작을수록 해상도 높고 빠름",
                    },
                ),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Bicoherence²",
                        "label_off": "Magnitude",
                        "tooltip": "ON: 이중일관성(bicoherence²) [0,1] / OFF: 절댓값",
                    },
                ),
                "log_scale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "log(1+|B|) 적용 — Magnitude 모드에서만 유효",
                    },
                ),
                "colormap": (COLORMAPS, {"default": "jet"}),
                "width": (
                    "INT",
                    {"default": 768, "min": 256, "max": 4096, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 256, "max": 4096, "step": 64},
                ),
                "max_frames": (
                    "INT",
                    {
                        "default": 500,
                        "min": 50,
                        "max": 5000,
                        "step": 50,
                        "tooltip": "최대 프레임 수. 오디오가 길면 균등 서브샘플링",
                    },
                ),
                "show_axes": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("bispectrum",)
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(
        self,
        audio,
        nfft,
        hop_length,
        win_length,
        fmax,
        normalize,
        log_scale,
        colormap,
        width,
        height,
        max_frames,
        show_axes,
    ):
        waveform = audio["waveform"]       # [1, C, T]
        sample_rate = audio["sample_rate"]

        mono = _to_mono(waveform)
        audio_np = mono.numpy()

        bispec = _compute_bispectrum(
            audio_np, nfft, hop_length, win_length,
            normalize, max_frames, fmax, sample_rate,
        )

        image = _render_bispectrum(
            bispec, sample_rate, nfft, normalize, log_scale,
            colormap, fmax, width, height, show_axes,
        )

        return (image,)


NODE_CLASS_MAPPINGS = {
    "AudioToBispectrum": AudioToBispectrum,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToBispectrum": "Audio to Bispectrum",
}
