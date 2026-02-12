import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


COLORMAPS = ["inferno", "magma", "viridis", "plasma", "hot", "cool", "gray", "jet"]

# 청크당 최대 메모리 목표: ~200 MB (complex128 = 16 bytes)
_TARGET_CHUNK_BYTES = 200 * 1024 * 1024


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """waveform: [1, C, T] → [T] mono float32"""
    audio = waveform.squeeze(0)  # [C, T]
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    return audio.float()


def _make_frames(audio_np: np.ndarray, win_length: int, hop_length: int, nfft: int) -> np.ndarray:
    """오디오를 윈도우 프레임으로 나눠 rfft 적용 → [n_frames, nfft//2+1] complex128"""
    window = np.hanning(win_length)
    N = len(audio_np)
    frames = []
    for start in range(0, max(N - win_length + 1, 1), hop_length):
        frame = audio_np[start: start + win_length]
        if len(frame) < win_length:
            frame = np.pad(frame, (0, win_length - len(frame)))
        frames.append(np.fft.rfft(frame * window, n=nfft))
    if not frames:
        frames.append(np.zeros(nfft // 2 + 1, dtype=complex))
    return np.array(frames, dtype=np.complex128)  # [n_frames, half]


def _compute_bispectrum(
    audio_np: np.ndarray,
    nfft: int,
    hop_length: int,
    win_length: int,
    normalize: bool,
    max_frames: int,
) -> np.ndarray:
    """
    Bispectrum 계산
      B(f1, f2) = E[ X(f1) · X(f2) · X*(f1+f2) ]

    삼각 영역(0 ≤ f2 ≤ f1, f1+f2 < half)만 계산.
    normalize=True 이면 이중일관성(bicoherence²) 반환 [0, 1].
    프레임 수가 max_frames를 초과하면 균등 서브샘플링으로 제한.
    """
    half = nfft // 2 + 1
    X = _make_frames(audio_np, win_length, hop_length, nfft)  # [n_frames, half]

    # 프레임 수 제한: 균등 간격 서브샘플링
    n_frames = len(X)
    if n_frames > max_frames:
        idx = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        X = X[idx]

    n_frames = len(X)

    # 메모리에 맞는 청크 크기 자동 계산 (복소128 = 16 bytes, 3개 배열)
    chunk_size = max(256, _TARGET_CHUNK_BYTES // (n_frames * 16 * 3))

    # 유효한 (f1, f2) 쌍 인덱스 생성
    f1_g, f2_g = np.mgrid[0:half, 0:half]
    f12_g = f1_g + f2_g
    valid = (f12_g < half) & (f2_g <= f1_g)

    f1_v = f1_g[valid]   # [n_valid]
    f2_v = f2_g[valid]
    f12_v = f12_g[valid]

    # 청크 단위 bispectrum 누적
    bispec_v = np.zeros(len(f1_v), dtype=np.complex128)
    denom_v = np.zeros(len(f1_v), dtype=np.float64) if normalize else None

    for i in range(0, len(f1_v), chunk_size):
        sl = slice(i, i + chunk_size)
        Xf1 = X[:, f1_v[sl]]    # [n_frames, chunk]
        Xf2 = X[:, f2_v[sl]]
        Xf12 = X[:, f12_v[sl]]

        triple = Xf1 * Xf2 * np.conj(Xf12)
        bispec_v[sl] = triple.mean(axis=0)

        if normalize:
            denom_v[sl] = (
                np.mean(np.abs(Xf1 * Xf2) ** 2, axis=0) *
                np.mean(np.abs(Xf12) ** 2, axis=0)
            )

    # 2D 배열로 재구성 — 무효 영역은 NaN (컬러맵 범위 계산에서 제외)
    result = np.full((half, half), np.nan, dtype=np.float64)
    if normalize:
        bic2 = np.abs(bispec_v) ** 2 / np.maximum(denom_v, 1e-30)
        result[f1_v, f2_v] = np.clip(bic2, 0.0, 1.0)
    else:
        result[f1_v, f2_v] = np.abs(bispec_v)

    return result


def _render_bispectrum(
    bispec: np.ndarray,
    sample_rate: int,
    normalize: bool,
    log_scale: bool,
    colormap: str,
    fmax_hz: float,
    width: int,
    height: int,
    show_axes: bool,
) -> torch.Tensor:
    """Bispectrum 2D 배열 → matplotlib 렌더링 → [1, H, W, 3] float32 tensor"""
    half = bispec.shape[0]
    nyquist = sample_rate / 2.0

    # fmax_hz=0 이면 Nyquist 전체 표시
    display_fmax = fmax_hz if fmax_hz > 0 else nyquist
    display_fmax = min(display_fmax, nyquist)

    # 표시 영역에 해당하는 bin 인덱스 범위
    bin_max = int(round(display_fmax / nyquist * (half - 1))) + 1
    bin_max = min(bin_max, half)

    data = bispec[:bin_max, :bin_max].copy()

    if log_scale and not normalize:
        # NaN은 그대로 두고 유효값만 변환
        valid_mask = ~np.isnan(data)
        data[valid_mask] = np.log1p(data[valid_mask])

    # 퍼센타일 기반 컬러맵 범위 (무효 NaN 제외)
    valid_vals = data[~np.isnan(data)]
    if len(valid_vals) > 0 and valid_vals.max() > 0:
        vmin = 0.0
        vmax = float(np.percentile(valid_vals, 99.5))
        if vmax == 0:
            vmax = 1.0
    else:
        vmin, vmax = 0.0, 1.0

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # NaN을 배경색(검정)으로 렌더링
    cmap_obj = plt.get_cmap(colormap).copy()
    cmap_obj.set_bad(color="black")

    freq_tick = display_fmax
    extent = [0, freq_tick, 0, freq_tick]

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_obj,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    if show_axes:
        ax.set_xlabel("f₂ (Hz)")
        ax.set_ylabel("f₁ (Hz)")
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

    Bispectrum B(f1, f2) = E[ X(f1) · X(f2) · X*(f1+f2) ]
    는 신호의 2차 비선형 주파수 상관(위상 결합, phase coupling)을 시각화합니다.
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
                        "tooltip": "FFT 크기. 클수록 주파수 해상도↑, 속도↓, 메모리↑",
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
                        "tooltip": "Magnitude 모드에서 log(1+|B|) 적용 (normalize=OFF일 때만 유효)",
                    },
                ),
                "fmax": (
                    "FLOAT",
                    {
                        "default": 8000.0,
                        "min": 0.0,
                        "max": 24000.0,
                        "step": 500.0,
                        "tooltip": "표시할 최대 주파수 (Hz). 0 = Nyquist 전체",
                    },
                ),
                "colormap": (COLORMAPS, {"default": "inferno"}),
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
                        "tooltip": "사용할 최대 프레임 수. 오디오가 길면 균등 서브샘플링. 클수록 정확하지만 느림",
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
        normalize,
        log_scale,
        fmax,
        colormap,
        width,
        height,
        max_frames,
        show_axes,
    ):
        waveform = audio["waveform"]      # [1, C, T]
        sample_rate = audio["sample_rate"]

        mono = _to_mono(waveform)
        audio_np = mono.numpy()

        bispec = _compute_bispectrum(
            audio_np, nfft, hop_length, win_length, normalize, max_frames
        )

        image = _render_bispectrum(
            bispec, sample_rate, normalize, log_scale,
            colormap, fmax, width, height, show_axes,
        )

        return (image,)


NODE_CLASS_MAPPINGS = {
    "AudioToBispectrum": AudioToBispectrum,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToBispectrum": "Audio to Bispectrum",
}
