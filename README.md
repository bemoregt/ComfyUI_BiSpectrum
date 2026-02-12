# ComfyUI_BiSpectrum

ComfyUI custom node that computes the **Bispectrum** (triple correlation spectrum) of an audio signal and outputs it as an image.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_BiSpectrum/blob/main/ScrShot%2011.png)

---

## What is the Bispectrum?

The bispectrum is a higher-order spectral analysis tool defined as:

$$B(f_1, f_2) = \mathbb{E}\bigl[X(f_1)\cdot X(f_2)\cdot X^*(f_1+f_2)\bigr]$$

where $X(f)$ is the discrete Fourier transform of the signal.
It reveals **nonlinear frequency interactions (phase coupling)** between pairs of frequency components — information that is invisible in the ordinary power spectrum.

---

## Node

### Audio to Bispectrum

| | |
|---|---|
| **Input** | `AUDIO` (ComfyUI native audio type) |
| **Output** | `IMAGE` — 2-D bispectrum plot `[1, H, W, 3]` |
| **Category** | `audio` |

#### Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `nfft` | 512 | 64 – 1024 | FFT size. Larger → finer frequency resolution, slower |
| `hop_length` | 256 | 32 – 1024 | Frame shift in samples |
| `win_length` | 512 | 64 – 1024 | Analysis window length in samples |
| `normalize` | OFF | ON / OFF | **ON**: bicoherence² ∈ [0, 1] / **OFF**: raw magnitude |
| `log_scale` | ON | ON / OFF | Apply log(1 + \|B\|) — effective in Magnitude mode only |
| `colormap` | inferno | inferno / magma / viridis / plasma / hot / cool / gray / jet | Matplotlib colormap |
| `width` | 768 | 256 – 4096 | Output image width (px) |
| `height` | 768 | 256 – 4096 | Output image height (px) |
| `show_axes` | ON | ON / OFF | Show frequency axes, colorbar, and title |

---

## Output Modes

### Magnitude mode (`normalize = OFF`)
Displays `|B(f₁, f₂)|`, optionally on a log scale.
Bright regions indicate strong nonlinear coupling between frequencies $f_1$, $f_2$, and $f_1+f_2$.

### Bicoherence mode (`normalize = ON`)
Displays the **squared bicoherence**:

$$b^2(f_1, f_2) = \frac{|B(f_1,f_2)|^2}{\mathbb{E}[|X(f_1)X(f_2)|^2]\cdot\mathbb{E}[|X(f_1+f_2)|^2]}$$

Values are normalized to **[0, 1]**, making results comparable across signals of different amplitudes.

---

## Computed Region

Only the **triangular inner region** is computed:

```
0 ≤ f₂ ≤ f₁,   f₁ + f₂ < Nyquist
```

This exploits the symmetry of the bispectrum and avoids redundant computation.

---

## Installation

### Option A — Clone into custom_nodes directly
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/bemoregt/ComfyUI_BiSpectrum.git
```

### Option B — Symlink from a separate directory
```bash
git clone https://github.com/bemoregt/ComfyUI_BiSpectrum.git
ln -s /path/to/ComfyUI_BiSpectrum /path/to/ComfyUI/custom_nodes/ComfyUI_BiSpectrum
```

Restart ComfyUI after installation.
No additional packages are required beyond those already used by ComfyUI (`numpy`, `matplotlib`, `Pillow`, `torch`).

---

## Example Workflow

```
Load Audio ──► Audio to Bispectrum ──► Preview Image
```

Typical settings for music analysis:
- `nfft = 512`, `hop_length = 256`, `win_length = 512`
- `normalize = ON` (bicoherence), `colormap = inferno`

---

## License

MIT
