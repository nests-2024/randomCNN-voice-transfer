import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from torch.autograd import Variable

from model import RandomCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

N_FFT = 1024

def wav2spectrum(filename):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, n_fft=N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, sr


def wav2spectrum_keep_phase(filename):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, n_fft=N_FFT)
    p = np.angle(S)

    S = np.log1p(np.abs(S))
    return S, p, sr


def plot_spectrum(spectrum):
    spec_db = librosa.amplitude_to_db(spectrum, ref=np.max)
    librosa.display.specshow(spec_db)
    plt.show()



def spectrum2wav(spectrum, sr=None, outfile=None):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft=N_FFT))

    if sr is not None and outfile is not None:
        sf.write(outfile, x, sr)
    return x


def spectrum2wav_keep_phase(spectrum, p, sr, outfile):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft=N_FFT))
    sf.write(outfile, x, sr)


def compute_content_loss(a_C, a_G):
    """
    Compute the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)

    Returns:
    J_content -- scalar that you compute using equation 1 above
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    a_C_unrolled = a_C.view(m * n_C, n_H * n_W)
    a_G_unrolled = a_G.view(m * n_C, n_H * n_W)

    # Compute the cost
    J_content = 1.0 / (4 * m * n_C * n_H * n_W) * torch.sum((a_C_unrolled - a_G_unrolled) ** 2)

    return J_content


def gram(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_L)

    Returns:
    GA -- Gram matrix of shape (n_C, n_C)
    """
    GA = torch.matmul(A, A.t())

    return GA


def gram_over_time_axis(A):
    """
    Argument:
    A -- matrix of shape (1, n_C, n_H, n_W)

    Returns:
    GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
    """
    m, n_C, n_H, n_W = A.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    A_unrolled = A.view(m * n_C * n_H, n_W)
    GA = torch.matmul(A_unrolled, A_unrolled.t())

    return GA


def compute_layer_style_loss(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)

    Returns:
    J_style_layer -- tensor representing a scalar style cost.
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)

    # Calculate the gram
    # !!!!!! IMPORTANT !!!!! Here we compute the Gram along n_C,
    # not along n_H * n_W. But is the result the same? No.
    GS = gram_over_time_axis(a_S)
    GG = gram_over_time_axis(a_G)

    # Computing the loss
    J_style_layer = 1.0 / (4 * (n_C ** 2) * (n_H * n_W)) * torch.sum((GS - GG) ** 2)

    return J_style_layer


def run_transfer(content_s, style_s,
                 content_weight=1e2, style_weight=1,
                 cnn_channels=32, cnn_kernel=(3,3),
                 learning_rate=2e-3, num_steps=20000):

    content_torch = torch.from_numpy(content_s)[None, None, :, :].to(device)
    style_torch = torch.from_numpy(style_s)[None, None, :, :].to(device)

    model = RandomCNN(cnn_channels=cnn_channels, cnn_kernel=cnn_kernel)
    model.eval()

    a_C_var = Variable(content_torch, requires_grad=False).float()
    a_S_var = Variable(style_torch, requires_grad=False).float()

    a_C = model(a_C_var)
    a_S = model(a_S_var)

    # Optimizer
    a_G_var = Variable(torch.randn(content_torch.shape) * 1e-3)
    a_G_var.requires_grad = True
    optimizer = torch.optim.Adam([a_G_var], lr=learning_rate)

    print_every = int(num_steps / 20)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        a_G = model(a_G_var)
    
        content_loss = content_weight * compute_content_loss(a_C, a_G)
        style_loss = style_weight * compute_layer_style_loss(a_S, a_G)
        loss = content_loss + style_loss
        loss.backward()
        optimizer.step()

        if step % print_every == 0:
            print("{} {:.4f}% content_loss:{:.4f} style_loss:{:.4f}".format(step, step / num_steps * 100,
                                                                            content_loss.item(),
                                                                            style_loss.item()))

    return a_G_var

"""
# Test
test_S = torch.randn(1, 6, 2, 2)
test_G = torch.randn(1, 6, 2, 2)
print(test_S)
print(test_G)
print(compute_layer_style_loss(test_S, test_G))


# Test
test_C = torch.randn(1, 6, 2, 2)
test_G = torch.randn(1, 6, 2, 2)
print(test_C)
print(test_G)
print(compute_content_loss(test_C, test_G))
"""
