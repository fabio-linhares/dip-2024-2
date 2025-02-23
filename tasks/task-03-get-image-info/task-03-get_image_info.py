import numpy as np

def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """
    
    ### START CODE HERE ###

    # Dimensões:
    # Para imagens cinza: altura e largura;
    # Para imagens coloridas: altura, largura e profundidade;

    height, width = image.shape[:2] # [:2] retorna os dois primeiros elementos desta tupla, que sempre correspondem à altura e largura
    dtype = str(image.dtype) # retorna o tipo de dados do array: uint8 ou float32, por exemplo.
    
    # Profundidade: refere-se ao número de canais de cor em uma imagem:
    # Profundidade 1: Imagens em escala de cinza (um único canal).
    # Profundidade 3: Imagens coloridas típicas (RGB - Red, Green, Blue).
    # Profundidade 4: Imagens com canal alfa (RGBA - Red, Green, Blue, Alpha).

    # image.ndim retorna o número de dimensões do array.
    if image.ndim == 2: # é uma imagem em escala de cinza (profundidade 1).
        depth = 1
    elif image.ndim == 3:  # é uma imagem colorida.
        depth = image.shape[2] # dá o número de canais.
    else:
        depth = None
    
    # Estatísticas: fornecem informações sobre a distribuição dos valores dos pixels:
    min_val = np.min(image)
    max_val = np.max(image)
    mean_val = np.mean(image)
    std_val = np.std(image)
    ### END CODE HERE ###

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }

# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")