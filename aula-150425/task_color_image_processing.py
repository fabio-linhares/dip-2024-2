import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""
- **RGB** (Red, Green, Blue): É o formato padrão usado pela maioria das aplicações, 
                                incluindo o Matplotlib, onde as cores são armazenadas 
                                na ordem Vermelho, Verde e Azul.

- **BGR** (Blue, Green, Red): É o formato padrão usado pelo OpenCV, onde as cores 
                                são armazenadas na ordem Azul, Verde e Vermelho.

Se essa conversão não fosse feita, as cores apareceriam invertidas quando exibidas no 
                                Matplotlib: o que deveria ser vermelho apareceria como
                                azul e o que deveria ser azul apareceria como vermelho

Obs.: isso é necessária sempre que lemos uma imagem com OpenCV e pretende exibi-la com
                                Matplotlib ou outras bibliotecas que usam o padrão RGB.
"""

# Diretório comum para todas as imagens
img_dir = '../img'

# -----------------------------------------------------------------------------
# Funcionalidade 1: Geração de histogramas (e1.py)
# -----------------------------------------------------------------------------
def generate_histograms():
    """
    Gera histogramas dos canais RGB para todas as imagens no diretório.
    Origem: e1.py
    """
    print("\nGerando histogramas para todas as imagens...")
    
    # lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # Preparar figura única com todos os histogramas
    num_images = len(img_files)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    fig.suptitle('Histogramas de todas as imagens')

    for i, img_file in enumerate(img_files):
        # pega o path
        img_path = os.path.join(img_dir, img_file)
        
        # Carrega a imagem
        img = cv2.imread(img_path)
        if img is None:
            print(f"Não foi possível carregar a imagem: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR --> RGB

        # separa os canais
        r, g, b = cv2.split(img)
        
        # Ajusta os axes com base no número de imagens
        if num_images > 1:
            ax_row = axes[i]
        else:
            ax_row = axes  # caso só tenha uma imagem
        
        # Histograma do canal R
        ax = ax_row[0] if num_images > 1 else axes[0]
        ax.hist(r.ravel(), 256, [0, 256], color='red', alpha=0.7)
        ax.set_title(f'{img_file} - Canal R')
        ax.set_xlim([0, 256])
        
        # Histograma do canal G
        ax = ax_row[1] if num_images > 1 else axes[1]
        ax.hist(g.ravel(), 256, [0, 256], color='green', alpha=0.7)
        ax.set_title(f'{img_file} - Canal G')
        ax.set_xlim([0, 256])
        
        # Histograma do canal B
        ax = ax_row[2] if num_images > 1 else axes[2]
        ax.hist(b.ravel(), 256, [0, 256], color='blue', alpha=0.7)
        ax.set_title(f'{img_file} - Canal B')
        ax.set_xlim([0, 256])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Ajusta espaço para o título principal

    # Salva a figura antes de mostrar
    plt.savefig('histogramas_todas_imagens.png', dpi=300, bbox_inches='tight')
    print("Histograma salvo como 'histogramas_todas_imagens.png'")
    
    plt.show()

# -----------------------------------------------------------------------------
# Funcionalidade 2: Visualização e análise de canais (e2.py)
# -----------------------------------------------------------------------------
def process_image(img_path):
    """
    Processa uma imagem mostrando seus canais separados e reconstrução.
    Origem: e2.py
    """
    # Carregar a imagem
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Separar os canais
    r, g, b = cv2.split(img)
    
    # Criar figura para visualização
    plt.figure(figsize=(15, 10))
    
    # Exibir imagem original
    plt.subplot(331)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')
    
    # Exibir canais como imagens em escala de cinza
    plt.subplot(332)
    plt.imshow(r, cmap='gray')
    plt.title('Canal R (escala de cinza)')
    plt.axis('off')
    
    plt.subplot(333)
    plt.imshow(g, cmap='gray')
    plt.title('Canal G (escala de cinza)')
    plt.axis('off')
    
    plt.subplot(334)
    plt.imshow(b, cmap='gray')
    plt.title('Canal B (escala de cinza)')
    plt.axis('off')
    
    # Exibir canais como pseudo-cores
    plt.subplot(335)
    plt.imshow(np.zeros_like(img))
    plt.imshow(np.stack((r, np.zeros_like(r), np.zeros_like(r)), axis=2))
    plt.title('Canal R (pseudocor)')
    plt.axis('off')
    
    plt.subplot(336)
    plt.imshow(np.zeros_like(img))
    plt.imshow(np.stack((np.zeros_like(g), g, np.zeros_like(g)), axis=2))
    plt.title('Canal G (pseudocor)')
    plt.axis('off')
    
    plt.subplot(337)
    plt.imshow(np.zeros_like(img))
    plt.imshow(np.stack((np.zeros_like(b), np.zeros_like(b), b), axis=2))
    plt.title('Canal B (pseudocor)')
    plt.axis('off')
    
    # Reconstrução da imagem
    reconstructed = cv2.merge([r, g, b])
    plt.subplot(338)
    plt.imshow(reconstructed)
    plt.title('Imagem Reconstruída')
    plt.axis('off')
    
    # Adicionar o nome do arquivo como suptitle
    plt.suptitle(f'Análise de canais: {os.path.basename(img_path)}')
    plt.tight_layout()
    plt.show()

def analyze_image_channels():
    """
    Analisa os canais de todas as imagens no diretório.
    Origem: e2.py
    """
    print("\nAnalisando canais das imagens...")
    
    # Processar todas as imagens no diretório
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(img_dir, filename)
            print(f"Processando: {filename}")
            process_image(img_path)

# -----------------------------------------------------------------------------
# Funcionalidade 3: Conversão entre espaços de cores (e3.py)
# -----------------------------------------------------------------------------
def convert_color_spaces():
    """
    Converte uma imagem RGB para outros espaços de cores e exibe os resultados.
    """
    print("\nConvertendo entre espaços de cores...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Converter para diferentes espaços de cores
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    # Criar CMYK manualmente (aproximação)
    img_norm = img / 255.0
    k = 1 - np.max(img_norm, axis=2)
    c = (1 - img_norm[:,:,0] - k) / (1 - k + 1e-10)
    m = (1 - img_norm[:,:,1] - k) / (1 - k + 1e-10)
    y = (1 - img_norm[:,:,2] - k) / (1 - k + 1e-10)
    
    # Criar figura para visualização
    plt.figure(figsize=(15, 15))
    
    # Exibir imagem original
    plt.subplot(5, 4, 1)
    plt.imshow(img)
    plt.title('Original (RGB)')
    plt.axis('off')
    
    # Exibir canais RGB
    plt.subplot(5, 4, 2)
    plt.imshow(img[:,:,0], cmap='gray')
    plt.title('Canal R')
    plt.axis('off')
    
    plt.subplot(5, 4, 3)
    plt.imshow(img[:,:,1], cmap='gray')
    plt.title('Canal G')
    plt.axis('off')
    
    plt.subplot(5, 4, 4)
    plt.imshow(img[:,:,2], cmap='gray')
    plt.title('Canal B')
    plt.axis('off')
    
    # Exibir HSV
    plt.subplot(5, 4, 5)
    plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
    plt.title('HSV')
    plt.axis('off')
    
    plt.subplot(5, 4, 6)
    plt.imshow(img_hsv[:,:,0], cmap='hsv')
    plt.title('Canal H')
    plt.axis('off')
    
    plt.subplot(5, 4, 7)
    plt.imshow(img_hsv[:,:,1], cmap='gray')
    plt.title('Canal S')
    plt.axis('off')
    
    plt.subplot(5, 4, 8)
    plt.imshow(img_hsv[:,:,2], cmap='gray')
    plt.title('Canal V')
    plt.axis('off')
    
    # Exibir LAB
    plt.subplot(5, 4, 9)
    plt.imshow(cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB))
    plt.title('LAB')
    plt.axis('off')
    
    plt.subplot(5, 4, 10)
    plt.imshow(img_lab[:,:,0], cmap='gray')
    plt.title('Canal L')
    plt.axis('off')
    
    plt.subplot(5, 4, 11)
    plt.imshow(img_lab[:,:,1], cmap='gray')
    plt.title('Canal A')
    plt.axis('off')
    
    plt.subplot(5, 4, 12)
    plt.imshow(img_lab[:,:,2], cmap='gray')
    plt.title('Canal B (LAB)')
    plt.axis('off')
    
    # Exibir YCrCb
    plt.subplot(5, 4, 13)
    plt.imshow(cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB))
    plt.title('YCrCb')
    plt.axis('off')
    
    plt.subplot(5, 4, 14)
    plt.imshow(img_ycrcb[:,:,0], cmap='gray')
    plt.title('Canal Y')
    plt.axis('off')
    
    plt.subplot(5, 4, 15)
    plt.imshow(img_ycrcb[:,:,1], cmap='gray')
    plt.title('Canal Cr')
    plt.axis('off')
    
    plt.subplot(5, 4, 16)
    plt.imshow(img_ycrcb[:,:,2], cmap='gray')
    plt.title('Canal Cb')
    plt.axis('off')
    
    # Exibir CMYK
    plt.subplot(5, 4, 17)
    cmyk_display = np.stack([c, m, y], axis=2)
    plt.imshow(cmyk_display)
    plt.title('CMYK (visualização)')
    plt.axis('off')
    
    plt.subplot(5, 4, 18)
    plt.imshow(c, cmap='gray')
    plt.title('Canal C')
    plt.axis('off')
    
    plt.subplot(5, 4, 19)
    plt.imshow(m, cmap='gray')
    plt.title('Canal M')
    plt.axis('off')
    
    plt.subplot(5, 4, 20)
    plt.imshow(np.stack([y, y, y], axis=2))
    plt.title('Canal Y (CMYK)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Conversão de Espaços de Cor: {os.path.basename(img_path)}", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

# -----------------------------------------------------------------------------
# Funcionalidade 4: Comparar efeitos de borrão em RGB vs HSV (e4.py)
# -----------------------------------------------------------------------------
def compare_blur_rgb_hsv():
    """
    Compara o efeito do desfoque gaussiano nos espaços de cores RGB e HSV.
    """
    print("\nComparando efeitos de borrão em RGB vs HSV...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Solicita nível de desfoque
    try:
        kernel_size = int(input("Informe o tamanho do kernel para o desfoque (número ímpar, ex: 9): "))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Garante que seja ímpar
    except ValueError:
        print("Entrada inválida. Usando kernel 9x9.")
        kernel_size = 9
    
    # Aplicar desfoque no espaço RGB
    blurred_rgb = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Converter para HSV, aplicar desfoque apenas no canal V, depois converter de volta
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # Desfoque em todos os canais HSV
    blurred_hsv_all = cv2.GaussianBlur(img_hsv, (kernel_size, kernel_size), 0)
    
    # Desfoque apenas no canal V
    v_blurred = cv2.GaussianBlur(v, (kernel_size, kernel_size), 0)
    hsv_v_blurred = cv2.merge([h, s, v_blurred])
    
    # Converter de volta para RGB
    blurred_hsv_all_rgb = cv2.cvtColor(blurred_hsv_all, cv2.COLOR_HSV2RGB)
    blurred_hsv_v_rgb = cv2.cvtColor(hsv_v_blurred, cv2.COLOR_HSV2RGB)
    
    # Exibir resultados
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(blurred_rgb)
    plt.title(f'Desfoque em RGB (kernel {kernel_size}x{kernel_size})')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(blurred_hsv_all_rgb)
    plt.title(f'Desfoque em todos os canais HSV (kernel {kernel_size}x{kernel_size})')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(blurred_hsv_v_rgb)
    plt.title(f'Desfoque apenas no canal V do HSV (kernel {kernel_size}x{kernel_size})')
    plt.axis('off')
    
    plt.suptitle(f"Comparação de Borrão RGB vs HSV: {os.path.basename(img_path)}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
    
    print("\nObservação: O desfoque no espaço HSV com apenas o canal V afetado tende a preservar")
    print("melhor as cores originais, pois apenas o brilho é afetado, mantendo a matiz e saturação.")

# -----------------------------------------------------------------------------
# Funcionalidade 5: Aplicar filtros de detecção de bordas (e5.py)
# -----------------------------------------------------------------------------
def apply_edge_detection():
    """
    Aplica filtros de detecção de bordas (Sobel, Laplaciano) em imagens coloridas.
    """
    print("\nAplicando filtros de detecção de bordas...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Converter para escala de cinza para comparação
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Aplicar Sobel no canal de cinza
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_gray = cv2.magnitude(sobel_x, sobel_y)
    sobel_gray = cv2.normalize(sobel_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Aplicar Laplaciano no canal de cinza
    laplacian_gray = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_gray = cv2.normalize(np.abs(laplacian_gray), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Aplicar Sobel em cada canal RGB
    channels = cv2.split(img)
    sobel_channels = []
    
    for channel in channels:
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobel_x, sobel_y)
        sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobel_channels.append(sobel_norm)
    
    # Aplicar Laplaciano em cada canal RGB
    laplacian_channels = []
    
    for channel in channels:
        laplacian = cv2.Laplacian(channel, cv2.CV_64F)
        laplacian_norm = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        laplacian_channels.append(laplacian_norm)
    
    # Combinar os resultados dos canais
    sobel_combined = cv2.max(cv2.max(sobel_channels[0], sobel_channels[1]), sobel_channels[2])
    laplacian_combined = cv2.max(cv2.max(laplacian_channels[0], laplacian_channels[1]), laplacian_channels[2])
    
    # Criar visualização colorida dos resultados de Sobel
    sobel_color = np.zeros_like(img)
    sobel_color[:,:,0] = sobel_channels[0]  # R
    sobel_color[:,:,1] = sobel_channels[1]  # G
    sobel_color[:,:,2] = sobel_channels[2]  # B
    
    # Criar visualização colorida dos resultados de Laplaciano
    laplacian_color = np.zeros_like(img)
    laplacian_color[:,:,0] = laplacian_channels[0]  # R
    laplacian_color[:,:,1] = laplacian_channels[1]  # G
    laplacian_color[:,:,2] = laplacian_channels[2]  # B
    
    # Exibir resultados
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Imagem em Escala de Cinza')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(sobel_gray, cmap='gray')
    plt.title('Sobel (Escala de Cinza)')
    plt.axis('off')
    
    plt.subplot(3, 3, 4)
    plt.imshow(sobel_channels[0], cmap='gray')
    plt.title('Sobel (Canal R)')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(sobel_channels[1], cmap='gray')
    plt.title('Sobel (Canal G)')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(sobel_channels[2], cmap='gray')
    plt.title('Sobel (Canal B)')
    plt.axis('off')
    
    plt.subplot(3, 3, 7)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel (Canais Combinados)')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(sobel_color)
    plt.title('Sobel (Visualização Colorida)')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(laplacian_gray, cmap='gray')
    plt.title('Laplaciano (Escala de Cinza)')
    plt.axis('off')
    
    # Segunda figura para Laplaciano
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(laplacian_channels[0], cmap='gray')
    plt.title('Laplaciano (Canal R)')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(laplacian_channels[1], cmap='gray')
    plt.title('Laplaciano (Canal G)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(laplacian_channels[2], cmap='gray')
    plt.title('Laplaciano (Canal B)')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(laplacian_combined, cmap='gray')
    plt.title('Laplaciano (Canais Combinados)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(laplacian_color)
    plt.title('Laplaciano (Visualização Colorida)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Detecção de Bordas: {os.path.basename(img_path)}", fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.show()

# -----------------------------------------------------------------------------
# Funcionalidade 6: Filtragem no domínio da frequência (e6.py)
# -----------------------------------------------------------------------------
def frequency_domain_filtering():
    """
    Realiza filtragem passa-alta e passa-baixa no domínio da frequência para cada canal.
    """
    print("\nRealizando filtragem no domínio da frequência...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Função para processar um canal no domínio da frequência
    def process_channel(channel):
        # Expandir para tamanho ótimo para FFT
        rows, cols = channel.shape
        rows_padded = cv2.getOptimalDFTSize(rows)
        cols_padded = cv2.getOptimalDFTSize(cols)
        
        # Criar imagem padded
        channel_padded = np.zeros((rows_padded, cols_padded), dtype=np.float32)
        channel_padded[:rows, :cols] = channel
        
        # DFT
        dft = cv2.dft(channel_padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        
        # Criar máscaras (passa-baixa e passa-alta)
        center_row, center_col = rows_padded // 2, cols_padded // 2
        mask_size = min(center_row, center_col) // 2
        
        # Máscara passa-baixa (círculo branco no centro)
        low_pass = np.zeros((rows_padded, cols_padded, 2), dtype=np.float32)
        cv2.circle(low_pass, (center_col, center_row), mask_size, (1, 1), -1)
        
        # Máscara passa-alta (inverso do passa-baixa)
        high_pass = np.ones((rows_padded, cols_padded, 2), dtype=np.float32)
        cv2.circle(high_pass, (center_col, center_row), mask_size, (0, 0), -1)
        
        # Aplicar máscaras
        low_pass_result = dft_shift * low_pass
        high_pass_result = dft_shift * high_pass
        
        # Voltar para o domínio espacial
        low_pass_ifft_shift = np.fft.ifftshift(low_pass_result)
        low_pass_img = cv2.idft(low_pass_ifft_shift)
        low_pass_img = cv2.magnitude(low_pass_img[:,:,0], low_pass_img[:,:,1])
        
        high_pass_ifft_shift = np.fft.ifftshift(high_pass_result)
        high_pass_img = cv2.idft(high_pass_ifft_shift)
        high_pass_img = cv2.magnitude(high_pass_img[:,:,0], high_pass_img[:,:,1])
        
        # Normalizar os resultados para visualização
        low_pass_img = cv2.normalize(low_pass_img[:rows, :cols], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        high_pass_img = cv2.normalize(high_pass_img[:rows, :cols], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum[:rows, :cols], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return magnitude_spectrum, low_pass_img, high_pass_img
    
    # Processar cada canal
    channels = cv2.split(img)
    results = []
    
    for i, channel in enumerate(channels):
        magnitude, low_pass, high_pass = process_channel(channel)
        results.append((magnitude, low_pass, high_pass))
    
    # Reconstruir imagens coloridas dos resultados
    low_pass_image = cv2.merge([results[0][1], results[1][1], results[2][1]])
    high_pass_image = cv2.merge([results[0][2], results[1][2], results[2][2]])
    
    # Exibir resultados
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')
    
    # Exibir magnitude do espectro para cada canal
    plt.subplot(3, 4, 2)
    plt.imshow(results[0][0], cmap='gray')
    plt.title('Espectro de Magnitude (R)')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(results[1][0], cmap='gray')
    plt.title('Espectro de Magnitude (G)')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(results[2][0], cmap='gray')
    plt.title('Espectro de Magnitude (B)')
    plt.axis('off')
    
    # Exibir resultados de filtro passa-baixa
    plt.subplot(3, 4, 5)
    plt.imshow(low_pass_image)
    plt.title('Filtro Passa-Baixa (Reconstruído)')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(results[0][1], cmap='gray')
    plt.title('Passa-Baixa (R)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(results[1][1], cmap='gray')
    plt.title('Passa-Baixa (G)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(results[2][1], cmap='gray')
    plt.title('Passa-Baixa (B)')
    plt.axis('off')
    
    # Exibir resultados de filtro passa-alta
    plt.subplot(3, 4, 9)
    plt.imshow(high_pass_image)
    plt.title('Filtro Passa-Alta (Reconstruído)')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(results[0][2], cmap='gray')
    plt.title('Passa-Alta (R)')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(results[1][2], cmap='gray')
    plt.title('Passa-Alta (G)')
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.imshow(results[2][2], cmap='gray')
    plt.title('Passa-Alta (B)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Filtragem no Domínio da Frequência: {os.path.basename(img_path)}", fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.show()

# -----------------------------------------------------------------------------
# Funcionalidade 7: Visualizar e manipular planos de bits (e7.py)
# -----------------------------------------------------------------------------
def visualize_bit_planes():
    """
    Extrai e visualiza os planos de bits de cada canal de cor.
    """
    print("\nVisualizando planos de bits...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Separar os canais
    channels = cv2.split(img)
    channel_names = ['R', 'G', 'B']
    
    # Função para extrair e visualizar os planos de bits
    def extract_bit_planes(channel):
        bit_planes = []
        for i in range(8):  # 8 bits por pixel
            bit_plane = ((channel >> i) & 1) * 255
            bit_planes.append(bit_plane.astype(np.uint8))
        return bit_planes
    
    # Extrair planos de bits para cada canal
    all_bit_planes = []
    for channel in channels:
        all_bit_planes.append(extract_bit_planes(channel))
    
    # Reconstruir imagem usando apenas os 4 bits mais significativos
    reconstructed_channels = []
    for channel in channels:
        # Zerar os 4 bits menos significativos
        msb_only = (channel & 0xF0).astype(np.uint8)
        reconstructed_channels.append(msb_only)
    
    reconstructed_img = cv2.merge(reconstructed_channels)
    
    # Exibir os planos de bits - modificando a estrutura para evitar índices > 24
    plt.figure(figsize=(16, 12))
    
    # Primeira figura: imagem original, reconstruída e diferença
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(reconstructed_img)
    plt.title('Reconstruída (4 MSBs)')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    diff = cv2.absdiff(img, reconstructed_img)
    plt.imshow(diff)
    plt.title('Diferença')
    plt.axis('off')
    
    # Segunda figura: MSBs
    plt.figure(figsize=(16, 12))
    plt.suptitle(f"MSBs - Bits Mais Significativos: {os.path.basename(img_path)}", fontsize=16)
    
    # Para cada canal (R, G, B) mostrar os 4 MSBs
    for c in range(3):  # R, G, B
        for b in range(4):  # 4 MSBs (bits 7, 6, 5, 4)
            msb_idx = 7 - b
            plt.subplot(3, 4, 4*c + b + 1)
            plt.imshow(all_bit_planes[c][msb_idx], cmap='gray')
            plt.title(f'{channel_names[c]} Bit {msb_idx} (MSB)')
            plt.axis('off')
    
    # Terceira figura: LSBs
    plt.figure(figsize=(16, 12))
    plt.suptitle(f"LSBs - Bits Menos Significativos: {os.path.basename(img_path)}", fontsize=16)
    
    # Para cada canal (R, G, B) mostrar os 4 LSBs
    for c in range(3):  # R, G, B
        for b in range(4):  # 4 LSBs (bits 3, 2, 1, 0)
            lsb_idx = 3 - b
            plt.subplot(3, 4, 4*c + b + 1)
            plt.imshow(all_bit_planes[c][lsb_idx], cmap='gray')
            plt.title(f'{channel_names[c]} Bit {lsb_idx} (LSB)')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Funcionalidade 8: Segmentação de objetos baseada em cor usando HSV (e8.py)
# -----------------------------------------------------------------------------
def color_based_segmentation():
    """
    Segmenta objetos baseados em cor usando limiarização no espaço HSV.
    """
    print("\nRealizando segmentação baseada em cor...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Converter para HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Presets de cores comuns para segmentação
    color_presets = {
        "vermelho": [(0, 50, 50), (10, 255, 255), (160, 50, 50), (180, 255, 255)],  # Vermelho tem dois ranges em HSV
        "verde": [(35, 50, 50), (85, 255, 255)],
        "azul": [(100, 50, 50), (140, 255, 255)],
        "amarelo": [(20, 100, 100), (35, 255, 255)],
        "laranja": [(10, 100, 100), (25, 255, 255)],
        "roxo": [(125, 50, 50), (155, 255, 255)],
        "rosa": [(145, 50, 50), (165, 255, 255)],
        "personalizado": None  # Será preenchido pelo usuário se escolhido
    }
    
    # Menu de cores
    print("\nEscolha uma cor para segmentar:")
    colors = list(color_presets.keys())
    for i, color in enumerate(colors):
        print(f"{i+1}. {color.capitalize()}")
    
    try:
        color_choice = int(input("Número da cor: ")) - 1
        if color_choice < 0 or color_choice >= len(colors):
            print("Opção inválida. Usando vermelho.")
            color_choice = 0
    except ValueError:
        print("Entrada inválida. Usando vermelho.")
        color_choice = 0
    
    color_name = colors[color_choice]
    
    # Se o usuário escolher personalizado, solicitar os valores
    if color_name == "personalizado":
        print("\nDigite os valores de limiar HSV (0-180, 0-255, 0-255):")
        try:
            h_min = int(input("H mínimo: "))
            s_min = int(input("S mínimo: "))
            v_min = int(input("V mínimo: "))
            h_max = int(input("H máximo: "))
            s_max = int(input("S máximo: "))
            v_max = int(input("V máximo: "))
            color_presets["personalizado"] = [(h_min, s_min, v_min), (h_max, s_max, v_max)]
        except ValueError:
            print("Valores inválidos. Usando vermelho.")
            color_name = "vermelho"
    
    # Criar máscara para a cor selecionada
    ranges = color_presets[color_name]
    
    if len(ranges) == 4:  # Para vermelho que tem dois ranges
        lower1, upper1, lower2, upper2 = ranges
        mask1 = cv2.inRange(img_hsv, np.array(lower1), np.array(upper1))
        mask2 = cv2.inRange(img_hsv, np.array(lower2), np.array(upper2))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower, upper = ranges
        mask = cv2.inRange(img_hsv, np.array(lower), np.array(upper))
    
    # Aplicar operações morfológicas para melhorar a segmentação
    kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    
    # Aplicar a máscara na imagem original
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_closed)
    
    # Encontrar contornos para visualização
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar contornos na imagem original
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Exibir resultados
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(mask_closed, cmap='gray')
    plt.title(f'Máscara ({color_name.capitalize()})')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(segmented)
    plt.title('Imagem Segmentada')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(contour_img)
    plt.title('Contornos Detectados')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Segmentação de Cor - {color_name.capitalize()}: {os.path.basename(img_path)}", fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.show()

# -----------------------------------------------------------------------------
# Funcionalidade 9: Conversão e visualização no espaço de cores YIQ (e9.py)
# -----------------------------------------------------------------------------
def convert_to_yiq():
    """
    Converte uma imagem RGB para o espaço de cores YIQ e visualiza os canais.
    """
    print("\nConvertendo para espaço de cores YIQ...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalizar para [0,1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # Implementação manual da conversão RGB para YIQ
    # Matriz de transformação
    transform = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    
    # Reshape para aplicar a transformação
    pixels = img_norm.reshape(-1, 3)
    yiq_pixels = np.dot(pixels, transform.T)
    yiq = yiq_pixels.reshape(img_norm.shape)
    
    # Separar os canais Y, I e Q
    y = yiq[:,:,0]
    i = yiq[:,:,1]
    q = yiq[:,:,2]
    
    # Normalizar I e Q para visualização (podem ter valores negativos)
    i_normalized = (i - np.min(i)) / (np.max(i) - np.min(i))
    q_normalized = (q - np.min(q)) / (np.max(q) - np.min(q))
    
    # Reconstrução para RGB
    # Matriz inversa
    inverse_transform = np.linalg.inv(transform)
    
    # Aplicar transformação inversa
    rgb_pixels = np.dot(yiq_pixels, inverse_transform.T)
    rgb_reconstructed = np.clip(rgb_pixels.reshape(img_norm.shape), 0, 1)
    
    # Converter de volta para uint8
    rgb_reconstructed_uint8 = (rgb_reconstructed * 255).astype(np.uint8)
    
    # Exibir resultados
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(img_rgb)
    plt.title('Imagem Original (RGB)')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(y, cmap='gray')
    plt.title('Canal Y (Luminância)')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(i_normalized, cmap='gray')
    plt.title('Canal I (normalizado)')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(q_normalized, cmap='gray')
    plt.title('Canal Q (normalizado)')
    plt.axis('off')
    
    plt.subplot(235)
    # Visualização YIQ como pseudocor
    yiq_pseudo = np.zeros_like(img_norm)
    yiq_pseudo[:,:,0] = y  # Y no canal R
    yiq_pseudo[:,:,1] = i_normalized  # I no canal G
    yiq_pseudo[:,:,2] = q_normalized  # Q no canal B
    plt.imshow(yiq_pseudo)
    plt.title('YIQ (Pseudocor)')
    plt.axis('off')
    
    plt.subplot(236)
    plt.imshow(rgb_reconstructed_uint8)
    plt.title('RGB Reconstruído de YIQ')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Conversão para Espaço de Cores YIQ: {os.path.basename(img_path)}", fontsize=16)
    plt.subplots_adjust(top=0.90)
    plt.show()
    
    print("\nNota: O espaço de cores YIQ era usado pelo sistema de TV NTSC nos EUA.")
    print("Y contém informação de luminância (brilho), enquanto I e Q contêm informação de crominância (cor).")

# -----------------------------------------------------------------------------
# Funcionalidade 10: Equalização de histograma em diferentes espaços de cores (e10.py)
# -----------------------------------------------------------------------------
def color_histogram_equalization():
    """
    Aplica equalização de histograma em diferentes espaços de cores.
    """
    print("\nRealizando equalização de histograma em diferentes espaços de cores...")
    
    # Lista todas as imagens
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Permite o usuário escolher uma imagem
    print("Imagens disponíveis:")
    for i, file in enumerate(img_files):
        print(f"{i+1}. {file}")
    
    try:
        choice = int(input("Escolha uma imagem pelo número: ")) - 1
        if choice < 0 or choice >= len(img_files):
            print("Opção inválida. Usando a primeira imagem.")
            choice = 0
    except ValueError:
        print("Entrada inválida. Usando a primeira imagem.")
        choice = 0
    
    img_path = os.path.join(img_dir, img_files[choice])
    
    # Carrega a imagem
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Equalização direta nos canais RGB (NÃO RECOMENDADO)
    r, g, b = cv2.split(img_rgb)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    rgb_eq = cv2.merge([r_eq, g_eq, b_eq])
    
    # 2. Equalização apenas no canal V do HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    hsv_eq_rgb = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)
    
    # 3. Equalização apenas no canal L do LAB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b_ch = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b_ch])
    lab_eq_rgb = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    
    # 4. Equalização apenas no canal Y do YCrCb
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    ycrcb_eq_rgb = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    
    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization) no canal Y
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    ycrcb_clahe_rgb = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)
    
    # Função para plotar histogramas
    def plot_histograms(img, ax):
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
        ax.set_xlim([0, 256])
    
    # Exibir resultados e histogramas
    fig, axes = plt.subplots(5, 3, figsize=(15, 18))
    
    # Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Imagem Original')
    axes[0, 0].axis('off')
    plot_histograms(img_rgb, axes[0, 1])
    axes[0, 1].set_title('Histograma Original')
    axes[0, 2].axis('off')  # Célula vazia
    
    # RGB direto (não recomendado)
    axes[1, 0].imshow(rgb_eq)
    axes[1, 0].set_title('Equalização em RGB\n(NÃO RECOMENDADO)')
    axes[1, 0].axis('off')
    plot_histograms(rgb_eq, axes[1, 1])
    axes[1, 1].set_title('Histograma RGB Equalizado')
    axes[1, 2].imshow(np.abs(img_rgb.astype(np.int32) - rgb_eq.astype(np.int32)).astype(np.uint8))
    axes[1, 2].set_title('Diferença')
    axes[1, 2].axis('off')
    
    # HSV (canal V)
    axes[2, 0].imshow(hsv_eq_rgb)
    axes[2, 0].set_title('Equalização do canal V (HSV)')
    axes[2, 0].axis('off')
    plot_histograms(hsv_eq_rgb, axes[2, 1])
    axes[2, 1].set_title('Histograma após Eq. HSV')
    axes[2, 2].imshow(np.abs(img_rgb.astype(np.int32) - hsv_eq_rgb.astype(np.int32)).astype(np.uint8))
    axes[2, 2].set_title('Diferença')
    axes[2, 2].axis('off')
    
    # LAB (canal L)
    axes[3, 0].imshow(lab_eq_rgb)
    axes[3, 0].set_title('Equalização do canal L (LAB)')
    axes[3, 0].axis('off')
    plot_histograms(lab_eq_rgb, axes[3, 1])
    axes[3, 1].set_title('Histograma após Eq. LAB')
    axes[3, 2].imshow(np.abs(img_rgb.astype(np.int32) - lab_eq_rgb.astype(np.int32)).astype(np.uint8))
    axes[3, 2].set_title('Diferença')
    axes[3, 2].axis('off')
    
    # YCrCb (canal Y) com CLAHE
    axes[4, 0].imshow(ycrcb_clahe_rgb)
    axes[4, 0].set_title('CLAHE no canal Y (YCrCb)')
    axes[4, 0].axis('off')
    plot_histograms(ycrcb_clahe_rgb, axes[4, 1])
    axes[4, 1].set_title('Histograma após CLAHE')
    axes[4, 2].imshow(np.abs(img_rgb.astype(np.int32) - ycrcb_clahe_rgb.astype(np.int32)).astype(np.uint8))
    axes[4, 2].set_title('Diferença')
    axes[4, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Equalização de Histograma em Diferentes Espaços de Cor: {os.path.basename(img_path)}", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    print("\nObservação: A equalização direta nos canais RGB não é recomendada porque")
    print("altera as relações entre os canais, distorcendo as cores da imagem.")
    print("É preferível equalizar apenas os canais de luminância (Y, L, V) nos espaços")
    print("YCrCb, LAB ou HSV para preservar as informações de cor original.")

# -----------------------------------------------------------------------------
# Menu principal e execução do programa
# -----------------------------------------------------------------------------
def main():
    while True:
        print("\n=== Processamento de Imagens Coloridas ===")
        print("1. Gerar histogramas RGB de todas as imagens (e1.py)")
        print("2. Analisar canais de imagens coloridas (e2.py)")
        print("3. Converter entre espaços de cores (RGB ↔ HSV, LAB, YCrCb, CMYK)")
        print("4. Comparar efeitos de borrão em RGB vs HSV")
        print("5. Aplicar filtros de detecção de bordas em imagens coloridas")
        print("6. Realizar filtragem no domínio da frequência (passa-alta e passa-baixa)")
        print("7. Visualizar e manipular planos de bits de imagens coloridas")
        print("8. Segmentação de objetos baseada em cor usando limiarização HSV")
        print("9. Converter e visualizar imagens no espaço de cores NTSC (YIQ)")
        print("10. Equalização de histograma em diferentes espaços de cores")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção: ")
        
        if choice == '1':
            generate_histograms()
        elif choice == '2':
            analyze_image_channels()
        elif choice == '3':
            convert_color_spaces()
        elif choice == '4':
            compare_blur_rgb_hsv()
        elif choice == '5':
            apply_edge_detection()
        elif choice == '6':
            frequency_domain_filtering()
        elif choice == '7':
            visualize_bit_planes()
        elif choice == '8':
            color_based_segmentation()
        elif choice == '9':
            convert_to_yiq()
        elif choice == '10':
            color_histogram_equalization()
        elif choice == '0':
            print("Encerrando programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
