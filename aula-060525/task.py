import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from skimage import data, color

def hit_or_miss(img, hit_kernel, miss_kernel):
    """
    Aplica a transformada Hit-or-Miss.
    """
    erosion_img = cv2.erode(img, hit_kernel, iterations=1)
    erosion_comp = cv2.erode(255 - img, miss_kernel, iterations=1)
    return cv2.bitwise_and(erosion_img, erosion_comp)

def thinning(img):
    """
    Aplica o algoritmo de afinamento usando apenas hit-or-miss.
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    se1 = [
        (np.array([[0, 0, 0],
                   [0, 1, 0],
                   [1, 1, 1]], dtype=np.uint8),
         np.array([[1, 1, 1],
                   [0, 0, 0],
                   [0, 0, 0]], dtype=np.uint8)),
        
        (np.array([[0, 0, 1],
                   [0, 1, 1],
                   [0, 0, 1]], dtype=np.uint8),
         np.array([[1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0]], dtype=np.uint8)),
        
        (np.array([[1, 1, 1],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=np.uint8),
         np.array([[0, 0, 0],
                   [0, 0, 0],
                   [1, 1, 1]], dtype=np.uint8)),
        
        (np.array([[1, 0, 0],
                   [1, 1, 0],
                   [1, 0, 0]], dtype=np.uint8),
         np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]], dtype=np.uint8))
    ]
    
    se2 = [
        (np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 0]], dtype=np.uint8),
         np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.uint8)),
        
        (np.array([[1, 0, 0],
                   [1, 1, 0],
                   [0, 0, 1]], dtype=np.uint8),
         np.array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 1, 0]], dtype=np.uint8)),
        
        (np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 0]], dtype=np.uint8),
         np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.uint8)),
        
        (np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=np.uint8),
         np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]], dtype=np.uint8))
    ]
    
    all_se = se1 + se2
    
    prev = np.zeros_like(binary)
    curr = binary.copy()
    
    iteration = 0
    while np.sum(prev != curr) > 0:
        prev = curr.copy()
        iteration += 1
        
        for hit_kernel, miss_kernel in all_se:
            hm = hit_or_miss(curr, hit_kernel, miss_kernel)
            curr = cv2.subtract(curr, hm)
        
        print(f"Iteração {iteration}: {np.sum(prev != curr)} pixels alterados")
        
        if iteration > 100:
            print("Número máximo de iterações atingido")
            break
    
    return curr

def geodesic_dilation(marker, mask, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(marker, se)
    return np.minimum(dilated, mask)

def reconstruction_by_dilation(marker, mask, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    marker = np.minimum(marker, mask)
    prev = np.zeros_like(marker)
    curr = marker.copy()
    iteration = 0
    while np.any(curr != prev):
        prev = curr.copy()
        curr = geodesic_dilation(curr, mask, se)
        iteration += 1
        if iteration > 100:
            print("Reconstrução: Número máximo de iterações atingido")
            break
    return curr

def reconstruction_by_erosion(marker, mask, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    marker = np.maximum(marker, mask)
    prev = np.zeros_like(marker)
    curr = marker.copy()
    iteration = 0
    while np.any(curr != prev):
        prev = curr.copy()
        eroded = cv2.erode(curr, se)
        curr = np.maximum(eroded, mask)
        iteration += 1
        if iteration > 100:
            print("Reconstrução: Número máximo de iterações atingido")
            break
    return curr

def opening_by_reconstruction(img, se=None, se_recon=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(img, se)
    return reconstruction_by_dilation(eroded, img, se_recon)

def remove_border_objects(img):
    markers = np.zeros_like(img)
    border_mask = np.zeros_like(img)
    border_mask[0, :] = img[0, :]
    border_mask[-1, :] = img[-1, :]
    border_mask[:, 0] = img[:, 0]
    border_mask[:, -1] = img[:, -1]
    if np.any(border_mask > 0):
        labeled, _ = ndimage.label(img, structure=np.ones((3, 3), dtype=np.int32))
        border_labels = labeled[border_mask > 0]
        border_labels = np.unique(border_labels)
        border_labels = border_labels[border_labels > 0]
        border_objects = np.zeros_like(img)
        for label in border_labels:
            border_objects[labeled == label] = 255
        return cv2.subtract(img, border_objects)
    else:
        return img

def grayscale_dilation(img, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(img, se)

def grayscale_erosion(img, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(img, se)

def grayscale_opening(img, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, se)

def grayscale_closing(img, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

def update_image(img, mask, operation='median'):
    result = img.copy()
    if operation == 'median':
        filtered = cv2.medianBlur(img, 5)
    elif operation == 'mean':
        filtered = cv2.blur(img, (5, 5))
    elif operation == 'min':
        filtered = grayscale_erosion(img)
    elif operation == 'max':
        filtered = grayscale_dilation(img)
    else:
        raise ValueError("Operação não suportada")
    result[mask > 0] = filtered[mask > 0]
    return result

def morphological_gradient(img, se=None):
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)
    dilated = grayscale_dilation(img, se)
    eroded = grayscale_erosion(img, se)
    return cv2.subtract(dilated, eroded)

def top_hat(img, se=None):
    if se is None:
        se = np.ones((5, 5), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)

def bottom_hat(img, se=None):
    if se is None:
        se = np.ones((5, 5), dtype=np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)

def granulometry(img, sizes=range(1, 30, 2)):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    pattern_spectrum = []
    previous_area = np.sum(binary) / 255
    for size in sizes:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened = grayscale_opening(binary, se)
        current_area = np.sum(opened) / 255
        pattern_spectrum.append((previous_area - current_area) / (previous_area if previous_area > 0 else 1))
        previous_area = current_area
    return sizes, pattern_spectrum

def texture_segmentation(img, se_size=7):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    top = top_hat(gray, se)
    bottom = bottom_hat(gray, se)
    texture_enhanced = cv2.add(top, bottom)
    texture_segmented = cv2.adaptiveThreshold(
        texture_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return texture_segmented

def demo_skeleton(save_path):
    """Esqueletonização usando a imagem 'horse.png'"""
    horse_img_path = 'img/horse.png'
    if not os.path.exists(horse_img_path):
        print(f"Erro: Imagem '{horse_img_path}' não encontrada!")
        return False
    
    img = cv2.imread(horse_img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skeleton = thinning(binary)
    
    # Mostrar resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Imagem Binária Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Esqueletização por Afinamento')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/1_skeleton.png')
    plt.close()
    return True

def demo_morphological_reconstruction(save_path):
    """Reconstrução morfológica usando a mesma 'coins' do professor"""
    # Carregar imagem de moedas do scikit-image
    coins = data.coins()
    
    # Binarizar a imagem
    _, binary = cv2.threshold(coins, 127, 255, cv2.THRESH_BINARY)
    
    # Criar marcador (erosão da imagem original)
    marker = cv2.erode(binary, np.ones((5, 5), dtype=np.uint8))
    
    # 2.1 Dilatação geodésica
    geodesic_dilated = geodesic_dilation(marker, binary)
    
    # 2.2 Reconstrução por dilatação
    recon_dilation = reconstruction_by_dilation(marker, binary)
    
    # 2.2 Reconstrução por erosão (usando complemento)
    recon_erosion = reconstruction_by_erosion(255 - marker, 255 - binary)
    recon_erosion = 255 - recon_erosion  # Inverter de volta
    
    # 2.3 Abertura por reconstrução
    opened_recon = opening_by_reconstruction(binary)
    
    # 2.4 Eliminação de elementos de borda
    no_border = remove_border_objects(binary)
    
    # Mostrar resultados
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Original (Moedas)')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(marker, cmap='gray')
    plt.title('Marcador (Erodido)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(geodesic_dilated, cmap='gray')
    plt.title('2.1 Dilatação Geodésica')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(recon_dilation, cmap='gray')
    plt.title('2.2 Reconstrução por Dilatação')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(opened_recon, cmap='gray')
    plt.title('2.3 Abertura por Reconstrução')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(no_border, cmap='gray')
    plt.title('2.4 Eliminação de Elementos de Borda')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/2_reconstruction.png')
    plt.close()
    return True

def demo_grayscale_basic(save_path):
    """Operações básicas em escala de cinza usando imagem 'camera'"""
    # Pegando a imagem da câmera do scikit-image
    camera = data.camera()
    
    # 4.1 Dilatação/erosão
    dilated = grayscale_dilation(camera)
    eroded = grayscale_erosion(camera)
    
    # 4.2 Abertura/fechamento
    opened = grayscale_opening(camera)
    closed = grayscale_closing(camera)
    
    # 4.3 Atualização
    _, mask = cv2.threshold(camera, 150, 255, cv2.THRESH_BINARY)
    updated = update_image(camera, mask)
    
    # Mostrar resultados
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(camera, cmap='gray')
    plt.title('Original (Camera)')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(dilated, cmap='gray')
    plt.title('4.1 Dilatação')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(eroded, cmap='gray')
    plt.title('4.1 Erosão')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(opened, cmap='gray')
    plt.title('4.2 Abertura')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(closed, cmap='gray')
    plt.title('4.2 Fechamento')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(updated, cmap='gray')
    plt.title('4.3 Atualização')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/4_1_2_3_basic_grayscale.png')
    plt.close()
    return True

def demo_grayscale_advanced(save_path):
    """Operações avançadas em escala de cinza"""
    # 4.4 Gradiente morfológico - usando astronaut
    astronaut = color.rgb2gray(data.astronaut())
    astronaut = (astronaut * 255).astype(np.uint8)
    gradient = morphological_gradient(astronaut)
    
    # 4.5 Top-hat/bottom-hat - com iluminação desigual
    moon = data.moon()
    tophat = top_hat(moon)
    bothat = bottom_hat(moon)
    
    # resultados
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(gradient, cmap='gray')
    plt.title('4.4 Gradiente Morfológico (Astronaut)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(tophat, cmap='gray')
    plt.title('4.5 Top-Hat (Moon)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(bothat, cmap='gray')
    plt.title('4.5 Bottom-Hat (Moon)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/4_4_5_gradient_hat.png')
    plt.close()
    return True

def demo_granulometry(save_path):
    """Granulometria usando a imagem 'coffee'"""
    try:
        coffee = data.coffee()
        coffee_gray = color.rgb2gray(coffee)
        coffee_gray = (coffee_gray * 255).astype(np.uint8)
    except:
        # Se não estiver disponível, usa outra imagem
        coffee_gray = data.coins()
    
    _, binary = cv2.threshold(coffee_gray, 127, 255, cv2.THRESH_BINARY)
    sizes = range(1, 30, 2)
    sizes, spectrum = granulometry(binary, sizes)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, spectrum)
    plt.title('4.6 Espectro de Padrão (Granulometria)')
    plt.xlabel('Tamanho do Elemento Estruturante')
    plt.ylabel('Mudança Relativa de Área')
    plt.grid(True)
    plt.savefig(f'{save_path}/4_6_granulometry.png')
    plt.close()
    
    # Mostrando a original para referência
    plt.figure(figsize=(8, 8))
    plt.imshow(coffee_gray, cmap='gray')
    plt.title('Imagem utilizada para Granulometria')
    plt.axis('off')
    plt.savefig(f'{save_path}/4_6_original.png')
    plt.close()
    return True

def demo_texture_segmentation(save_path):
    """Segmentação de texturas usando a imagem 'grass'"""
    try:
        grass = data.grass()
        grass_gray = color.rgb2gray(grass)
        grass_gray = (grass_gray * 255).astype(np.uint8)
    except:
        # Se não estiver disponível, usa outra imagem
        retina = data.retina()
        grass_gray = color.rgb2gray(retina)
        grass_gray = (grass_gray * 255).astype(np.uint8)
    
    textures = texture_segmentation(grass_gray)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(grass_gray, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(textures, cmap='gray')
    plt.title('4.7 Segmentação de Texturas')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/4_7_texture_segmentation.png')
    plt.close()
    return True

def run_all_demos():
    """Com as imagens do scikit-image"""
    results_dir = 'resultados/'
    os.makedirs(results_dir, exist_ok=True)
    
    demos = [
        (demo_skeleton, "1. Esqueletonização"),
        (demo_morphological_reconstruction, "2. Reconstrução Morfológica"),
        (demo_grayscale_basic, "4.1-4.3 Operações Básicas em Escala de Cinza"),
        (demo_grayscale_advanced, "4.4-4.5 Gradiente e Top/Bottom-Hat"),
        (demo_granulometry, "4.6 Granulometria"),
        (demo_texture_segmentation, "4.7 Segmentação de Texturas")
    ]
    
    for demo_func, description in demos:
        print(f"Executando: {description}")
        success = demo_func(results_dir)
        if success:
            print(f"✓ {description} - concluído com sucesso.")
        else:
            print(f"✗ {description} - erro durante a execução.")
    
    print(f"\nTodas as demonstrações foram concluídas! Resultados salvos em: {results_dir}")

if __name__ == "__main__":
    run_all_demos()