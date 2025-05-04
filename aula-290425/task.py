import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
## Técnicas de Segmentação de Imagens com OpenCV e K-means ##

1. **flowers.jpg**: 
   - Técnica: **K-Means ou GrabCut**
   - Razão: Imagem colorida com várias flores diferentes onde a segmentação baseada em cores é
    mais eficaz

2. **gecko.png**: 
   - Técnica: **Otsu ou GrabCut**
   - Razão: Objeto bem definido sobre fundo contrastante

3. **rice.png**: 
   - Técnica: **Otsu ou Watershed**
   - Razão: Objetos pequenos (grãos) com bom contraste em relação ao fundo

4. **beans.png**: 
   - Técnica: **Adaptive Threshold ou Watershed**
   - Razão: Objetos (feijões) com variação de brilho

5. **blobs.png**: 
   - Técnica: **Simple Threshold ou Otsu**
   - Razão: Formas simples com alto contraste

6. **chips.png**: 
   - Técnica: **K-Means ou Watershed**
   - Razão: Objetos com textura e formas irregulares

7. **coffee.png**: 
   - Técnica: **Adaptive Threshold ou Watershed**
   - Razão: Grãos de café com textura e iluminação variável

8. **dowels.tif**: 
   - Técnica: **Canny + Contornos ou Watershed**
   - Razão: Objetos circulares bem definidos
"""


def apply_simple_threshold(img, threshold=127):
    """Aplica limiarização simples."""
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def apply_otsu_threshold(img):
    """Aplica limiarização de Otsu."""
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def apply_adaptive_threshold(img):
    """Aplica limiarização adaptativa."""
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    return thresh

def apply_kmeans_segmentation(img, k=3):
    """Aplica segmentação K-means baseada em cores."""
    # Reshape da imagem para ser uma lista de pixels RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Aplicar K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Converter de volta para formato uint8 e reshape para a imagem original
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(img_rgb.shape)
    
    # Converter para escala de cinza para comparar com outras técnicas
    segmented_gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
    _, segmented_binary = cv2.threshold(segmented_gray, 127, 255, cv2.THRESH_BINARY)
    
    return segmented_binary

def apply_canny_edges(img, threshold1=100, threshold2=200):
    """Aplica detecção de bordas Canny."""
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    edges = cv2.Canny(gray, threshold1, threshold2)
    # Preencher os contornos para melhor visualização
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(gray)
    cv2.drawContours(result, contours, -1, 255, -1)  # Preencher contornos
    return result

def apply_watershed(img):
    """Aplica o algoritmo watershed."""
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Aplicar limiar de Otsu para obter uma máscara inicial
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operações morfológicas para limpar o fundo
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Dilatação para determinar a área de fundo
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Transformada de distância para encontrar a área de primeiro plano
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Região desconhecida (nem fundo nem primeiro plano)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marcação dos componentes
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Aplicar watershed
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    
    # Criar imagem de resultado
    result = np.zeros_like(gray)
    result[markers == -1] = 255  # Marcar as fronteiras
    
    # Preencher as regiões de interesse
    for i in range(2, np.max(markers) + 1):
        result[markers == i] = 255
    
    return result

def apply_grabcut(img):
    """Aplica o algoritmo GrabCut com uma inicialização automática."""
    # Criar máscara inicial (retângulo no centro da imagem)
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Definir um retângulo no centro da imagem
    height, width = img.shape[:2]
    rect = (width//4, height//4, width//2, height//2)
    
    # Criar modelos temporários para o algoritmo
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    # Aplicar GrabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modificar a máscara para criar uma máscara binária
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Aplicar a máscara à imagem
    result = mask2 * 255
    
    return result

# tudo isso por causa do arroz...
def process_image(img_path):
    """Processa uma imagem com diferentes técnicas de segmentação."""
    # Tentativa 1: Carregar normalmente
    img = cv2.imread(img_path)
    
    # Tentativa 2: Se falhar, tente tratar como TIFF mesmo com extensão .png
    if img is None and img_path.endswith('.png'):
        try:
            # Algumas imagens como 'rice.png' são na verdade arquivos TIFF com extensão .png
            import imageio.v2 as imageio
            img = imageio.imread(img_path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            print(f"Imagem {img_path} carregada usando imageio (formato TIFF)")
        except Exception as e:
            print(f"Erro ao tentar carregar com imageio: {e}")
    
    # Tentativa 3: Se ainda falhar, tente carregar uma versão alternativa da imagem
    if img is None:
        alt_path = img_path.replace('.png', '.tif')
        img = cv2.imread(alt_path)
        if img is not None:
            print(f"Imagem carregada de caminho alternativo: {alt_path}")
    
    if img is None:
        print(f"Erro ao carregar a imagem: {img_path}")
        return
    
    # Aplicando diferentes técnicas
    simple_thresh = apply_simple_threshold(img)
    otsu_thresh = apply_otsu_threshold(img)
    adaptive_thresh = apply_adaptive_threshold(img)
    kmeans_seg = apply_kmeans_segmentation(img)
    canny_edges = apply_canny_edges(img)
    watershed_result = apply_watershed(img)
    grabcut_result = apply_grabcut(img)
    
    # Plota... 
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(simple_thresh, cmap='gray')
    plt.title('Limiarização Simples')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title('Limiarização de Otsu')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title('Limiarização Adaptativa')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(kmeans_seg, cmap='gray')
    plt.title('K-Means (k=3)')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Detecção de Contornos (Canny)')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(watershed_result, cmap='gray')
    plt.title('Watershed')
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(grabcut_result, cmap='gray')
    plt.title('GrabCut')
    plt.axis('off')
    
    plt.suptitle(f'Técnicas de Segmentação: {img_path.split("/")[-1]}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def main():
    image_paths = [
        "/home/zerocopia/Ufal/dip-2024-2/img/flowers.jpg",
        "/home/zerocopia/Ufal/dip-2024-2/img/gecko.png",
        "/home/zerocopia/Ufal/dip-2024-2/img/rice.tif",
        "/home/zerocopia/Ufal/dip-2024-2/img/beans.png",
        "/home/zerocopia/Ufal/dip-2024-2/img/blobs.png", 
        "/home/zerocopia/Ufal/dip-2024-2/img/chips.png",
        "/home/zerocopia/Ufal/dip-2024-2/img/coffee.png",
        "/home/zerocopia/Ufal/dip-2024-2/img/dowels.tif",
    ]

    print("Escolha uma imagem para processar:")
    for i, path in enumerate(image_paths):
        print(f"{i+1}. {path.split('/')[-1]}")
    
    try:
        choice = int(input("Digite o número da imagem (1-8): ")) - 1
        if 0 <= choice < len(image_paths):
            process_image(image_paths[choice])
        else:
            print("Escolha inválida.")
    except ValueError:
        print("Por favor, digite um número válido.")
    
    # Todas as imagens em sequência
    process_all = input("Deseja processar todas as imagens? (s/n): ").lower()
    if process_all == 's':
        for img_path in image_paths:
            process_image(img_path)

if __name__ == "__main__":
    main()