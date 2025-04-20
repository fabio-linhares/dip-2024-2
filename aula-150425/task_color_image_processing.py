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
# Menu principal e execução do programa
# -----------------------------------------------------------------------------
def main():
    while True:
        print("\n=== Processamento de Imagens Coloridas ===")
        print("1. Gerar histogramas RGB de todas as imagens (e1.py)")
        print("2. Analisar canais de imagens coloridas (e2.py)")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção: ")
        
        if choice == '1':
            generate_histograms()
        elif choice == '2':
            analyze_image_channels()
        elif choice == '0':
            print("Encerrando programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
