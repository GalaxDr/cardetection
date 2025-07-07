import os
import cv2
import numpy as np
from collections import Counter


def classify_color(bgr_color):
    """
    Classifica a cor BGR de um objeto tentando inferir a cor real
    a partir da cor aparente, lidando com condições de baixa iluminação e saturação.
    """
    b, g, r = bgr_color
    #[81.96428571 73.96428571 75.50510204]
    # Normaliza os valores BGR para a faixa [0, 255]
    b, g, r = int(b), int(g), int(r)

    mean = (np.mean([b, g, r]) + 0.5) / 255.0

    print(f"Valores normalizados: B={b}, G={g}, R={r}, Média={mean}")
    if mean < 0.3:
        return 'Preto'  # Baixa iluminação, assume preto
    elif mean > 0.8:
        return 'Branco'
    elif b > 150 and g < 100 and r < 100:
        return 'Azul'
    elif b < 100 and g > 150 and r < 100:
        return 'Verde'
    elif b < 100 and g < 100 and r > 135:
        return 'Vermelho'
    elif b > 100 and g > 100 and r < 100:
        return 'Amarelo'
    elif b < 100 and g > 100 and r > 100:
        return 'Ciano'
    elif b > 100 and g < 100 and r > 100:
        return 'Magenta'
    elif b > 100 and g > 100 and r > 100:
        return 'Cinza'
    else:
        return 'N/D'








def get_car_color_robust(roi_bgr, show_debug=False):
    """
    Implementa uma abordagem de amostragem em 5 pontos em forma de cruz (+)
    com prioridade central em caso de ambiguidade.
    """
    if roi_bgr is None or roi_bgr.size < 400:
        return 'N/D'

    h, w, _ = roi_bgr.shape

    # 1. Definir as coordenadas dos 5 pontos de amostragem
    center_x, center_y = w // 2, h // 2
    points = {
        'centro': (center_x + 10, center_y),
        'esquerda': (w // 4, center_y),
        'direita': (w * 3 // 4, center_y),
        'cima': (center_x, h // 8),
        'baixo': (center_x, h * 3 // 4),
    }

    SAMPLE_SIZE = max(10, min(h, w) // 10)
    half_ss = SAMPLE_SIZE // 2

    classifications = {}

    debug_img = roi_bgr.copy() if show_debug else None

    # 2. Extrair e classificar a cor de cada pequena área
    for name, (px, py) in points.items():
        y1, y2 = max(0, py - half_ss), min(h, py + half_ss)
        x1, x2 = max(0, px - half_ss), min(w, px + half_ss)

        sample_area = roi_bgr[y1:y2, x1:x2]

        if sample_area.size == 0:
            continue

        avg_bgr_color = np.mean(sample_area, axis=(0, 1))
        classifications[name] = classify_color(avg_bgr_color)

        if show_debug:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(debug_img, classifications[name], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if not classifications:
        return "N/D"

    if show_debug:
        print("--- Classificações Individuais (5 Pontos) ---")
        for name in ['cima', 'esquerda', 'centro', 'direita', 'baixo']:
            if name in classifications:
                print(f"{name.capitalize():<10}: {classifications[name]}")
        cv2.imshow("Amostragem em 5 Pontos", debug_img)
        cv2.waitKey(1)

    # 3. Sistema de Votação para 5 amostras com Prioridade Central
    class_list = list(classifications.values())
    counts = Counter(class_list)

    # Se não houver votos, retorna N/D
    if not counts:
        return 'N/D'

    winner, win_count = counts.most_common(1)[0]

    # Se uma cor tem 3 ou mais votos (maioria), ela é a vencedora.
    if win_count >= 3:
        return winner
    else:
        # Se não há maioria (contagem máxima é 2 ou 1), há ambiguidade.
        # Nesses casos, a prioridade do ponto central decide.
        return classifications.get('centro', 'N/D')


# --- CONFIGURAÇÃO INICIAL ---

# Constrói o caminho absoluto para o vídeo
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'Estacionamento.mp4')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo em '{video_path}'.")
    exit()

# Coordenadas das 4 vagas de estacionamento (x, y, largura, altura)
vagas = [
    (5, 90, 180, 350),  # Vaga 1 (esquerda)
    (240, 60, 180, 380),  # Vaga 2 (centro-esquerda)
    (500, 60, 180, 380),  # Vaga 3 (centro-direita)
    (700, 60, 140, 340)  # Vaga 4 (direita)
]

# --- PROCESSAMENTO DO VÍDEO ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro ao ler o frame. Finalizando...")
        break

    vagas_disponiveis = 0


    def gray_blur_canny(video):
        """
        Aplica conversão para escala de cinza, desfoque e detecção de bordas Canny.
        """
        gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        canny = cv2.Canny(blur, 50, 150)
        return canny


    # Itera sobre cada vaga definida
    for i, (x_vaga, y_vaga, w_vaga, h_vaga) in enumerate(vagas):
        # 1. Recorta a região da vaga do frame original
        vaga_roi = frame[y_vaga:y_vaga + h_vaga, x_vaga:x_vaga + w_vaga]

        # 2. Converte para escala de cinza e aplica um desfoque para suavizar a imagem
        vaga_canny = gray_blur_canny(vaga_roi)

        # 4. Conta os pixels de borda (pixels brancos na imagem Canny)
        edge_pixels = cv2.countNonZero(vaga_canny)

        # 5. Define um limiar de bordas para considerar a vaga ocupada.
        # ESTE VALOR É CRÍTICO E PODE PRECISAR DE AJUSTE!
        # Um carro tem muitas bordas, uma vaga vazia tem poucas.
        edge_threshold = 950  # Valor inicial para teste, ajuste conforme necessário

        if edge_pixels > edge_threshold:
            # Vaga OCUPADA
            cor_carro = get_car_color_robust(vaga_roi, show_debug=True)
            cv2.rectangle(frame, (x_vaga, y_vaga), (x_vaga + w_vaga, y_vaga + h_vaga), (0, 0, 255), 2)
            cv2.putText(frame, f'Vaga {i + 1}: Ocupada', (x_vaga, y_vaga - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            cv2.putText(frame, f'Cor: {cor_carro}', (x_vaga, y_vaga + h_vaga + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
        else:
            # Vaga LIVRE
            vagas_disponiveis += 1
            cv2.rectangle(frame, (x_vaga, y_vaga), (x_vaga + w_vaga, y_vaga + h_vaga), (0, 255, 0), 2)
            cv2.putText(frame, f'Vaga {i + 1}: Livre', (x_vaga, y_vaga - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    texto_vagas = f'Vagas Livres: {vagas_disponiveis}/{len(vagas)}'
    cv2.putText(frame, texto_vagas, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detector de Vagas', frame)

    vaga_canny = gray_blur_canny(frame)
    cv2.imshow('Vaga Canny', vaga_canny)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
