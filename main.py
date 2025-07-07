import os
import cv2
import numpy as np
from collections import Counter


def classify_color_from_equalized(bgr_color):
    """
    Nova função de classificação projetada para trabalhar com cores que JÁ FORAM
    normalizadas por uma forte equalização (CLAHE).

    As regras são mais diretas, pois esperamos cores mais 'puras' e vibrantes.
    """
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color[0], hsv_color[1], hsv_color[2]
    print(f"HSV: {h}, {s}, {v}")

    # Após a equalização, esperamos valores de V e S mais altos em geral.
    # Os limiares aqui são calibrados para essa nova realidade de alto contraste.

    # REGRA 1: PRETO
    # Mesmo após a equalização, áreas pretas puras devem ter baixo brilho (V).
    if v < 60:
        return 'Preto'

    # REGRA 2: BRANCO
    # Branco equalizado terá brilho máximo (V) e saturação quase nula (S).
    if s < 30 and v > 110:
        return 'Branco'

    # REGRA 3: CINZA
    # Cinza equalizado terá baixa saturação (S), mas brilho mediano.
    if s < 35:
        return 'Cinza'

    # REGRA 4: CORES CROMÁTICAS
    # Com o contraste e brilho normalizados, podemos confiar mais no Matiz (H).
    if (h < 10) or (h > 168):
        return 'Vermelho'
    elif h < 25:
        return 'Laranja'
    elif h < 40:
        return 'Amarelo'
    elif h < 85:
        return 'Verde'
    elif h < 135:
        return 'Azul'
    elif h < 155:
        return 'Roxo'
    else:
        return 'Rosa'


def get_car_color(roi_bgr, show_debug=False):
    """
    Pipeline final:
    1. Aplica uma FORTE equalização de contraste (CLAHE) para normalizar a iluminação.
    2. Usa a amostragem em 5 pontos na imagem normalizada.
    3. Classifica as amostras com a nova função `classify_color_from_equalized`.
    """
    if roi_bgr is None or roi_bgr.size < 400:
        return 'N/D'

    # --- 1. Normalização Agressiva com CLAHE ---
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Aumentamos o clipLimit para uma equalização BEM mais forte.
    # VALOR PARA AJUSTAR: Aumente para mais contraste, diminua para menos. 8.0 é um valor forte.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(v)

    hsv_equalized = cv2.merge([h, s, v_equalized])
    roi_bgr_processed = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

    h, w, _ = roi_bgr_processed.shape

    # --- 2. Amostragem em 5 Pontos ---
    center_x, center_y = w // 2, h // 2
    points = {
        'centro': (center_x + 15, center_y), 'esquerda': (w // 4 + 10, center_y),
        'direita': (w * 3 // 4, center_y), 'cima': (center_x, h // 4),
        'baixo': (center_x, h * 3 // 4 - 40),
    }

    SAMPLE_SIZE = max(10, min(h, w) // 10)
    half_ss = SAMPLE_SIZE // 2
    classifications = {}

    # --- 3. Classificação com a Nova Função ---
    for name, (px, py) in points.items():
        y1, y2 = max(0, py - half_ss), min(h, py + half_ss)
        x1, x2 = max(0, px - half_ss), min(w, px + half_ss)

        sample_area = roi_bgr_processed[y1:y2, x1:x2]
        if sample_area.size == 0:
            continue

        avg_bgr_color = np.mean(sample_area, axis=(0, 1))
        # Chamando a NOVA função de classificação
        classifications[name] = classify_color_from_equalized(avg_bgr_color)

    # Lógica de votação e depuração
    if not classifications:
        return "N/D"

    if show_debug:
        print("\n--- Classificações Individuais (após Equalização Forte) ---")
        debug_img = roi_bgr_processed.copy()
        for name in ['cima', 'esquerda', 'centro', 'direita', 'baixo']:
            if name in classifications:
                (px, py) = points[name]
                x1, y1 = max(0, px - half_ss), max(0, py - half_ss)
                cv2.rectangle(debug_img, (x1, y1), (x1 + SAMPLE_SIZE, y1 + SAMPLE_SIZE), (0, 255, 255), 2)
                print(f"{name.capitalize():<10}: {classifications[name]}")
        cv2.imshow("Amostragem na Imagem Equalizada", debug_img)

        original_resized = cv2.resize(roi_bgr, (debug_img.shape[1], debug_img.shape[0]))
        comparison = np.hstack([original_resized, debug_img])
        cv2.imshow("Original vs. Equalizada", comparison)
        cv2.waitKey(1)

    class_list = list(classifications.values())
    counts = Counter(class_list)
    if not counts:
        return 'N/D'
    winner, win_count = counts.most_common(1)[0]

    if win_count >= 3:
        return winner
    else:
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
        edge_threshold = 950 

        if edge_pixels > edge_threshold:
            # Vaga OCUPADA
            cor_carro = get_car_color(vaga_roi.copy(), show_debug=True)
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
