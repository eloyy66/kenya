import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from collections import deque


# =======================
# Configuración general
# =======================
EAR_THRESHOLD = 0.22
EYES_CLOSED_FRAMES = 15   # ~0.5s a 30 FPS
YAW_THRESHOLD = 25        # grados aprox (proxy)
WINDOW_SIZE = 30          # frames para suavizado


# =======================
# Utilidades EAR
# =======================
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_ear(eye_points):
    # eye_points: array de 6 puntos (x, y)
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)


# =======================
# Selección de puntos de ojo (InsightFace 106)
# =======================


#def get_eye_landmarks(kps):
    # Índices aproximados (funcionan bien en práctica)
    left_eye_idx = [33, 35, 37, 39, 41, 43]
    right_eye_idx = [87, 89, 91, 93, 95, 97]

    left_eye = np.array([kps[i] for i in left_eye_idx])
    right_eye = np.array([kps[i] for i in right_eye_idx])

    return left_eye, right_eye

def get_eye_landmarks(kps):
    if kps.shape[0] < 98:
        return None, None

    left_eye_idx = [33, 35, 37, 39, 41, 43]
    right_eye_idx = [87, 89, 91, 93, 95, 97]

    left_eye = np.array([kps[i] for i in left_eye_idx])
    right_eye = np.array([kps[i] for i in right_eye_idx])

    return left_eye, right_eye



# =======================
# Estado del conductor
# =======================
def decide_state(avg_ear, yaw, closed_counter):
    if closed_counter >= EYES_CLOSED_FRAMES:
        return "SOMNOLENT"
    if abs(yaw) > YAW_THRESHOLD:
        return "DISTRACTED"
    return "ATTENTIVE"


# =======================
# Main
# =======================

def find_available_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    #cameras = find_available_cameras()
    #print(f"Cámaras disponibles: {cameras}")

    #cameras = find_available_cameras()
    #print(f"Cámaras disponibles: {cameras}")

    #if not cameras:
    #    print("No se encontraron cámaras")
    #    return

    #CAMERA_INDEX = cameras[1]  # primera cámara disponible

    #cap = cv2.VideoCapture(CAMERA_INDEX)
    #cap = cv2.VideoCapture(1)  # Usar cámara por defecto
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # MacOS específico

    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara")
        return

    ear_window = deque(maxlen=WINDOW_SIZE)
    closed_counter = 0
    prev_time = 0.0

    print("Presiona 'q' para salir")

    
    for _ in range(10):
        cap.read()


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        state = "NO_FACE"

        #for face in faces:
        #    x1, y1, x2, y2 = face.bbox.astype(int)
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #    if face.kps is None:
        #        continue

        #    kps = face.kps

        #    # Dibujar landmarks
        #    #for (x, y) in kps:
        #    #    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)

        #    # solo ojos
        #    for (x, y) in left_eye:
        #        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
        #    for (x, y) in right_eye:
        #        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

            # EAR
            
            #if kps.shape[0] < 98:
            #    state = "UNCERTAIN"
            #    break

            #eyes = get_eye_landmarks(kps)
            #if eyes[0] is None:
            #    state = "UNCERTAIN"
            #    break
            #
            #left_eye, right_eye = eyes


            #left_eye, right_eye = get_eye_landmarks(kps)

            
            #ear_left = compute_ear(left_eye)
            #ear_right = compute_ear(right_eye)
            #ear = (ear_left + ear_right) / 2.0

            #ear_window.append(ear)

            #if len(ear_window) < WINDOW_SIZE:
            #    state = "UNCERTAIN"
            #    break

            #avg_ear = np.mean(ear_window)

            #if avg_ear < EAR_THRESHOLD:
            #    closed_counter += 1
            #else:
            #    closed_counter = 0

            ## Head pose (yaw proxy)
            #yaw = face.pose[1] if face.pose is not None else 0.0

            #state = decide_state(avg_ear, yaw, closed_counter)

            ## Mostrar métricas
            #cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            #cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 90),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            #break  # solo un rostro (conductor)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
            if face.kps is None:
                state = "UNCERTAIN"
                break
            
            kps = face.kps
        
            # --- VALIDACIÓN DE LANDMARKS ---
            if kps.shape[0] < 98:
                state = "UNCERTAIN"
                break
            
            eyes = get_eye_landmarks(kps)
            if eyes[0] is None:
                state = "UNCERTAIN"
                break
            
            left_eye, right_eye = eyes
        
            # --- DIBUJO (DESPUÉS de validar) ---
            for (x, y) in left_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
        
            # --- EAR ---
            ear_left = compute_ear(left_eye)
            ear_right = compute_ear(right_eye)
            ear = (ear_left + ear_right) / 2.0
        
            ear_window.append(ear)
        
            if len(ear_window) < WINDOW_SIZE:
                state = "UNCERTAIN"
                break
            
            avg_ear = np.mean(ear_window)
        
            if avg_ear < EAR_THRESHOLD:
                closed_counter += 1
            else:
                closed_counter = 0
        
            yaw = face.pose[1] if face.pose is not None else 0.0
        
            state = decide_state(avg_ear, yaw, closed_counter)
        
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
            break  # solo un rostro


        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time = current_time

        # Estado visual
        color = (0, 255, 0)
        if state == "DISTRACTED":
            color = (0, 165, 255)
        elif state == "SOMNOLENT":
            color = (0, 0, 255)

        cv2.putText(frame, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Kenya Vision - Distraction Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
