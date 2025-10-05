import cv2 as cv
import os
import argparse

def capture_faces(name, num_photos=30):
    """
    Captura fotos desde la webcam para entrenar el modelo

    Args:
        name: Nombre de la persona
        num_photos: Cantidad de fotos a capturar (default: 30)
    """
    # Crear carpeta para la persona
    person_dir = os.path.join('dataset', name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        print(f"Carpeta creada: {person_dir}")
    else:
        print(f"Usando carpeta existente: {person_dir}")

    # Cargar detector de rostros
    haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Iniciar captura de video
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    print(f"\n=== Capturando {num_photos} fotos para: {name} ===")
    print("Instrucciones:")
    print("- Mira a la cámara y mueve lentamente tu cabeza")
    print("- Diferentes ángulos: frente, derecha, izquierda, arriba, abajo")
    print("- Presiona ESPACIO para capturar manualmente")
    print("- Presiona Q para salir")
    print("\nCaptura automática cada 1 segundo...\n")

    frame_counter = 0

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo acceder a la webcam")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray, 1.3, 5)

        # Dibujar rectángulo en el rostro detectado
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"{name} - {count}/{num_photos}", (x, y - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar instrucciones en pantalla
        cv.putText(frame, "ESPACIO: capturar | Q: salir", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.imshow(f"Capturando fotos - {name}", frame)

        # Captura automática cada 30 frames (aprox. 1 segundo)
        frame_counter += 1
        if frame_counter >= 30 and len(faces) > 0:
            # Guardar la imagen del rostro detectado
            photo_path = os.path.join(person_dir, f"{name}_{count + 1}.jpg")
            cv.imwrite(photo_path, frame)
            print(f"✓ Foto {count + 1}/{num_photos} guardada")
            count += 1
            frame_counter = 0

        # Captura manual con ESPACIO
        key = cv.waitKey(1) & 0xFF
        if key == ord(' ') and len(faces) > 0:
            photo_path = os.path.join(person_dir, f"{name}_{count + 1}.jpg")
            cv.imwrite(photo_path, frame)
            print(f"✓ Foto {count + 1}/{num_photos} guardada (manual)")
            count += 1
        elif key == ord('q'):
            print("\nCaptura cancelada por el usuario")
            break

    cap.release()
    cv.destroyAllWindows()

    print(f"\n✅ Captura completada: {count} fotos guardadas en {person_dir}")
    print(f"\nPróximo paso: Ejecuta 'python train_model.py' para entrenar el modelo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Captura fotos desde webcam para entrenamiento')
    parser.add_argument('--name', type=str, required=True, help='Nombre de la persona')
    parser.add_argument('--photos', type=int, default=30, help='Número de fotos a capturar (default: 30)')

    args = parser.parse_args()

    capture_faces(args.name, args.photos)
