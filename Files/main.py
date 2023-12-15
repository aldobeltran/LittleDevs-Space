import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from IPython.display import Video

np.random.seed(20)

def cargarClasesCOCO():
    with open(os.path.join("Datos", "clases_COCO.txt"),"r") as f:
        clases = f.readlines()
    clases.insert(0, "__fondo__")
    colores = np.random.uniform(low=0, high=255, size=(len(clases),3))
    return clases, colores

def crearDetector():
    arquitectura = os.path.join("Datos", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    pesos_preentrenados = os.path.join("Datos", "frozen_inference_graph.pb")
    modelo = cv.dnn_DetectionModel(pesos_preentrenados, arquitectura)
    modelo.setInputSize(320,320)
    modelo.setInputScale(1.0/127.5)
    modelo.setInputMean((127.5,127.5,127.5))
    modelo.setInputSwapRB(True)
    return modelo

def cargarVideoDeteccion(ruta: str):
    video = cv.VideoCapture(ruta)
    if (video.isOpened()==False):
        print("Error al intentar abrir el archivo...")
    return video

def detectar_Y_generar(ruta_video: str):
    clases, colores = cargarClasesCOCO()
    modelo = crearDetector()
    video = cargarVideoDeteccion(ruta_video)
    ancho = video.get(3)
    alto = video.get(4)
    out = cv.VideoWriter(os.path.join("output_video.mp4"),cv.VideoWriter_fourcc(*'mpv4'), 12, (int(ancho),int(alto)))
    
    tiempo_maximo = time.time() + 60 # 60 segundos de cámara
    (exito, imagen) = video.read()
    print("Procesando video...")
    while exito and time.time()<tiempo_maximo:
        clasesEtiquetasID, confianzas, cajas =modelo.detect(imagen, confThreshold=0.5)
        cajas = list(cajas)
        confianzas = list(np.array(confianzas).reshape(1,-1)[0])
        confianzas = list(map(float, confianzas))
        cajaIds = cv.dnn.NMSBoxes(cajas, confianzas, score_threshold=0.5, nms_threshold=0.2)
        
        for i in cajaIds:
            caja = cajas[np.squeeze(i)]
            confianza_clase = confianzas[np.squeeze(i)]
            confianza_etiquetaID = np.squeeze(clasesEtiquetasID[np.squeeze(i)])
            etiqueta_clase = clases[confianza_etiquetaID]
            claseColor = [int(c) for c in colores[confianza_etiquetaID]]

            texto_imprimir="{} ({:.2f}%)".format(etiqueta_clase.strip(),confianza_clase*100)
            x,y,w,h = caja

            cv.rectangle(imagen, (x,y), (x+w,y+h), color= claseColor, thickness=2)
            cv.putText(imagen, texto_imprimir, (x,y-10), cv.FONT_HERSHEY_PLAIN, 1, claseColor, 2)
        out.write(imagen)
        key = cv.waitKey(1) & 0xFF
        if key== ord("q"):
            break
        (exito, imagen) = video.read()
    video.release()
    out.release()
    cv.destroyAllWindows()

ruta_origen = os.path.join("Datos","Video.mp4")
Video(url=ruta_origen,width=1000, height=600)


ruta = os.path.join("Datos","Video.mp4")
detectar_Y_generar(ruta)
print("Video procesado exitosamente")

video_url = os.path.join("Datos", "output_video.mp4")
video = Video(url=video_url, embed=True, width=1000, height=600)
video

