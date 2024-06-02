import pytesseract
import cv2
import numpy as np
import re
from datetime import datetime
import asyncio
import winsdk.windows.devices.geolocation as wdg
import firebase_admin
from firebase_admin import credentials, firestore


async def getCoords():
  locator = wdg.Geolocator()
  pos = await locator.get_geoposition_async()
  return [pos.coordinate.latitude, pos.coordinate.longitude]

def getLoc():
  try:
    return asyncio.run(getCoords())
  except PermissionError:
    print("ERROR: You need to allow applications to access you location in Windows settings")

cred = credentials.Certificate(r"C:\Users\caminho\chave.json")
app = firebase_admin.initialize_app(cred)
db = firestore.client()

placa_anterior = None
veiculos_roubados = ["lista-placas-veiculos-roubados"]

while True:
  video = cv2.VideoCapture()
  url = "https://172.20.10.7:8080/video" #IP fornecido pelo app IPWebcam
  video.open(url)
  check, imagem = video.read()
  imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

  kernel_retangular = cv2.getStructuringElement(cv2.MORPH_RECT, (40,13))
  chapeu_preto = cv2.morphologyEx(imagem, cv2.MORPH_BLACKHAT, kernel_retangular)
  sobel_x = cv2.Sobel(chapeu_preto, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = 1)
  sobel_x = np.absolute(sobel_x)
  sobel_x = sobel_x.astype('uint8')
  sobel_x = cv2.GaussianBlur(sobel_x, (5,5), 0)
  sobel_x = cv2.morphologyEx(sobel_x, cv2.MORPH_CLOSE, kernel_retangular)
  valor, limiarizacao = cv2.threshold(sobel_x, 0 , 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  kernel_quadrado = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  limiarizacao = cv2.erode(limiarizacao, kernel_quadrado, iterations = 2)
  limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations = 2)
  fechamento = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel_quadrado)
  valor, mascara = cv2.threshold(fechamento, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  limiarizacao = cv2.bitwise_and(limiarizacao, limiarizacao, mask = mascara)
  limiarizacao = cv2.dilate(limiarizacao, kernel_quadrado, iterations = 2)
  limiarizacao = cv2.erode(limiarizacao, kernel_quadrado)

  contornos, hierarquia = cv2.findContours(limiarizacao, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contornos = sorted(contornos, key = cv2.contourArea, reverse = True)[:10]
  for contorno in contornos:
    regiao_interesse= np.array([])
    x, y, w, h = cv2.boundingRect(contorno)
    proporcao = float(w)/h
    if proporcao >= 2 and proporcao <= 4:
      placa = imagem[y:y+h, x:x+w]
      valor, regiao_interesse = cv2.threshold(placa, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
      break

  if not regiao_interesse.any():
    cv2.waitKey(1)
    continue

  pytesseract.pytesseract.tesseract_cmd = r'C:\Users\caminho\Tesseract-OCR\tesseract.exe'
  config_tesseract = '--tessdata-dir tessdata --psm 6'
  texto = pytesseract.image_to_string(regiao_interesse, lang= 'por', config = config_tesseract)
  texto_regex = re.search('\w{3}\d{1}\w{1}\d{2}', texto)
  if texto_regex and texto_regex.group(0) != placa_anterior:
    placa_reconhecida = texto_regex.group(0)
    placa_anterior = placa_reconhecida
    if placa_reconhecida in veiculos_roubados:
      data_hora = datetime.now()
      dados = {"placa": placa_reconhecida,
               "data_hora": data_hora,
                "latitude": getLoc()[0],
                "longitude": getLoc()[1]}
      print("Veículo roubado encontrado!!! Placa:", placa_reconhecida)
      print(dados)
      doc_ref = db.collection("veiculos").document()
      doc_ref.set(dados)
    else:
      print("Placa reconhecida:", placa_reconhecida)
  cv2.waitKey(1)
