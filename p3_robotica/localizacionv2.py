#! /usr/bin/env python3

# Robótica Computacional
# Grado en Ingeniería Informática (Cuarto)
# Práctica 5: Localización de robots móviles.

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# ******************************************************************************
# Declaración de funciones

def distancia(a, b):
    # Distancia entre dos puntos (admite poses)
    return np.linalg.norm(np.subtract(a[:2], b[:2]))

def angulo_rel(pose, p):
    # Diferencia angular entre una pose y un punto objetivo 'p'
    w = atan2(p[1]-pose[1], p[0]-pose[0]) - pose[2]
    while w >  pi: w -= 2*pi
    while w < -pi: w += 2*pi
    return w

def mostrar(objetivos, ideal, trayectoria):
    # Mostrar objetivos y trayectoria
    plt.figure('Trayectoria')
    plt.clf()
    plt.ion() # modo interactivo
    objT   = np.array(objetivos).T.tolist()
    trayT  = np.array(trayectoria).T.tolist()
    ideT   = np.array(ideal).T.tolist()
    
    # Ajustar bordes
    bordes = [min(trayT[0]+objT[0]+ideT[0]), max(trayT[0]+objT[0]+ideT[0]),
              min(trayT[1]+objT[1]+ideT[1]), max(trayT[1]+objT[1]+ideT[1])]
    centro = [(bordes[0]+bordes[1])/2., (bordes[2]+bordes[3])/2.]
    radio  = max(bordes[1]-bordes[0], bordes[3]-bordes[2]) * 0.75
    plt.xlim(centro[0]-radio, centro[0]+radio)
    plt.ylim(centro[1]-radio, centro[1]+radio)

    # Representar ideal (verde) y real (rojo)
    plt.plot(ideT[0], ideT[1], '-g', label='Ideal')
    plt.plot(trayectoria[0][0], trayectoria[0][1], 'or')
    r = radio * 0.1
    for p in trayectoria:
        plt.plot([p[0], p[0]+r*cos(p[2])], [p[1], p[1]+r*sin(p[2])], '-r')
    
    plt.plot(objT[0], objT[1], '-.o', label='Objetivos')
    plt.draw()
    plt.pause(0.001)

def localizacion(balizas, real, ideal, centro, radio, step=0.1):
    """
    Busca la localización más probable del robot usando sensores de distancia y ángulo.
    """
    # 1. CORRECCIÓN DE ORIENTACIÓN (Requisito: usar senseAngle)
    orientacion_estimada = real.senseAngle(balizas)
    
    # 2. CORRECCIÓN DE POSICIÓN (Grid Search)
    medida_real = real.senseDistance(balizas)
    min_error = float('inf')
    best_x, best_y = ideal.x, ideal.y

    # Búsqueda en rejilla (Grid Search)
    for y in np.arange(centro[1]-radio, centro[1]+radio + step, step):
        for x in np.arange(centro[0]-radio, centro[0]+radio + step, step):
            d_virtual = [np.linalg.norm(np.subtract([x, y], b)) for b in balizas]
            error = np.linalg.norm(np.subtract(medida_real, d_virtual))
            
            if error < min_error:
                min_error = error
                best_x, best_y = x, y
    
    # Actualizar el robot ideal con la pose estimada
    ideal.set(best_x, best_y, orientacion_estimada)

# ******************************************************************************

# Definición del robot y constantes
P_INICIAL = [0., 4., 0.]        # Pose inicial real
P_INICIAL_IDEAL = [2., 2., 0.]  # Pose inicial ideal
V_LINEAL  = .7                  # m/s
V_ANGULAR = 140.                # º/s
FPS       = 10.                 # Hz
MOSTRAR   = True                # Mostrar gráfica

HOLONOMICO = 1
GIROPARADO = 0
LONGITUD   = .2

trayectorias = [
    [[1,3]],
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)]
]

if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
    sys.exit(f"{sys.argv[0]} <indice entre 0 y {len(trayectorias)-1}>")
objetivos = trayectorias[int(sys.argv[1])]

balizas = [[0,0], [0,4], [4,0], [4,4]]
EPSILON = 0.15 
V = V_LINEAL/FPS
W = V_ANGULAR*pi/(180*FPS)

ideal = robot()
ideal.set_noise(0,0,0)
ideal.set(*P_INICIAL_IDEAL)

real = robot()
real.set_noise(.01, .01, .1) 
real.set(*P_INICIAL)

random.seed(0)
tray_real = [real.pose()]
tiempo  = 0.
espacio = 0.
tic = time.time()

# --- LOCALIZACIÓN INICIAL (Búsqueda global) ---
localizacion(balizas, real, ideal, centro=[2, 2], radio=3.0, step=0.2)
tray_ideal = [ideal.pose()]

distanciaObjetivos = []

for punto in objetivos:
    while distancia(tray_ideal[-1], punto) > EPSILON and len(tray_ideal) <= 1000:
        pose_filt = ideal.pose()

        w = angulo_rel(pose_filt, punto)
        if w > W:  w =  W
        if w < -W: w = -W
        v = distancia(pose_filt, punto)
        if v > V: v = V

        if HOLONOMICO:
            if GIROPARADO and abs(w) > 0.1: v = 0
            ideal.move(w, v)
            real.move(w, v)
        else:
            ideal.move_triciclo(w, v, LONGITUD)
            real.move_triciclo(w, v, LONGITUD)
        
        tray_real.append(real.pose())

        # --- RE-LOCALIZACIÓN EN TIEMPO REAL (Búsqueda local optimizada) ---
        localizacion(balizas, real, ideal, centro=ideal.pose(), radio=0.4, step=0.05)
        tray_ideal.append(ideal.pose())

        if MOSTRAR and int(tiempo) % 2 == 0:
            mostrar(objetivos, tray_ideal, tray_real)

        espacio += v
        tiempo  += 1

    distanciaObjetivos.append(distancia(tray_real[-1], punto))

toc = time.time()

# Métricas
desviacion = np.sum([distancia(tray_real[i], tray_ideal[i]) for i in range(len(tray_real))])
tiempo_total = toc - tic

print(f"Recorrido: {espacio:.3f}m / {tiempo/FPS}s")
print(f"Distancia real al objetivo final: {distanciaObjetivos[-1]:.3f}m")
print(f"Suma de distancias a objetivos: {np.sum(distanciaObjetivos):.3f}m")
print(f"Tiempo real invertido: {tiempo_total:.3f}sg")
print(f"Desviacion de las trayectorias: {desviacion:.3f}")

# LINEA DE RESUMEN FINAL
print(f"Resumen: {tiempo_total:.3f} {desviacion:.3f} {np.sum(distanciaObjetivos):.3f}")

if MOSTRAR:
    mostrar(objetivos, tray_ideal, tray_real)
    input("Pulsa Enter para terminar...")