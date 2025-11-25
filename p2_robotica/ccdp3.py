#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - 
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).
# Alejandro Rodríguez Rojas - alu0101317038@ull.edu.es

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O,final=0):
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final != 0:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O_hist,obj):
  plt.figure()
  plt.xlim(-L,L)
  plt.ylim(-L,L)
  T = [np.array(o).T.tolist() for o in O_hist]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i/float(len(T)),1,1))
  plt.plot(obj[0], obj[1], '*')
  plt.pause(0.0001)
  plt.show()
  plt.close()

def matriz_T(d,th,a,al):
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)],
          [sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)],
          [0,        sin(al),         cos(al),          d],
          [0,        0,               0,                1        ]]

def cin_dir(th,a):
  T = np.identity(4)
  o = [[0.0,0.0]]
  for i in range(len(th)):
    T = np.dot(T, matriz_T(0, th[i], a[i], 0))
    tmp = np.dot(T, [0,0,0,1])
    o.append([tmp[0], tmp[1]])
  return o

def clamp(x, xmin, xmax):
  return max(xmin, min(x, xmax))

# ******************************************************************************
# Configuración del robot

NUM_JOINTS = 3 # Número de articulaciones

th  = [0.0, 0.0, 0.0] # Ángulos iniciales en radianes
a   = [5.0, 5.0, 5.0] # Longitudes de los eslabones
joint_types  = ['R', 'R', 'R'] # Tipo de articulaciones: 'R' para rotacional, 'P' para prismática

# -------------------------------
# AÑADIDO: LÍMITES EN GRADOS → RADIANES
# -------------------------------

limite_sup_deg = [45, 90, 90] # Límites superiores en grados
limite_inf_deg = [-45, -90, -90] # Límites inferiores en grados

joint_limits = [
    (radians(limite_inf_deg[i]), radians(limite_sup_deg[i]))
    for i in range(NUM_JOINTS)
]

# -------------------------------

assert len(th) == len(a) == len(joint_types) == len(joint_limits)

L = sum(a)
EPSILON = 0.01

# ******************************************************************************
# Introducción del objetivo

if len(sys.argv) != 3:
  sys.exit("python " + sys.argv[0] + " x y")

objetivo = [float(i) for i in sys.argv[1:]]

O_actual = cin_dir(th, a)
print("- Posicion inicial:")
muestra_origenes(O_actual)

dist = np.linalg.norm(np.subtract(objetivo, O_actual[-1]))
prev = dist + 2*EPSILON
iteracion = 1

# ******************************************************************************
# Bucle principal CCD

while (dist > EPSILON and abs(prev - dist) > EPSILON/100.0):

  prev = dist
  O_hist = []
  O_actual = cin_dir(th, a)
  O_hist.append(O_actual)

  for i in reversed(range(len(th))):

    O_actual = cin_dir(th, a)
    O_hist.append(O_actual)

    p_joint = np.array(O_actual[i])
    p_ee    = np.array(O_actual[-1])
    v_ee    = p_ee - p_joint
    v_obj   = np.array(objetivo) - p_joint

    if joint_types[i] == 'R':
      ang_ee  = atan2(v_ee[1], v_ee[0])
      ang_obj = atan2(v_obj[1], v_obj[0])
      delta_th = (ang_obj - ang_ee + pi) % (2*pi) - pi

      th[i] += delta_th
      th[i] = clamp(th[i], joint_limits[i][0], joint_limits[i][1])

    elif joint_types[i] == 'P':
      # (No usas prismáticas, esto se queda igual)
      link_vec = np.array(O_actual[i+1]) - p_joint
      norm_link = np.linalg.norm(link_vec)
      if norm_link < 1e-6:
        dir_unit = v_obj / np.linalg.norm(v_obj)
      else:
        dir_unit = link_vec / norm_link
      desired_len = np.dot(v_obj, dir_unit)
      a[i] = clamp(desired_len, joint_limits[i][0], joint_limits[i][1])

  O_actual = cin_dir(th, a)
  O_hist.append(O_actual)

  dist = np.linalg.norm(np.subtract(objetivo, O_actual[-1]))
  print("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O_actual)
  muestra_robot(O_hist, objetivo)
  print("Distancia al objetivo = " + str(round(dist,5)))
  iteracion += 1

# ******************************************************************************
# Resultados finales

if dist <= EPSILON:
  print("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")

print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist,5)))
print("- Valores finales de las articulaciones:")

for i in range(len(th)):
  if joint_types[i] == 'R':
    print("  theta" + str(i+1) + " = " + str(round(th[i],3)) + " rad")
  else:
    print("  d" + str(i+1) + "     = " + str(round(a[i],3)) + " (prismática)")

for i in range(len(a)):
  print("  L" + str(i+1) + "     = " + str(round(a[i],3)))
