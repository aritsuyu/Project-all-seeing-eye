import json
import numpy as np
from sklearn.linear_model import LinearRegression

# --------- Função para converter HH:MM em minutos ---------
def hora_para_minutos(hora_str):
    h, m = map(int, hora_str.split(":"))
    return h * 60 + m

def minutos_para_hora(minutos):
    h = int(minutos // 60)
    m = int(minutos % 60)
    return f"{h:02d}:{m:02d}"

# --------- Ler o JSON ---------
with open("database.json", "r") as f:
    data = json.load(f)

# Pegar horários do aluno
horarios_str = list(data["Alumm"].values())  
horarios = [hora_para_minutos(h) for h in horarios_str]

# --------- Preparar dados ---------
X = np.arange(len(horarios)).reshape(-1, 1)  # Dias
y = np.array(horarios)

# --------- Treinar modelo ---------
modelo = LinearRegression()
modelo.fit(X, y)

# --------- Prever próximo horário ---------
proximo_dia = np.array([[len(horarios)]])
prev = modelo.predict(proximo_dia)[0]
horario_previsto = minutos_para_hora(prev)

print(f"📌 Horários históricos: {horarios_str}")
print(f"⏩ Próxima chegada prevista: {horario_previsto}")

# --------- Verificação (simulando chegada real) ---------
chegada_real = "07:35"  # tu pode trocar ou puxar de algum input
chegada_minutos = hora_para_minutos(chegada_real)

# Checagem 1: diferença do previsto
if chegada_real != horario_previsto:
    print("⚠️ Alerta: Chegada diferente do horário previsto!")

# Checagem 2: atraso (máximo 7:30)
horario_maximo = hora_para_minutos("07:30")
if chegada_minutos > horario_maximo:
    print("⏰ Atrasado! Chegou depois das 07:30.")
else:
    print("✅ Dentro do horário limite.")
