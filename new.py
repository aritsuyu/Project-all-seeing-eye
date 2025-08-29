import json
import pandas as pd
from prophet import Prophet

# -------- Funções auxiliares --------
def hora_para_minutos(hora_str):
    h, m = map(int, hora_str.split(":"))
    return h * 60 + m

def minutos_para_hora(minutos):
    h = int(minutos // 60)
    m = int(minutos % 60)
    return f"{h:02d}:{m:02d}"

# -------- Ler JSON --------
with open("database.json", "r") as f:
    data = json.load(f)

# -------- Extrair todos os horários do mês --------
horarios_str = []
for day, weeks in data["Alumm"].items():        # Monday, Tuesday...
    for week, hora in weeks.items():           # week1, week2...
        horarios_str.append(hora)

horarios = [hora_para_minutos(h) for h in horarios_str]

# -------- Preparar dados para o Prophet --------
# Datas fictícias só pra representar os dias
datas = pd.date_range(start="2025-08-01", periods=len(horarios), freq="D")
df = pd.DataFrame({"ds": datas, "y": horarios})

# -------- Treinar o modelo --------
modelo = Prophet(daily_seasonality=True)
modelo.fit(df)

# -------- Previsão do próximo dia --------
futuro = pd.DataFrame({"ds": [datas[-1] + pd.Timedelta(days=1)]})
prev = modelo.predict(futuro)

minutos_prev = prev["yhat"].values[0]
horario_previsto = minutos_para_hora(minutos_prev)

# -------- Mostrar resultados --------
print(f" Histórico ({len(horarios)} dias): {horarios_str}")
print(f" Previsão de próxima chegada: {horario_previsto}")
