# Reconocimiento de comandos de voz (LPC + HMM)

Asistente de voz para Linux con activacion por frase y comandos del sistema.

**Pipeline:** audio WAV → preenfasis → ventana Hamming → LPC → `GaussianHMM` → comando Bash (lista blanca).

**Intenciones:** `activacion`, `listar`, `memoria`, `disco`, `red`, `procesos`

---

## Requisitos

- Python 3.10+
- Linux (presentacion) o macOS (desarrollo; algunos comandos Bash tienen equivalente)
- Microfono

---

## Instalacion

```bash
git clone git@github.com:emmanuelmal2/reconocedorComandos.git
cd reconocedorComandos

python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
pip install -r requirements.txt
```

---

## Estructura del proyecto

```
ReconocimientoPatrones/
├── dataset/
│   ├── activacion/
│   ├── listar/
│   ├── memoria/
│   ├── disco/
│   ├── red/
│   └── procesos/
├── models/              # escalador.pkl y *.pkl (generados al entrenar)
├── src/
│   ├── configuracion.py
│   ├── caracteristicas.py
│   ├── grabar.py
│   ├── entrenar.py
│   ├── predecir.py
│   ├── evaluar.py
│   ├── asistente.py
│   ├── consola.py
│   ├── demo_linux.py
│   └── ejecutar.py
├── scripts/
│   └── iniciar_demo_linux.sh
└── requirements.txt
```

### Nombre de audios en el dataset

```
<intencion>_<hablante>_fraseXX_repYY.wav
```

Ejemplos:

```
dataset/memoria/memoria_emmanuel_frase03_rep02.wav
dataset/memoria/memoria_elioth_frase01_rep01.wav
```

Varios hablantes pueden convivir en la misma carpeta; `entrenar.py` usa **todos** los `.wav`.

---

## Instrucciones para Elioth (segundo hablante)

El repo ya incluye los audios de **emmanuel**. Tu parte es grabar las mismas frases con tu voz.

### 1. Clonar e instalar

```bash
git clone git@github.com:emmanuelmal2/reconocedorComandos.git
cd reconocedorComandos
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Grabar tus audios

```bash
python -m src.grabar
```

Cuando pregunte quien graba, elige **2. elioth**.

- Son **6 intenciones** x **5 frases** x **5 repeticiones** = **150 audios** (~25 min).
- Se guardan como `*_elioth_*.wav` junto a los de emmanuel.
- Para grabar solo una intencion:

```bash
python -m src.grabar --hablante elioth --intencion memoria
```

### 3. Entrenar y evaluar

```bash
python -m src.entrenar
python -m src.evaluar
```

### 4. Subir tus cambios a GitHub

```bash
git add dataset/
git commit -m "Agrega audios de elioth al dataset"
git push
```

O, si no usas Git: comprime `dataset/` y enviaselo a emmanuel.

```bash
zip -r dataset_con_elioth.zip dataset/
```

---

## Uso general

### Grabar dataset

```bash
python -m src.grabar                  # pregunta: emmanuel o elioth
python -m src.grabar --hablante elioth --intencion red
python -m src.grabar --hablante emmanuel --forzar   # re-grabar sin preguntar
```

### Entrenar modelos

```bash
python -m src.entrenar
python -m src.entrenar --verbose
```

Genera `models/escalador.pkl` y un HMM por intencion (`activacion.pkl`, `listar.pkl`, ...).

### Evaluar precision

```bash
python -m src.evaluar
```

### Probar un audio

```bash
python -m src.predecir dataset/red/red_emmanuel_frase01_rep01.wav --verbose
python -m src.predecir audio.wav --excluir activacion
```

### Asistente en vivo

```bash
python -m src.asistente
python -m src.asistente --una-vez --verbose
python -m src.asistente --probar-comando dataset/red/red_emmanuel_frase01_rep01.wav
```

**Flujo:**

1. Di frase de activacion (*"oye computadora"*).
2. Si detecta `activacion`, di un comando (*"muestra la memoria"*, etc.).
3. Ejecuta Bash solo para: `listar`, `memoria`, `disco`, `red`, `procesos`.

### Modo demo (presentacion en Linux / VM)

Para la exposicion: proceso **siempre activo** (bucle hasta Ctrl+C), consola con colores y, en Linux, ventanas de terminal que se abren al activar y al ejecutar comandos.

```bash
# Opcion recomendada en la VM (abre gnome-terminal / xfce4-terminal si hay GUI)
./scripts/iniciar_demo_linux.sh

# O directamente en una terminal ya abierta
python -m src.asistente --demo
```

**Que hace `--demo`:**

- Banner de bienvenida y estados visibles (`ESCUCHA` → `ACTIVADO` → `COMANDO` → `EJECUTANDO`).
- Al reconocer *"oye computadora"*: notificacion de escritorio (`notify-send`) y una terminal extra con el mensaje de activacion.
- Al reconocer un comando: el Bash (`free -h`, `ip addr`, etc.) corre en **otra terminal nueva** para que se vea en pantalla.
- En macOS sirve para probar colores; las ventanas extra solo funcionan en Linux.

Requisitos en la VM: emulador de terminal (`gnome-terminal`, `xfce4-terminal`, `konsole` o `xterm`) y microfono accesible desde la VM.

---

## Comandos Bash (Linux)

| Intencion  | Comando              |
|-----------|----------------------|
| listar    | `ls -la`             |
| memoria   | `free -h`            |
| disco     | `df -h`              |
| red       | `ip addr`            |
| procesos  | `ps aux --sort=-pcpu`|

`activacion` no ejecuta nada (solo activa el modo escucha).

En macOS se usan equivalentes (`vm_stat`, `ifconfig`, etc.) para pruebas locales.

---

## Parametros LPC (practica de clase)

| Parametro        | Valor   |
|------------------|---------|
| Muestreo         | 16 kHz mono |
| Preénfasis       | alpha = 0.95 |
| Ventana          | Hamming 512 |
| Salto            | 170 |
| LPC              | orden 12 |
| Salida           | matriz `(n_bloques, 12)` |

## HMM (`hmmlearn`)

| Parametro        | Valor   |
|------------------|---------|
| Modelo           | `GaussianHMM` |
| `n_components`   | 4 |
| `covariance_type`| `diag` |
| Entrenamiento    | `model.fit(X, longitudes)` |
| Prediccion       | `model.score(X, [len(X)])` |

---

## Licencia / materia

Proyecto academico — Reconocimiento de Patrones.
