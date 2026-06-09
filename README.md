# Reconocimiento de comandos de voz (LPC + HMM)

Asistente de voz para Linux con activacion por frase y comandos del sistema.

**Pipeline:** audio WAV → preenfasis → ventana Hamming → LPC → `GaussianHMM` → comando Bash (lista blanca).

**Intenciones:** `activacion`, `listar`, `memoria`, `procesos` (3 comandos + activacion)

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

### Puesta en marcha en VM Ubuntu (presentacion)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip portaudio19-dev python3-tk \
  gnome-terminal libnotify-bin

cd ~/reconocedorComandos   # o donde clonaste el repo
git pull

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Los .pkl no van en git: hay que entrenar una vez en la VM
python3 -m src.entrenar

# Calibracion en la VM: solo Emmanuel re-graba (~5 min, obligatorio para demo en vivo)
python3 -m src.calibrar_vm --hablante emmanuel
# ^ entrena con: python3 -m src.entrenar --hablante emmanuel (solo tu voz + este micro)

# Probar un comando suelto antes de la demo:
python3 -m src.probar_microfono --hablante emmanuel --comandos

# Demo en vivo
python3 -m src.asistente --demo --hablante emmanuel
```

Frases exactas en la demo: **"oye computadora"** → activacion; luego **"listar archivos"**, **"muestra la memoria"** o **"ver procesos"**.

---

## Estructura del proyecto

```
ReconocimientoPatrones/
├── dataset/             # 40 WAV (4 intenciones x 1 frase x 5 reps x 2 hablantes)
│   ├── activacion/      # frase01: oye computadora
│   ├── listar/          # frase01: listar archivos
│   ├── memoria/         # frase01: muestra la memoria
│   └── procesos/        # frase02: ver procesos
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
│   ├── regrabar_fallos.py
│   └── ejecutar.py
├── scripts/
│   ├── iniciar_demo_linux.sh
│   ├── instalar_acceso_directo_linux.sh
│   ├── instalar_servicio_linux.sh
│   ├── reconocedor-comandos.desktop
│   └── reconocedor-comandos.service
└── requirements.txt
```

### Nombre de audios en el dataset

```
<intencion>_<hablante>_fraseXX_repYY.wav
```

Solo se usan **40 audios** (una frase activa por intencion, 5 repeticiones, 2 hablantes):

| Carpeta | Frase en el nombre | Texto |
|---------|-------------------|--------|
| `activacion/` | `frase01` | oye computadora |
| `listar/` | `frase01` | listar archivos |
| `memoria/` | `frase01` | muestra la memoria |
| `procesos/` | `frase02` | ver procesos |

Ejemplo: `dataset/memoria/memoria_elioth_frase01_rep03.wav`

---

## Elioth y la demo en vivo

Los **40 audios de Elioth siguen en el repo** (`*_elioth_*.wav`). Sirven para:

- entrenar el escalador y los HMM (evaluacion offline ~97 %)
- el reporte academico / matriz de confusion

**Elioth no necesita volver a grabar.** Si no puede estar en la VM, no pasa nada.

En la **exposicion en vivo** solo habla **Emmanuel** con el micro de la VM:

```bash
python3 -m src.calibrar_vm --hablante emmanuel   # solo sobrescribe *_emmanuel_*.wav
python3 -m src.asistente --demo --hablante emmanuel
```

Importante: sin `--hablante emmanuel`, el sistema mezcla modelos de Elioth (grabados en Mac) y puede clasificar mal en el micro de la VM.

Si algun dia **los dos** pueden grabar en la misma VM: `python3 -m src.calibrar_vm --todos` y demo sin `--hablante`.

---

## Instrucciones para Elioth (dataset en GitHub)

Sus audios **ya estan en el repo**; no hace falta grabar de nuevo salvo que quieran mejorar una toma desde Mac:

```bash
python -m src.grabar --hablante elioth --intencion memoria --frase 1 --repeticion 1 --forzar
python -m src.entrenar
```

---

## Uso general

### Grabar dataset

```bash
python -m src.grabar                  # pregunta: emmanuel o elioth
python -m src.grabar --hablante elioth --intencion memoria
python -m src.grabar --hablante emmanuel --forzar   # re-grabar sin preguntar
```

### Entrenar modelos

```bash
python -m src.entrenar
python -m src.entrenar --verbose
```

Genera `models/escalador.pkl` y un HMM por intencion **y hablante**
(`memoria_emmanuel.pkl`, `memoria_elioth.pkl`, ...). En prediccion se usa el
mejor score entre hablantes por intencion, asi funciona con ambas voces.

### Evaluar precision

```bash
python -m src.evaluar
python -m src.evaluar --grafica

# Metrica mas honesta: entrena en 80% y evalua en 20% (holdout)
python -m src.evaluar --holdout
python -m src.evaluar --holdout --grafica

# Solo listar audios mal clasificados
python -m src.evaluar --solo-fallos
```

La evaluacion normal usa los modelos ya guardados sobre **todo** el dataset (optimista).
`--holdout` entrena modelos temporales en memoria sin pisar `models/*.pkl`.

### Re-grabar audios flojos (memoria, listar, etc.)

Detecta WAV mal clasificados y permite re-grabarlos uno por uno:

```bash
python -m src.regrabar_fallos --hablante emmanuel
python -m src.regrabar_fallos --hablante emmanuel --intenciones memoria listar --forzar
```

Re-grabar una toma concreta:

```bash
python -m src.grabar --hablante emmanuel --intencion memoria --frase 1 --repeticion 2
```

Despues de re-grabar: `python -m src.entrenar`

### Probar un audio

```bash
python -m src.predecir dataset/memoria/memoria_emmanuel_frase01_rep01.wav --verbose
python -m src.predecir audio.wav --excluir activacion
```

### Asistente en vivo

```bash
python -m src.asistente
python -m src.asistente --una-vez --verbose
python -m src.asistente --probar-comando dataset/listar/listar_emmanuel_frase01_rep01.wav
```

**Flujo:**

1. Di frase de activacion (*"oye computadora"*).
2. Si detecta `activacion`, di un comando (*"muestra la memoria"*, etc.).
3. Ejecuta Bash solo para: `listar`, `memoria`, `procesos`.

### Modo demo (presentacion en Linux / VM)

Para la exposicion: proceso **siempre activo** (bucle hasta Ctrl+C), consola con colores y, en Linux, ventanas de terminal que se abren al activar y al ejecutar comandos.

```bash
# Opcion recomendada en la VM (abre gnome-terminal / xfce4-terminal si hay GUI)
./scripts/iniciar_demo_linux.sh

# O directamente en una terminal ya abierta
python -m src.asistente --demo
```

**Que hace `--demo`:**

- Al iniciar: evaluacion offline con **precision por intencion** y **matriz de confusion en matplotlib** (ventana grafica + PNG en `models/matriz_confusion.png`).
- Banner de bienvenida y estados visibles (`ESCUCHA` → `ACTIVADO` → `COMANDO` → `EJECUTANDO`).
- Al reconocer *"oye computadora"*: notificacion de escritorio (`notify-send`) y una terminal extra con el mensaje de activacion.
- Al reconocer un comando: el Bash (`ls -la`, `free -h`, `ps aux`, etc.) corre en **otra terminal nueva** para que se vea en pantalla.
- En macOS sirve para probar colores; las ventanas extra solo funcionan en Linux.

Para saltar la evaluacion inicial (arranque mas rapido):

```bash
python -m src.asistente --demo --sin-evaluacion
```

Requisitos en la VM: emulador de terminal (`gnome-terminal`, `xfce4-terminal`, `konsole` o `xterm`), microfono accesible y, para la grafica, `sudo apt install -y python3-tk` (backend de ventanas de matplotlib).

### Acceso directo en el menu de Ubuntu

```bash
./scripts/instalar_acceso_directo_linux.sh
```

Crea un icono **Asistente de voz LPC+HMM** en el menu de aplicaciones.

### Servicio systemd (asistente siempre activo)

```bash
./scripts/instalar_servicio_linux.sh
systemctl --user start reconocedor-comandos
```

El servicio corre `asistente --demo --sin-evaluacion` en segundo plano (sin bloquear en la grafica inicial).
Ver logs: `journalctl --user -u reconocedor-comandos -f`

---

## Comandos Bash (Linux)

| Intencion  | Comando              | Frase demo sugerida        |
|-----------|----------------------|----------------------------|
| listar    | `ls -la`             | "listar archivos"          |
| memoria   | `free -h`            | "muestra la memoria"       |
| procesos  | `ps aux --sort=-pcpu`| "ver procesos"             |

Activacion: **"oye computadora"** (frase unica por intencion; 5 repeticiones en el dataset).

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
