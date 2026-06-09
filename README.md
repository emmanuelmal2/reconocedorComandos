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

cd ~/reconocedorComandos
git pull
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Los `.pkl` no van en git. **Antes de la demo en vivo hay que calibrar** con el microfono de la VM (ver seccion siguiente).

Frases exactas: **"oye computadora"** → activacion; luego **"listar archivos"**, **"muestra la memoria"** o **"ver procesos"**.

---

## Calibracion en la VM y demo por hablante

Los audios del repo se grabaron en otro equipo. En la VM hay que **re-grabar con el mismo micro** que usaran en la expo. Sin eso, la activacion puede funcionar pero los comandos se confunden.

`calibrar_vm` sobrescribe solo los WAV del hablante indicado (`*_emmanuel_*` o `*_elioth_*`). Los del otro hablante en `dataset/` **no se borran** y siguen valiendo para el informe (evaluacion offline ~97 %).

| Situacion | Calibrar | Demo en vivo |
|-----------|----------|--------------|
| Solo Emmanuel presenta | `python3 -m src.calibrar_vm --hablante emmanuel` | `python3 -m src.asistente --demo --hablante emmanuel` |
| Solo Elioth presenta (cuando pueda) | `python3 -m src.calibrar_vm --hablante elioth` | `python3 -m src.asistente --demo --hablante elioth` |
| Los dos presentan el mismo dia | `python3 -m src.calibrar_vm --todos` | `python3 -m src.asistente --demo` (ambas voces) |

**Importante:** `calibrar_vm --hablante X` entrena con `entrenar --hablante X` y **solo deja modelos `.pkl` de ese hablante** en `models/`. Si Elioth calibra despues de Emmanuel, Emmanuel debe volver a calibrar (o usar `--todos` cuando los dos puedan grabar).

### Emmanuel (expo actual)

```bash
python3 -m src.calibrar_vm --hablante emmanuel

# Probar micro antes de la demo
python3 -m src.probar_microfono --activacion --hablante emmanuel
python3 -m src.probar_microfono --comandos --hablante emmanuel

# Demo
python3 -m src.asistente --demo --hablante emmanuel
# o: ./scripts/iniciar_demo_linux.sh
```

### Elioth (cuando pueda grabar en la VM)

Sus audios **ya estan en GitHub** para el dataset academico. Para hablar **en vivo** en la VM necesita calibrar una vez (mismo micro, ~5 min):

```bash
python3 -m src.calibrar_vm --hablante elioth

python3 -m src.probar_microfono --activacion --hablante elioth
python3 -m src.probar_microfono --comandos --hablante elioth

python3 -m src.asistente --demo --hablante elioth
```

`--hablante elioth` usa **solo** sus modelos (`activacion_elioth.pkl`, etc.); no mezcla con Emmanuel en la prediccion.

### Los dos hablantes

```bash
python3 -m src.calibrar_vm --todos
python3 -m src.asistente --demo
```

### Flujo de la demo (UX)

1. `[Enter]` → graba 3 s → di **"oye computadora"**
2. Si activa → **graba el comando al instante** (sin otro Enter la primera vez)
3. Di el comando; si falla, pide Enter para reintentar

Arranque rapido (sin matriz al inicio):

```bash
python3 -m src.asistente --demo --hablante emmanuel --sin-evaluacion
```

**No uses** `--hablante elioth` en vivo si Elioth no calibro en esa VM: los modelos del repo son de otro microfono y clasificaran mal.

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
│   ├── calibrar_vm.py
│   ├── probar_microfono.py
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

## Uso general

### Grabar dataset

```bash
python -m src.grabar                  # pregunta: emmanuel o elioth
python -m src.grabar --hablante elioth --intencion memoria
python -m src.grabar --hablante emmanuel --forzar   # re-grabar sin preguntar
```

### Entrenar modelos

```bash
python3 -m src.entrenar                    # todos los hablantes (informe / evaluacion)
python3 -m src.entrenar --hablante emmanuel   # solo un hablante (tras calibrar_vm)
python3 -m src.entrenar --verbose
```

Genera `models/escalador.pkl` y un HMM por intencion **y hablante**
(`memoria_emmanuel.pkl`, `memoria_elioth.pkl`, ...). Sin `--hablante`, en prediccion se usa el
mejor score entre hablantes (max-pooling). En la demo en vivo conviene `--hablante` tras calibrar en la VM.

### Calibrar microfono (VM)

```bash
python3 -m src.calibrar_vm --hablante emmanuel
python3 -m src.calibrar_vm --hablante elioth
python3 -m src.calibrar_vm --todos
```

### Probar microfono (sin bucle del asistente)

```bash
python3 -m src.probar_microfono --activacion --hablante emmanuel   # "oye computadora"
python3 -m src.probar_microfono --comandos --hablante emmanuel       # listar / memoria / procesos
```

No uses `--comandos` para probar activacion: ese modo excluye "oye computadora" a proposito.

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

Consola con colores, matriz de confusion al inicio (opcional) y terminales extra en Linux.

```bash
./scripts/iniciar_demo_linux.sh
# equivalente a: python3 -m src.asistente --demo --hablante emmanuel

python3 -m src.asistente --demo --hablante emmanuel
python3 -m src.asistente --demo --hablante elioth      # tras calibrar_vm --hablante elioth
python3 -m src.asistente --demo                        # tras calibrar_vm --todos
python3 -m src.asistente --demo --sin-evaluacion       # sin matriz al arrancar
```

**Que hace `--demo`:**

- Evaluacion offline y matriz matplotlib al inicio (salvo `--sin-evaluacion`).
- Estados visibles: `ESCUCHA` → `ACTIVADO` → `COMANDO` → `EJECUTANDO`.
- En Linux: `notify-send` al activar y Bash en terminal nueva al ejecutar comando.

Requisitos VM: `python3-tk`, emulador de terminal, microfono.

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
