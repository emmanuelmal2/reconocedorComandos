"""
Grabacion del dataset de entrenamiento con sounddevice y soundfile.

Formato de archivo:
    dataset/<intencion>/<intencion>_<hablante>_fraseXX_repYY.wav

Ejemplo:
    dataset/memoria/memoria_emmanuel_frase03_rep02.wav
    dataset/memoria/memoria_elioth_frase01_rep01.wav

Al iniciar pregunta quien graba: emmanuel o elioth.
Si emmanuel ya tiene audios completos, ofrece volver a grabar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.configuracion import (
    CARPETA_DATASET,
    DURACION_GRABACION,
    FRASES,
    FRASES_POR_INTENCION,
    FRECUENCIA_MUESTREO,
    HABLANTES,
    INTENCIONES,
    REPETICIONES_POR_FRASE,
)

# Sugerencia de variacion por numero de repeticion
SUGERENCIAS_VARIACION: dict[int, str] = {
    1: "normal",
    2: "un poco mas rapido",
    3: "un poco mas lento",
    4: "ligeramente alargado",
    5: "natural para prueba",
}


def _nombre_archivo(
    intencion: str,
    hablante: str,
    numero_frase: int,
    repeticion: int,
) -> str:
    """Genera nombre: <intencion>_<hablante>_fraseXX_repYY.wav"""
    return f"{intencion}_{hablante}_frase{numero_frase:02d}_rep{repeticion:02d}.wav"


def _audios_esperados_por_hablante(intenciones: list[str]) -> int:
    return len(intenciones) * FRASES_POR_INTENCION * REPETICIONES_POR_FRASE


def _listar_audios_hablante(
    hablante: str,
    intenciones: list[str],
    carpeta_dataset: Path = CARPETA_DATASET,
) -> list[Path]:
    """Lista audios existentes de un hablante en las intenciones indicadas."""
    archivos: list[Path] = []
    for intencion in intenciones:
        carpeta = carpeta_dataset / intencion
        if not carpeta.is_dir():
            continue
        archivos.extend(sorted(carpeta.glob(f"{intencion}_{hablante}_frase*.wav")))
    return archivos


def _hablante_tiene_audios(
    hablante: str,
    intenciones: list[str],
    carpeta_dataset: Path = CARPETA_DATASET,
) -> bool:
    """True si el hablante ya tiene al menos un audio en el alcance."""
    return len(_listar_audios_hablante(hablante, intenciones, carpeta_dataset)) > 0


def _hablante_dataset_completo(
    hablante: str,
    intenciones: list[str],
    carpeta_dataset: Path = CARPETA_DATASET,
) -> bool:
    """True si el hablante tiene todos los audios esperados en el alcance."""
    existentes = len(_listar_audios_hablante(hablante, intenciones, carpeta_dataset))
    esperados = _audios_esperados_por_hablante(intenciones)
    return existentes >= esperados


def seleccionar_hablante() -> str:
    """Pregunta al inicio quien va a grabar."""
    print("\n" + "=" * 60)
    print("  GRABACION DEL DATASET DE VOZ")
    print("=" * 60)
    print("  ¿Quien va a grabar?")
    print("    1. emmanuel")
    print("    2. elioth")

    while True:
        opcion = input("  Elige 1 o 2: ").strip().lower()
        if opcion in ("1", "emmanuel"):
            return "emmanuel"
        if opcion in ("2", "elioth"):
            return "elioth"
        print("  Opcion invalida. Escribe 1 o 2.")


def _confirmar_si_no(mensaje: str) -> bool:
    """True si el usuario responde afirmativo."""
    while True:
        respuesta = input(f"  {mensaje} (s/n): ").strip().lower()
        if respuesta in ("s", "si", "sí", "y", "yes"):
            return True
        if respuesta in ("n", "no"):
            return False
        print("  Responde s o n.")


def verificar_inicio_grabacion(
    hablante: str,
    intenciones: list[str],
    carpeta_dataset: Path = CARPETA_DATASET,
) -> bool:
    """
    Decide si continuar con la grabacion.

    - elioth: graba normal (sin mensaje de audios listos).
    - emmanuel sin audios: graba normal.
    - emmanuel con audios: avisa y pregunta si re-grabar.
    """
    existentes = _listar_audios_hablante(hablante, intenciones, carpeta_dataset)
    esperados = _audios_esperados_por_hablante(intenciones)

    if hablante != "emmanuel":
        print(f"\n  Hablante: {hablante}")
        if existentes:
            print(f"  Audios de {hablante} encontrados: {len(existentes)}/{esperados}")
        else:
            print(f"  Sin audios previos de {hablante}. Comenzando grabacion.")
        return True

    if not existentes:
        print("\n  Hablante: emmanuel")
        print("  Aun no hay audios de emmanuel en este alcance. Comenzando grabacion.")
        return True

    print(f"\n  Hablante: emmanuel")
    print(f"  Audios encontrados: {len(existentes)}/{esperados}")

    if _hablante_dataset_completo(hablante, intenciones, carpeta_dataset):
        print("\n  Ya estan los audios de emmanuel listos en este alcance.")
    else:
        print("\n  Hay audios de emmanuel, pero el dataset aun no esta completo.")

    return _confirmar_si_no("Quieres volver a grabar?")


def grabar_audio(
    duracion_segundos: float = DURACION_GRABACION,
    frecuencia: int = FRECUENCIA_MUESTREO,
) -> np.ndarray:
    """Graba audio mono desde el microfono."""
    n_muestras = int(duracion_segundos * frecuencia)
    grabacion = sd.rec(
        n_muestras,
        samplerate=frecuencia,
        channels=1,
        dtype="float64",
    )
    sd.wait()
    return grabacion.reshape(-1)


def preparar_audio_en_vivo(
    audio: np.ndarray,
    frecuencia: int = FRECUENCIA_MUESTREO,
    umbral_relativo: float = 0.08,
    ventana_ms: int = 25,
    margen_ms: int = 80,
) -> np.ndarray:
    """
    Recorta silencios al inicio/fin y normaliza amplitud.

    Acerca la grabacion en vivo al perfil de los WAV del dataset (voz centrada).
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(audio) == 0:
        return audio

    pico = float(np.max(np.abs(audio)))
    if pico > 1e-8:
        audio = audio * (0.95 / pico)

    ventana = max(1, int(frecuencia * ventana_ms / 1000))
    margen = int(frecuencia * margen_ms / 1000)
    n_frames = max(1, len(audio) // ventana)
    energias = np.array(
        [float(np.mean(audio[i * ventana : (i + 1) * ventana] ** 2)) for i in range(n_frames)]
    )
    energia_max = float(np.max(energias))
    if energia_max <= 1e-14:
        return audio

    umbral = umbral_relativo * energia_max
    activos = np.where(energias >= umbral)[0]
    if len(activos) == 0:
        return audio

    inicio = max(0, activos[0] * ventana - margen)
    fin = min(len(audio), (activos[-1] + 1) * ventana + margen)
    return audio[inicio:fin]


def grabar_audio_asistente(
    duracion_segundos: float = DURACION_GRABACION,
    frecuencia: int = FRECUENCIA_MUESTREO,
    *,
    aviso: bool = True,
) -> np.ndarray:
    """Graba y prepara audio para prediccion en vivo."""
    if aviso:
        print("\a", end="", flush=True)  # beep: empieza a grabar ya
    audio = grabar_audio(duracion_segundos, frecuencia)
    return preparar_audio_en_vivo(audio, frecuencia)


def guardar_audio(
    ruta: Path,
    audio: np.ndarray,
    frecuencia: int = FRECUENCIA_MUESTREO,
) -> None:
    """Guarda audio mono en formato WAV PCM 16 bits."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sf.write(ruta, audio, frecuencia, subtype="PCM_16")


def _mostrar_info_grabacion(
    hablante: str,
    intencion: str,
    frase: str,
    numero_frase: int,
    repeticion: int,
    total_repeticiones: int,
    ruta_salida: Path,
) -> None:
    """Imprime en consola toda la informacion de la toma actual."""
    sugerencia = SUGERENCIAS_VARIACION.get(repeticion, "natural")

    print("\n" + "=" * 60)
    print(f"  Hablante             : {hablante}")
    print(f"  Intencion actual     : {intencion}")
    print(f'  Frase exacta         : "{frase}"')
    print(f"  Numero de frase      : {numero_frase:02d} / {FRASES_POR_INTENCION}")
    print(f"  Repeticion actual    : {repeticion:02d} / {total_repeticiones:02d}")
    print(f"  Total repeticiones   : {total_repeticiones}")
    print(f"  Variacion sugerida   : rep{repeticion:02d} -> {sugerencia}")
    print(f"  Ruta de salida       : {ruta_salida}")
    print(f"  Formato              : WAV mono {FRECUENCIA_MUESTREO} Hz, {DURACION_GRABACION}s")
    print("=" * 60)


def grabar_repeticion(
    hablante: str,
    intencion: str,
    numero_frase: int,
    frase: str,
    repeticion: int,
    carpeta_dataset: Path = CARPETA_DATASET,
) -> None:
    """Graba una sola repeticion de una frase (sobrescribe el WAV si existe)."""
    carpeta_intencion = carpeta_dataset / intencion
    carpeta_intencion.mkdir(parents=True, exist_ok=True)

    nombre = _nombre_archivo(intencion, hablante, numero_frase, repeticion)
    ruta_salida = carpeta_intencion / nombre

    _mostrar_info_grabacion(
        hablante=hablante,
        intencion=intencion,
        frase=frase,
        numero_frase=numero_frase,
        repeticion=repeticion,
        total_repeticiones=REPETICIONES_POR_FRASE,
        ruta_salida=ruta_salida,
    )

    input("  Presiona Enter para grabar...")
    print(f"  Grabando {DURACION_GRABACION}s...")
    audio = grabar_audio()
    guardar_audio(ruta_salida, audio)
    print(f"  [ok] Guardado: {ruta_salida}")


def grabar_frase(
    hablante: str,
    intencion: str,
    numero_frase: int,
    frase: str,
    carpeta_dataset: Path = CARPETA_DATASET,
    repeticiones: int = REPETICIONES_POR_FRASE,
    repeticion_unica: int | None = None,
) -> None:
    """Graba repeticiones de una frase para un hablante e intencion."""
    carpeta_intencion = carpeta_dataset / intencion
    carpeta_intencion.mkdir(parents=True, exist_ok=True)

    if repeticion_unica is not None:
        grabar_repeticion(
            hablante,
            intencion,
            numero_frase,
            frase,
            repeticion_unica,
            carpeta_dataset,
        )
        return

    for repeticion in range(1, repeticiones + 1):
        grabar_repeticion(
            hablante,
            intencion,
            numero_frase,
            frase,
            repeticion,
            carpeta_dataset,
        )


def grabar_intencion(
    hablante: str,
    intencion: str,
    carpeta_dataset: Path = CARPETA_DATASET,
) -> None:
    """Graba las 5 frases x 5 repeticiones de una intencion para un hablante."""
    if intencion not in FRASES:
        raise ValueError(f"Intencion desconocida: {intencion}")

    frases = FRASES[intencion]
    audios_intencion = len(frases) * REPETICIONES_POR_FRASE

    print("\n" + "#" * 60)
    print(f"  HABLANTE: {hablante} | INTENCION: {intencion}")
    print(f"  Frases a grabar    : {len(frases)}")
    print(f"  Repeticiones/frase : {REPETICIONES_POR_FRASE}")
    print(f"  Audios esta sesion : {audios_intencion}")
    print("#" * 60)

    for indice, frase in enumerate(frases, start=1):
        grabar_frase(hablante, intencion, indice, frase, carpeta_dataset)

    print(f"\n  [ok] {hablante} / '{intencion}' completado ({audios_intencion} audios).")


def grabar_dataset(
    hablante: str,
    intenciones: list[str] | None = None,
    carpeta_dataset: Path = CARPETA_DATASET,
) -> None:
    """Graba el dataset completo para un hablante."""
    if intenciones is None:
        intenciones = INTENCIONES

    total = _audios_esperados_por_hablante(intenciones)

    print(f"\n  Intenciones          : {', '.join(intenciones)}")
    print(f"  Frases por intencion : {FRASES_POR_INTENCION}")
    print(f"  Repeticiones/frase   : {REPETICIONES_POR_FRASE}")
    print(f"  Audios a grabar      : {total}")
    print(f"  Formato              : WAV mono {FRECUENCIA_MUESTREO} Hz, {DURACION_GRABACION}s")
    print(f"  Carpeta destino      : {carpeta_dataset}")

    for intencion in intenciones:
        grabar_intencion(hablante, intencion, carpeta_dataset)

    print("\n" + "=" * 60)
    print(f"  GRABACION COMPLETADA — {hablante}: {total} audios")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graba audios WAV para entrenar el clasificador HMM."
    )
    parser.add_argument(
        "--intencion",
        choices=INTENCIONES,
        help="Grabar solo una intencion (por defecto: todas)",
    )
    parser.add_argument(
        "--hablante",
        choices=HABLANTES,
        help="Omitir pregunta inicial (emmanuel o elioth)",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Grabar sin preguntar si emmanuel ya tiene audios",
    )
    parser.add_argument(
        "--frase",
        type=int,
        choices=range(1, FRASES_POR_INTENCION + 1),
        metavar="N",
        help=f"Grabar solo la frase N (1-{FRASES_POR_INTENCION}); requiere --intencion",
    )
    parser.add_argument(
        "--repeticion",
        type=int,
        choices=range(1, REPETICIONES_POR_FRASE + 1),
        metavar="N",
        help=f"Grabar solo la repeticion N (1-{REPETICIONES_POR_FRASE}); requiere --frase",
    )
    args = parser.parse_args()

    if args.repeticion and not args.frase:
        parser.error("--repeticion requiere --frase")
    if args.frase and not args.intencion:
        parser.error("--frase requiere --intencion")

    intenciones = [args.intencion] if args.intencion else INTENCIONES

    if args.hablante:
        hablante = args.hablante
    else:
        hablante = seleccionar_hablante()

    if not args.forzar and not verificar_inicio_grabacion(hablante, intenciones):
        print("\n  Grabacion cancelada. Tus audios anteriores se conservan.")
        return

    if args.frase:
        intencion = args.intencion
        frase = FRASES[intencion][args.frase - 1]
        grabar_frase(
            hablante,
            intencion,
            args.frase,
            frase,
            repeticion_unica=args.repeticion,
        )
    elif args.intencion:
        grabar_intencion(hablante, args.intencion)
    else:
        grabar_dataset(hablante)


if __name__ == "__main__":
    main()
